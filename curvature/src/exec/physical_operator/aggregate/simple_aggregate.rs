//! Aggregation without group by keys

use std::sync::Arc;
use std::time::Duration;

use curvature_procedural_macro::MetricsSetBuilder;
use data_block::array::ArrayImpl;
use data_block::block::{DataBlock, MutateArrayError};
use data_block::types::LogicalType;
use parking_lot::Mutex;
use snafu::{ensure, ResultExt, Snafu};

use crate::common::client_context::ClientContext;
use crate::common::profiler::ScopedTimerGuard;
use crate::exec::physical_expr::function::aggregate::{
    AggregationFunctionExpr, AggregationFunctionList, AggregationStatesPtr,
};
use crate::exec::physical_operator::metric::{Count, MetricsSet, Time};
use crate::exec::physical_operator::utils::{
    downcast_mut_local_state, impl_regular_for_non_regular,
    use_types_for_impl_regular_for_non_regular,
};
use crate::exec::physical_operator::{
    DummyGlobalSinkState, DummyGlobalSourceState, FinalizeSinkSnafu, GlobalSinkState,
    GlobalSourceState, LocalSinkState, LocalSourceState, OperatorResult, ParallelismDegree,
    PhysicalOperator, ReadDataSnafu, SinkExecStatus, SourceExecStatus, StateStringify, Stringify,
    WriteDataSnafu,
};
use_types_for_impl_regular_for_non_regular!();

#[allow(missing_docs)]
#[derive(Debug, Snafu)]
pub enum SimpleAggregateError {
    #[snafu(display("`FieldRef` with index `{ref_index}` is out or range, the input only has `{input_size}` fields"))]
    FieldRefOutOfRange { ref_index: usize, input_size: usize },
    #[snafu(display(
        "TypeMismatch: `FieldRef` with index `{ref_index}` has logical type `{:?}`, however the field in the input has logical type `{:?}`",
        ref_type,
        input_type
    ))]
    FieldRefTypeMismatch {
        ref_index: usize,
        ref_type: LogicalType,
        input_type: LogicalType,
    },
    #[snafu(display("Aggregation functions passed to SimpleAggregate is empty. SimpleAggregate needs at least one aggregation function"))]
    EmptyAggregationFuncs,
}

type Result<T> = std::result::Result<T, SimpleAggregateError>;

/// Aggregation without group by keys
#[derive(Debug)]
pub struct SimpleAggregate {
    children: [Arc<dyn PhysicalOperator>; 1],
    agg_func_list: AggregationFunctionList,
    output_types: Vec<LogicalType>,
    global_states: Mutex<GlobalState>,
    payloads_indexes: Vec<usize>,
    metrics: SimpleAggregateMetrics,
}

#[derive(Debug)]
struct GlobalState {
    ptr: AggregationStatesPtr,
    have_been_read: bool,
}

impl SimpleAggregate {
    /// Try to create a new SimpleAggregate
    pub fn try_new(
        input: Arc<dyn PhysicalOperator>,
        agg_funcs: Vec<AggregationFunctionExpr<'_>>,
    ) -> Result<Self> {
        let input_logical_types = input.output_types();
        let input_size = input_logical_types.len();

        let mut payloads_indexes = Vec::with_capacity(agg_funcs.len());

        ensure!(!agg_funcs.is_empty(), EmptyAggregationFuncsSnafu {});

        let output_types = agg_funcs
            .iter()
            .map(|agg_func| agg_func.func.return_type())
            .collect();

        agg_funcs.iter().try_for_each(|agg_func| {
            agg_func.payloads.iter().try_for_each(|field_ref| {
                ensure!(
                    field_ref.field_index < input_size,
                    FieldRefOutOfRangeSnafu {
                        ref_index: field_ref.field_index,
                        input_size
                    }
                );

                ensure!(
                    field_ref.output_type == input_logical_types[field_ref.field_index],
                    FieldRefTypeMismatchSnafu {
                        ref_index: field_ref.field_index,
                        ref_type: field_ref.output_type.clone(),
                        input_type: input_logical_types[field_ref.field_index].clone()
                    }
                );

                payloads_indexes.push(field_ref.field_index);
                Ok::<_, SimpleAggregateError>(())
            })?;

            Ok(())
        })?;

        let agg_func_list = AggregationFunctionList::new(
            agg_funcs
                .into_iter()
                .map(|agg_func| agg_func.func)
                .collect(),
        );

        Ok(Self {
            children: [input],
            global_states: Mutex::new(GlobalState {
                ptr: agg_func_list.alloc_states(),
                have_been_read: false,
            }),
            agg_func_list,
            output_types,
            payloads_indexes,
            metrics: SimpleAggregateMetrics::default(),
        })
    }
}

impl Drop for SimpleAggregate {
    fn drop(&mut self) {
        let global_state = self.global_states.lock();
        unsafe { self.agg_func_list.dealloc_states(global_state.ptr) }
    }
}

#[derive(Debug, Default, MetricsSetBuilder)]
struct SimpleAggregateMetrics {
    /// The number of rows written to the operator
    num_rows: Count,
    /// Time used to perform aggregation
    agg_time: Time,
}

impl SimpleAggregateMetrics {
    fn combine_local_metrics(&self, local_metrics: &LocalSinkMetrics) {
        self.num_rows.add(local_metrics.num_rows);
        self.agg_time.add_duration(local_metrics.agg_time);
    }
}

/// Local source state of the [`SimpleAggregate`]
#[derive(Debug)]
pub struct SimpleAggregateLocalSourceState;

impl StateStringify for SimpleAggregateLocalSourceState {
    fn name(&self) -> &'static str {
        "SimpleAggregateLocalSourceState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl LocalSourceState for SimpleAggregateLocalSourceState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Local sink state of the [`SimpleAggregate`]
#[derive(Debug)]
pub struct SimpleAggregateLocalSinkState {
    /// Used for drop. The lifetime of the agg_func_list is the lifetime of the
    /// physical operator that create this local sink state, which guarantees out-lives
    /// this state
    agg_func_list: *const AggregationFunctionList,

    /// Pointer to the dynamic aggregation states
    states_ptr: AggregationStatesPtr,

    /// Note: The 'static lifetime is fake.....
    payloads: Vec<&'static ArrayImpl>,

    /// local metrics
    metrics: LocalSinkMetrics,
}

#[derive(Debug, Default)]
struct LocalSinkMetrics {
    /// The number of rows to perform aggregation
    num_rows: u64,
    /// The time used to perform aggregation
    agg_time: Duration,
}

impl Drop for SimpleAggregateLocalSinkState {
    fn drop(&mut self) {
        unsafe { (*self.agg_func_list).dealloc_states(self.states_ptr) }
    }
}

impl StateStringify for SimpleAggregateLocalSinkState {
    fn name(&self) -> &'static str {
        "SimpleAggregateLocalSinkState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl LocalSinkState for SimpleAggregateLocalSinkState {
    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Stringify for SimpleAggregate {
    fn name(&self) -> &'static str {
        "SimpleAggregate"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SimpleAggregate: aggregation_functions={}",
            self.agg_func_list
        )
    }
}

impl PhysicalOperator for SimpleAggregate {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn output_types(&self) -> &[data_block::types::LogicalType] {
        &self.output_types
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn metrics(&self) -> MetricsSet {
        self.metrics.metrics_set()
    }

    impl_regular_for_non_regular!();

    // Source
    fn is_source(&self) -> bool {
        true
    }

    fn source_parallelism_degree(
        &self,
        _global_state: &dyn GlobalSourceState,
    ) -> ParallelismDegree {
        ParallelismDegree::MIN
    }

    fn read_data(
        &self,
        output: &mut DataBlock,
        _global_state: &dyn GlobalSourceState,
        _local_state: &mut dyn LocalSourceState,
    ) -> OperatorResult<SourceExecStatus> {
        let mut global_state = self.global_states.lock();
        if global_state.have_been_read {
            Ok(SourceExecStatus::Finished)
        } else {
            global_state.have_been_read = true;
            let mutate_guard = output.mutate_arrays(1);
            // SAFETY: Global state is not read, its ptr is valid

            if let Err(e) = mutate_guard.mutate(|output| unsafe {
                self.agg_func_list.take_states(&[global_state.ptr], output)
            }) {
                match e {
                    MutateArrayError::Inner { source } => {
                        Err(Box::new(source) as _).context(ReadDataSnafu)?
                    }
                    MutateArrayError::Length { inner } => {
                        panic!(
                            "SimpleAggregate should produce array with length 1. Error: `{}`",
                            inner
                        )
                    }
                }
            }

            Ok(SourceExecStatus::HaveMoreOutput)
        }
    }

    fn global_source_state(&self, _client_ctx: &ClientContext) -> Arc<dyn GlobalSourceState> {
        Arc::new(DummyGlobalSourceState)
    }

    fn local_source_state(
        &self,
        _global_state: &dyn GlobalSourceState,
    ) -> Box<dyn LocalSourceState> {
        Box::new(SimpleAggregateLocalSourceState)
    }

    fn merge_local_source_metrics(&self, _local_state: &dyn LocalSourceState) {}

    fn progress(&self, _global_state: &dyn GlobalSourceState) -> f64 {
        1.0
    }

    // Sink

    fn is_sink(&self) -> bool {
        true
    }

    fn write_data(
        &self,
        input: &DataBlock,
        _global_state: &dyn GlobalSinkState,
        local_state: &mut dyn LocalSinkState,
    ) -> OperatorResult<SinkExecStatus> {
        debug_assert_eq!(input.num_arrays(), self.children[0].output_types().len());
        debug_assert!(input
            .arrays()
            .iter()
            .map(|array| array.logical_type())
            .zip(self.children[0].output_types())
            .all(|(input, output)| input == output));

        let local_state =
            downcast_mut_local_state!(self, local_state, SimpleAggregateLocalSinkState, SINK);

        local_state.metrics.num_rows += input.len() as u64;
        let _guard = ScopedTimerGuard::new(&mut local_state.metrics.agg_time);

        self.payloads_indexes.iter().for_each(|&index| unsafe{
            let payload = input
                    .get_array(index)
                    .expect("Index in payloads_indexes is guaranteed to be valid, checked by the `try_new` constructor");

            local_state.payloads.push(&*(payload as *const _))
        });

        // SAFETY:
        // - all of the payloads fetched from input such that they all have same length
        // - ptr is guaranteed to be valid
        unsafe {
            self.agg_func_list
                .batch_update_states(input.len(), &local_state.payloads, local_state.states_ptr)
                .boxed()
                .context(WriteDataSnafu)?;
        }

        // Clear the payloads to avoid use after free
        local_state.payloads.clear();

        Ok(SinkExecStatus::NeedMoreInput)
    }

    fn combine_sink(
        &self,
        _global_state: &dyn GlobalSinkState,
        local_state: &mut dyn LocalSinkState,
    ) -> OperatorResult<()> {
        let local_state =
            downcast_mut_local_state!(self, local_state, SimpleAggregateLocalSinkState, SINK);

        // Combine metrics
        tracing::debug!(
            "Aggregate {} rows into 1 row in: `{:?}`",
            local_state.metrics.num_rows,
            local_state.metrics.agg_time
        );
        self.metrics.combine_local_metrics(&local_state.metrics);

        let global_state = self.global_states.lock();

        // SAFETY: both pointers are valid and have length 1
        unsafe {
            self.agg_func_list
                .combine_states(&[local_state.states_ptr], &[global_state.ptr])
                .boxed()
                .context(FinalizeSinkSnafu)
        }
    }

    unsafe fn finalize_sink(&self, _global_state: &dyn GlobalSinkState) -> OperatorResult<()> {
        Ok(())
    }

    fn global_sink_state(&self, _client_ctx: &ClientContext) -> Arc<dyn GlobalSinkState> {
        Arc::new(DummyGlobalSinkState)
    }

    fn local_sink_state(&self, _global_state: &dyn GlobalSinkState) -> Box<dyn LocalSinkState> {
        Box::new(SimpleAggregateLocalSinkState {
            agg_func_list: &self.agg_func_list,
            states_ptr: self.agg_func_list.alloc_states(),
            payloads: Vec::new(),
            metrics: LocalSinkMetrics::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use data_block::array::{Int32Array, StringArray};

    use crate::common::client_context::tests::mock_client_context;
    use crate::exec::physical_expr::field_ref::FieldRef;
    use crate::exec::physical_expr::function::aggregate::count::Count;
    use crate::exec::physical_expr::function::aggregate::min_max::Min;
    use crate::exec::physical_operator::table_scan::empty_table_scan::EmptyTableScan;

    use super::*;

    #[test]
    fn test_simple_aggregate() {
        let data_block = DataBlock::try_new(
            vec![
                ArrayImpl::Int32(Int32Array::from_iter([Some(-10), None, None])),
                ArrayImpl::String(StringArray::from_iter([
                    Some("curvature"),
                    Some("curve"),
                    None,
                ])),
            ],
            3,
        )
        .unwrap();

        let input = Arc::new(EmptyTableScan::new(vec![
            LogicalType::Integer,
            LogicalType::VarChar,
        ]));

        let count_payloads = [FieldRef::new(0, LogicalType::Integer, "f0".to_string())];
        let min_payloads = [FieldRef::new(1, LogicalType::VarChar, "f1".to_string())];
        let agg_funcs = vec![
            AggregationFunctionExpr::try_new(
                &count_payloads,
                Arc::new(Count::<false>::new(LogicalType::Integer)),
            )
            .unwrap(),
            AggregationFunctionExpr::try_new(
                &min_payloads,
                Arc::new(Min::<StringArray>::try_new(LogicalType::VarChar).unwrap()),
            )
            .unwrap(),
        ];

        let aggregate = SimpleAggregate::try_new(input, agg_funcs).unwrap();

        let client_ctx = mock_client_context();

        // Sink phase
        {
            let global_state = aggregate.global_sink_state(&client_ctx);
            let mut local_state = aggregate.local_sink_state(&*global_state);

            aggregate
                .write_data(&data_block, &*global_state, &mut *local_state)
                .unwrap();

            aggregate
                .combine_sink(&*global_state, &mut *local_state)
                .unwrap();

            unsafe {
                aggregate.finalize_sink(&*global_state).unwrap();
            }
        }

        // Read phase
        {
            let global_state = aggregate.global_source_state(&client_ctx);
            let mut local_state = aggregate.local_source_state(&*global_state);
            let mut output = DataBlock::with_logical_types(vec![
                LogicalType::UnsignedBigInt,
                LogicalType::VarChar,
            ]);
            let source_exec_status = aggregate
                .read_data(&mut output, &*global_state, &mut *local_state)
                .unwrap();

            assert!(matches!(
                source_exec_status,
                SourceExecStatus::HaveMoreOutput
            ));

            let expected = expect_test::expect![[r#"
                ┌────────────────┬───────────┐
                │ UnsignedBigInt │ VarChar   │
                ├────────────────┼───────────┤
                │ 1              │ curvature │
                └────────────────┴───────────┘"#]];
            expected.assert_eq(&output.to_string());

            let source_exec_status = aggregate
                .read_data(&mut output, &*global_state, &mut *local_state)
                .unwrap();
            assert!(matches!(source_exec_status, SourceExecStatus::Finished));
        }
    }
}
