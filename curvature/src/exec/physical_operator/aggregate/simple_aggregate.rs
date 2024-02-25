//! Aggregation without group by keys

use std::sync::Arc;

use data_block::block::DataBlock;
use data_block::types::LogicalType;

use crate::common::client_context::ClientContext;
use crate::error::SendableError;
use crate::exec::physical_expr::function::aggregate::{
    AggregationFunction, AggregationFunctionList,
};
use crate::exec::physical_operator::utils::{
    impl_regular_for_non_regular, use_types_for_impl_regular_for_non_regular,
};
use crate::exec::physical_operator::{
    GlobalSinkState, GlobalSourceState, LocalSinkState, LocalSourceState, OperatorResult,
    ParallelismDegree, PhysicalOperator, SinkExecStatus, SourceExecStatus, StateStringify,
    Stringify,
};
use snafu::ResultExt;
use_types_for_impl_regular_for_non_regular!();

#[allow(missing_docs)]
#[derive(Debug)]
pub enum SimpleAggregateError {}

type Result<T> = std::result::Result<T, SimpleAggregateError>;

/// Aggregation without group by keys
#[derive(Debug)]
pub struct SimpleAggregate {
    input: Vec<Arc<dyn PhysicalOperator>>,
    agg_func_list: AggregationFunctionList,
    output_types: Vec<LogicalType>,
}

impl SimpleAggregate {
    /// FIXME: lots of check
    pub fn try_new(
        input: Arc<dyn PhysicalOperator>,
        agg_funcs: Vec<Arc<dyn AggregationFunction>>,
    ) -> Result<Self> {
        let output_types = agg_funcs
            .iter()
            .map(|agg_func| agg_func.return_type())
            .collect();
        Ok(Self {
            input: vec![input],
            agg_func_list: AggregationFunctionList::new(agg_funcs),
            output_types,
        })
    }
}

/// Global source state of the [`SimpleAggregate`]
#[derive(Debug)]
pub struct SimpleAggregateGlobalSourceState;

impl StateStringify for SimpleAggregateGlobalSourceState {
    fn name(&self) -> &'static str {
        "SimpleAggregateGlobalSourceState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl GlobalSourceState for SimpleAggregateGlobalSourceState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
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
    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn read_data(&mut self, _output: &mut DataBlock) -> OperatorResult<SourceExecStatus> {
        todo!()
    }
}

/// Global sink state of the [`SimpleAggregate`]
#[derive(Debug)]
pub struct SimpleAggregateGlobalSinkState;

impl StateStringify for SimpleAggregateGlobalSinkState {
    fn name(&self) -> &'static str {
        "SimpleAggregateGlobalSinkState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl GlobalSinkState for SimpleAggregateGlobalSinkState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Local sink state of the [`SimpleAggregate`]
#[derive(Debug)]
pub struct SimpleAggregateLocalSinkState;

impl StateStringify for SimpleAggregateLocalSinkState {
    fn name(&self) -> &'static str {
        "SimpleAggregateLocalSinkState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl LocalSinkState for SimpleAggregateLocalSourceState {
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

    /// FIXME: display
    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SimpleAggregate")
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
        &self.input
    }

    impl_regular_for_non_regular!();

    // Source
    fn is_source(&self) -> bool {
        true
    }

    fn source_parallelism_degree(
        &self,
        _global_state: &dyn GlobalSourceState,
    ) -> OperatorResult<ParallelismDegree> {
        todo!()
    }

    fn read_data(
        &self,
        _output: &mut data_block::block::DataBlock,
        _global_state: &dyn GlobalSourceState,
        _local_state: &mut dyn LocalSourceState,
    ) -> OperatorResult<SourceExecStatus> {
        todo!()
    }

    /// FIXME: Correct state
    fn global_source_state(
        &self,
        _client_ctx: &ClientContext,
    ) -> OperatorResult<Arc<dyn GlobalSourceState>> {
        Ok(Arc::new(SimpleAggregateGlobalSourceState))
    }

    fn local_source_state(
        &self,
        _global_state: &dyn GlobalSourceState,
    ) -> OperatorResult<Box<dyn LocalSourceState>> {
        todo!()
    }

    fn progress(&self, _global_state: &dyn GlobalSourceState) -> OperatorResult<f64> {
        todo!()
    }

    // Sink

    fn is_sink(&self) -> bool {
        true
    }

    fn is_parallel_sink(&self) -> OperatorResult<bool> {
        Ok(true)
    }

    fn write_data(
        &self,
        _input: &data_block::block::DataBlock,
        _global_state: &dyn GlobalSinkState,
        _local_state: &mut dyn LocalSinkState,
    ) -> OperatorResult<SinkExecStatus> {
        todo!()
    }

    fn finish_local_sink(
        &self,
        _global_state: &dyn GlobalSinkState,
        _local_state: &mut dyn LocalSinkState,
    ) -> OperatorResult<()> {
        todo!()
    }

    unsafe fn finalize_sink(&self, _global_state: &dyn GlobalSinkState) -> OperatorResult<()> {
        todo!()
    }

    /// FIXME: Correct state
    fn global_sink_state(
        &self,
        _client_ctx: &ClientContext,
    ) -> OperatorResult<Arc<dyn GlobalSinkState>> {
        Ok(Arc::new(SimpleAggregateGlobalSinkState))
    }

    fn local_sink_state(
        &self,
        _global_state: &dyn GlobalSinkState,
    ) -> OperatorResult<Box<dyn LocalSinkState>> {
        todo!()
    }
}
