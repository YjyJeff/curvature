//! Streaming limit operator

use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Duration;

use crate::common::client_context::ClientContext;
use crate::common::profiler::ScopedTimerGuard;
use crate::exec::physical_operator::ext_traits::RegularOperatorExt;
use crate::exec::physical_operator::metric::{MetricsSet, Time};
use crate::exec::physical_operator::utils::downcast_mut_local_state;
use crate::exec::physical_operator::{
    GlobalOperatorState, LocalOperatorState, OperatorExecStatus, OperatorResult, PhysicalOperator,
    StateStringify, Stringify, impl_sink_for_non_sink, impl_source_for_non_source,
    use_types_for_impl_sink_for_non_sink, use_types_for_impl_source_for_non_source,
};
use curvature_procedural_macro::MetricsSetBuilder;
use data_block::block::DataBlock;
use data_block::types::LogicalType;
use snafu::{Snafu, ensure};

use_types_for_impl_sink_for_non_sink!();
use_types_for_impl_source_for_non_source!();

/// Error returned by creating the [`StreamingLimit`]
#[derive(Debug, Snafu)]
#[snafu(display("Sum of the limit `{limit}` and offset `{offset}` overflow"))]
pub struct StreamingLimitError {
    /// Limit argument
    limit: NonZeroU64,
    /// offset argument
    offset: u64,
}

/// Streaming limit is a normal operator. It will globally limit the number of rows passed
/// through it
#[derive(Debug)]
pub struct StreamingLimit {
    input: [Arc<dyn PhysicalOperator>; 1],
    output_types: Vec<LogicalType>,

    /// Number of rows that can be passed through this operator
    limit: NonZeroU64,
    /// From which offset, the rows can be passed through
    offset: u64,

    /// upper_bound = limit + offset
    upper_bound: u64,

    metrics: StreamingLimitMetrics,
}

#[derive(Debug, MetricsSetBuilder, Default)]
struct StreamingLimitMetrics {
    execute_time: Time,
}

impl StreamingLimit {
    /// Try to create a new [`StreamingLimit`]
    pub fn try_new(
        input: Arc<dyn PhysicalOperator>,
        limit: NonZeroU64,
        offset: u64,
    ) -> Result<Self, StreamingLimitError> {
        ensure!(
            limit.get().checked_add(offset).is_some(),
            StreamingLimitSnafu { limit, offset }
        );

        let output_types = input.output_types().to_owned();
        Ok(Self {
            input: [input],
            output_types,
            limit,
            offset,
            upper_bound: limit.get() + offset,
            metrics: StreamingLimitMetrics::default(),
        })
    }
}

/// Global operator state for [`StreamingLimit`]
#[derive(Debug)]
pub struct StreamingLimitGlobalState {
    rows_count: AtomicU64,
}

impl StateStringify for StreamingLimitGlobalState {
    fn name(&self) -> &'static str {
        "StreamingLimitGlobalState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl GlobalOperatorState for StreamingLimitGlobalState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Local operator state for [`StreamingLimit`]
#[derive(Debug, Default)]
pub struct StreamingLimitLocalState {
    execute_time: Duration,
}

impl StateStringify for StreamingLimitLocalState {
    fn name(&self) -> &'static str {
        "StreamingLimitLocalState"
    }
    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl LocalOperatorState for StreamingLimitLocalState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Stringify for StreamingLimit {
    fn name(&self) -> &'static str {
        "StreamingLimit"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "StreamingLimit: limit={}, offset={}",
            self.limit, self.offset
        )
    }
}

impl PhysicalOperator for StreamingLimit {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn output_types(&self) -> &[LogicalType] {
        &self.output_types
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.input
    }

    fn metrics(&self) -> MetricsSet {
        self.metrics.metrics_set()
    }

    // Regular Operator

    fn is_regular_operator(&self) -> bool {
        true
    }

    fn execute(
        &self,
        input: &DataBlock,
        output: &mut DataBlock,
        global_state: &dyn GlobalOperatorState,
        local_state: &mut dyn LocalOperatorState,
    ) -> OperatorResult<OperatorExecStatus> {
        let local_state =
            downcast_mut_local_state!(self, local_state, StreamingLimitLocalState, OPERATOR);
        let _guard = ScopedTimerGuard::new(&mut local_state.execute_time);

        let input_len = input.len() as u64;
        let rows_count = {
            let global_state = self.downcast_ref_global_operator_state(global_state);
            global_state.rows_count.fetch_add(input_len, Relaxed)
        };

        if rows_count >= self.upper_bound {
            return Ok(OperatorExecStatus::Finished);
        }

        if rows_count >= self.offset {
            if rows_count + input_len <= self.upper_bound {
                // The full data block can be passed through
                unsafe {
                    output.reference(input).expect("Pipeline executor guarantees `StreamingLimit`'s output have same schema with its input");
                }
            } else {
                let start = 0_usize;
                let len = (self.upper_bound - rows_count) as usize;

                unsafe {
                    output.copy(input, start, len).expect("Pipeline executor guarantees `StreamingLimit`'s output have same schema with its input");
                }
            }
        } else if rows_count + input_len > self.offset {
            // Fetch partial rows
            let start = self.offset - rows_count;
            let len = if rows_count + input_len <= self.upper_bound {
                input_len - start
            } else {
                self.limit.get()
            } as usize;

            unsafe {
                output.copy(input, start as usize, len).expect("Pipeline executor guarantees `StreamingLimit`'s output have same schema with its input");
            }
        } else {
            unsafe {
                output.clear();
            }
        }

        Ok(OperatorExecStatus::NeedMoreInput)
    }

    fn global_operator_state(&self, _client_ctx: &ClientContext) -> Arc<dyn GlobalOperatorState> {
        Arc::new(StreamingLimitGlobalState {
            rows_count: AtomicU64::new(0),
        })
    }

    fn local_operator_state(&self) -> Box<dyn LocalOperatorState> {
        Box::new(StreamingLimitLocalState::default())
    }

    fn merge_local_operator_metrics(&self, local_state: &dyn LocalOperatorState) {
        let local_state = self.downcast_ref_local_operator_state(local_state);
        tracing::debug!("StreamingLimit takes: {:?}", local_state.execute_time);

        self.metrics
            .execute_time
            .add_duration(local_state.execute_time);
    }

    impl_source_for_non_source!();
    impl_sink_for_non_sink!();
}

impl RegularOperatorExt for StreamingLimit {
    type GlobalOperatorState = StreamingLimitGlobalState;
    type LocalOperatorState = StreamingLimitLocalState;
}

#[cfg(test)]
mod tests {
    use data_block::{
        array::{ArrayImpl, Int32Array},
        types::Array,
    };

    use super::*;
    use crate::common::client_context::tests::mock_client_context;
    use crate::exec::physical_operator::table_scan::empty_table_scan::EmptyTableScan;

    #[test]
    fn test_streaming_limit() {
        let streaming_limit = StreamingLimit::try_new(
            Arc::new(EmptyTableScan::new(vec![LogicalType::Integer])),
            NonZeroU64::new(4).unwrap(),
            3,
        )
        .unwrap();

        let global_state = streaming_limit.global_operator_state(&mock_client_context());
        let mut local_state = streaming_limit.local_operator_state();

        let input = DataBlock::try_new(
            vec![ArrayImpl::Int32(Int32Array::from_values_iter([1, 2]))],
            2,
        )
        .unwrap();
        let mut output = DataBlock::with_logical_types(vec![LogicalType::Integer]);
        let status = streaming_limit
            .execute(&input, &mut output, &*global_state, &mut *local_state)
            .unwrap();
        assert_eq!(status, OperatorExecStatus::NeedMoreInput);
        assert_eq!(output.arrays()[0].len(), 0);

        let status = streaming_limit
            .execute(&input, &mut output, &*global_state, &mut *local_state)
            .unwrap();
        assert_eq!(status, OperatorExecStatus::NeedMoreInput);
        let array: &Int32Array = (&output.arrays()[0]).try_into().unwrap();
        assert_eq!(array.values_iter().collect::<Vec<_>>(), [2]);

        let status = streaming_limit
            .execute(&input, &mut output, &*global_state, &mut *local_state)
            .unwrap();
        assert_eq!(status, OperatorExecStatus::NeedMoreInput);
        let array: &Int32Array = (&output.arrays()[0]).try_into().unwrap();
        assert_eq!(array.values_iter().collect::<Vec<_>>(), [1, 2]);

        let status = streaming_limit
            .execute(&input, &mut output, &*global_state, &mut *local_state)
            .unwrap();
        assert_eq!(status, OperatorExecStatus::NeedMoreInput);
        let array: &Int32Array = (&output.arrays()[0]).try_into().unwrap();
        assert_eq!(array.values_iter().collect::<Vec<_>>(), [1]);

        let status = streaming_limit
            .execute(&input, &mut output, &*global_state, &mut *local_state)
            .unwrap();
        assert_eq!(status, OperatorExecStatus::Finished);
    }
}
