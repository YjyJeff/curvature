//! Executor that execute a single pipeline

use data_block::block::DataBlock;
use snafu::{ensure, ResultExt, Snafu};
use std::convert::identity;

use super::{Pipeline, Sink, SinkTrait};
use crate::common::client_context::ClientContext;
use crate::exec::physical_operator::{
    LocalOperatorState, LocalSourceState, OperatorError, OperatorExecStatus, SinkExecStatus,
    SourceExecStatus,
};

type OperatorIndex = usize;

#[allow(missing_docs)]
#[derive(Debug, Snafu)]
pub enum PipelineExecutorError {
    #[snafu(display("Query is cancelled"))]
    Cancelled,
    #[snafu(display("Failed to read_data from Source"))]
    ExecuteSource { source: OperatorError },
    #[snafu(display("Failed to execute the regular operator"))]
    ExecuteOperator { source: OperatorError },
    #[snafu(display("Failed to write_data to Sink"))]
    ExecuteSink { source: OperatorError },
    #[snafu(display("Failed to execute merge_sink"))]
    MergeSink { source: OperatorError },
    #[snafu(display(
        "Regular operator `{op}` returns `HaveMoreOutput` but the output is empty.
         It breaks the contract: \"When `HaveMoreOutput` is returned, output can not be empty\""
    ))]
    HaveMoreOutputButEmpty { op: &'static str },
}

type Result<T> = std::result::Result<T, PipelineExecutorError>;

/// Executor that execute the pipeline. Each thread will create a executor to execute
/// the pipeline. The executor tries to read the data from the `Source` operator,
/// pass the data through the `Regular` operators and finally writes the data to
/// the `Sink` operator.
#[derive(Debug)]
pub struct PipelineExecutor<'a, S: SinkTrait> {
    /// Pipeline that should be executed
    pipeline: &'a Pipeline<S>,
    /// Local source state of the source in pipeline
    local_source_state: Box<dyn LocalSourceState>,
    /// Data blocks that contains the output of the source/regular operators
    /// Index 0: source produce result to it
    /// Index i: operator i take it as input and write result to i + 1
    intermediate_blocks: Vec<DataBlock>,
    /// Local states of the regular operators
    local_operator_states: Vec<Box<dyn LocalOperatorState>>,
    /// Local sink state of the sink in pipeline
    local_sink_state: S::LocalSinkState,
    /// The operators that are not yet finished executing and have data remaining. We need to view
    /// it as source and fetch more result from it. If the stack of in_process_operators is
    /// empty, we fetch from the source instead
    in_process_operators: Vec<OperatorIndex>,

    /// Client context of the query execution
    client_ctx: &'a ClientContext,
}

macro_rules! ensure_not_cancelled {
    ($self:ident) => {
        ensure!(!$self.client_ctx.is_cancelled(), CancelledSnafu);
    };
}

/// Returned by the execute_regular_operators function to control how to execute
/// the pipeline after executing the regular operators
enum RegularOperatorsExecutionControl {
    /// The entire pipeline is finished
    Finished,
    /// Execute the sink operator and then read data from source. Which means that
    /// the regular operators do not have more outputs
    SinkThenSource,
    /// Execute the sink operator and then execute the regular operator. Which means
    /// that the regular operators have more outputs
    SinkThenRegular,
    /// Read from the source operator, which means that the regular operators have
    /// empty output and do not have more outputs
    Source,
}

impl<'a, S: SinkTrait> PipelineExecutor<'a, S> {
    /// Try to create a new [`PipelineExecutor`]
    pub fn try_new(pipeline: &'a Pipeline<S>, client_ctx: &'a ClientContext) -> Result<Self> {
        let source = &pipeline.source;
        let operators = &pipeline.operators;
        let sink = &pipeline.sink;
        let mut intermediate_blocks = Vec::with_capacity(operators.len() + 1);
        let mut local_operator_states = Vec::with_capacity(operators.len());
        intermediate_blocks.push(DataBlock::with_logical_types(
            pipeline.source.op.output_types().to_owned(),
        ));
        pipeline.operators.iter().try_for_each(|operator| {
            intermediate_blocks.push(DataBlock::with_logical_types(
                operator.op.output_types().to_owned(),
            ));
            local_operator_states.push(operator.op.local_operator_state());
            Ok::<_, PipelineExecutorError>(())
        })?;

        let local_source_state = source.op.local_source_state(&*source.global_state);

        let local_sink_state = sink.local_sink_state();

        Ok(Self {
            pipeline,
            local_source_state,
            intermediate_blocks,
            local_operator_states,
            local_sink_state,
            in_process_operators: Vec::new(),
            client_ctx,
        })
    }

    #[inline]
    fn execute_source(&mut self) -> Result<SourceExecStatus> {
        ensure_not_cancelled!(self);

        // SAFETY: intermediate_blocks has at least one block
        let source_output_block =
            unsafe { self.intermediate_blocks.first_mut().unwrap_unchecked() };
        let source = &self.pipeline.source;
        source
            .op
            .read_data(
                source_output_block,
                &*source.global_state,
                &mut *self.local_source_state,
            )
            .context(ExecuteSourceSnafu)
    }

    fn execute_regular_operators(&mut self) -> Result<RegularOperatorsExecutionControl> {
        let operators = &self.pipeline.operators;
        let local_operator_states = &mut self.local_operator_states;
        let blocks = &mut self.intermediate_blocks;
        let current_index = self.in_process_operators.pop().map_or(0, identity);

        // Iterator of the regular operator and its local states
        let iter = unsafe {
            operators
                .get_unchecked(current_index..)
                .iter()
                .zip(local_operator_states.get_unchecked_mut(current_index..))
                .enumerate()
        };

        // Mutable iterator of the blocks
        let mut blocks_iter = unsafe {
            blocks
                .get_unchecked_mut(current_index..)
                .iter_mut()
                .peekable()
        };

        // Execute the regular operator in sequence
        for (i, (operator, local_state)) in iter {
            ensure_not_cancelled!(self);

            let input = unsafe { blocks_iter.next().unwrap_unchecked() };
            let output = unsafe { blocks_iter.peek_mut().unwrap_unchecked() };

            let exec_status = operator
                .op
                .execute(input, output, &*operator.global_state, &mut **local_state)
                .context(ExecuteOperatorSnafu)?;

            match exec_status {
                OperatorExecStatus::Finished => {
                    return Ok(RegularOperatorsExecutionControl::Finished)
                }
                OperatorExecStatus::HaveMoreOutput => {
                    // Push current operator to in process operator
                    self.in_process_operators.push(i + current_index);
                    ensure!(
                        !output.is_empty(),
                        HaveMoreOutputButEmptySnafu {
                            op: operator.op.name()
                        }
                    );
                }
                OperatorExecStatus::NeedMoreInput => {
                    if output.is_empty() {
                        // Output is empty, operator like filter may cause this
                        if self.in_process_operators.is_empty() {
                            // No more in process operators, early return and read
                            // from source
                            return Ok(RegularOperatorsExecutionControl::Source);
                        } else {
                            // Recursive call, execute the in process operators
                            return self.execute_regular_operators();
                        }
                    }
                }
            }
        }

        // Till now, we have executed all of the regular operators.
        // What's more, the output can not be empty 😊
        Ok(if self.in_process_operators.is_empty() {
            RegularOperatorsExecutionControl::SinkThenSource
        } else {
            RegularOperatorsExecutionControl::SinkThenRegular
        })
    }

    fn merge_local_metrics(&self) {
        self.pipeline
            .source
            .op
            .merge_local_source_metrics(&*self.local_source_state);

        self.pipeline
            .operators
            .iter()
            .zip(self.local_operator_states.iter())
            .for_each(|(operator, local_state)| {
                operator.op.merge_local_operator_metrics(&**local_state);
            });
    }
}

impl<'a> PipelineExecutor<'a, Sink> {
    /// Fully execute the pipeline until the source is completely exhausted
    pub fn execute(&mut self) -> Result<()> {
        'outer: loop {
            let exec_status = self.execute_source()?;
            if matches!(exec_status, SourceExecStatus::Finished) {
                break;
            }

            // Inner loop here, because single input may produce multiple output
            'inner: loop {
                // Execute regular
                let control_flow = self.execute_regular_operators()?;
                match control_flow {
                    RegularOperatorsExecutionControl::SinkThenSource => {
                        let exec_status = self.execute_sink()?;
                        if matches!(exec_status, SinkExecStatus::Finished) {
                            break 'outer;
                        }
                        // Need more input from the source
                        break 'inner;
                    }
                    RegularOperatorsExecutionControl::Source => {
                        // Need more input from the source
                        break 'inner;
                    }
                    RegularOperatorsExecutionControl::SinkThenRegular => {
                        let exec_status = self.execute_sink()?;
                        if matches!(exec_status, SinkExecStatus::Finished) {
                            break 'outer;
                        }
                        // Continue the inner loop, we need to execute the in_process_operators
                        // instead of read from source
                    }
                    RegularOperatorsExecutionControl::Finished => {
                        // Break the entire outer loop
                        break 'outer;
                    }
                }
            }
        }

        let sink = &self.pipeline.sink;
        sink.op
            .merge_sink(&*sink.global_state, &mut *self.local_sink_state)
            .context(MergeSinkSnafu)?;

        // Merge the metrics
        self.merge_local_metrics();

        Ok(())
    }

    /// Note that call this function should guarantee the sink input is not empty
    #[inline]
    fn execute_sink(&mut self) -> Result<SinkExecStatus> {
        ensure_not_cancelled!(self);
        let sink_input = unsafe { self.intermediate_blocks.last().unwrap_unchecked() };
        let sink = &self.pipeline.sink;
        sink.op
            .write_data(sink_input, &*sink.global_state, &mut *self.local_sink_state)
            .context(ExecuteSinkSnafu)
    }
}

impl<'a> PipelineExecutor<'a, ()> {
    /// Execute the pipeline once, return the reference to the data block that match
    /// the query result. If `Some` is returned, the implementation should guarantee
    /// the output block is not empty!
    ///
    /// It will return None if the pipeline is finished and caller should not
    /// execute it again
    pub fn execute_once(&mut self) -> Result<Option<&DataBlock>> {
        if self.local_operator_states.is_empty() {
            // The pipeline only contains source, read data from source
            match self.execute_source()? {
                SourceExecStatus::Finished => {
                    self.merge_local_metrics();
                    Ok(None)
                }
                SourceExecStatus::HaveMoreOutput => Ok(self.intermediate_blocks.last()),
            }
        } else {
            if self.in_process_operators.is_empty() {
                // Read data from source
                let exec_status = self.execute_source()?;
                if matches!(exec_status, SourceExecStatus::Finished) {
                    self.merge_local_metrics();
                    return Ok(None);
                }
            }
            // Execute regular operators
            let exec_status = self.execute_regular_operators()?;
            match exec_status {
                RegularOperatorsExecutionControl::Finished => {
                    self.merge_local_metrics();
                    Ok(None)
                }
                RegularOperatorsExecutionControl::SinkThenRegular
                | RegularOperatorsExecutionControl::SinkThenSource => {
                    Ok(self.intermediate_blocks.last())
                }
                RegularOperatorsExecutionControl::Source => self.execute_once(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;
    use std::sync::Arc;

    use super::*;
    use crate::exec::physical_expr::field_ref::FieldRef;
    use crate::exec::physical_operator::numbers::Numbers;
    use crate::exec::physical_operator::projection::Projection;
    use crate::exec::physical_operator::PhysicalOperator;
    use crate::exec::pipeline::Pipelines;
    use data_block::array::{Array, ArrayImpl};
    use data_block::types::LogicalType;

    fn projection(input: Arc<dyn PhysicalOperator>) -> Arc<dyn PhysicalOperator> {
        Arc::new(Projection::new(
            input,
            vec![Arc::new(FieldRef::new(
                0,
                LogicalType::UnsignedBigInt,
                "number".to_string(),
            ))],
        ))
    }

    fn numbers(count: u64) -> Arc<dyn PhysicalOperator> {
        Arc::new(Numbers::new(0, NonZeroU64::new(count).unwrap()))
    }

    #[test]
    fn test_execute_root_pipeline() {
        fn sum_numbers(pipelines: Arc<Pipelines>, client_ctx: &ClientContext) -> u64 {
            let mut pipeline_executor =
                PipelineExecutor::try_new(&pipelines.root_pipelines[0], client_ctx).unwrap();

            let mut sum = 0;
            while let Some(block) = pipeline_executor.execute_once().unwrap() {
                let ArrayImpl::UInt64(array) = block.get_array(0).unwrap() else {
                    panic!("Output array should be `UInt64Array`")
                };

                sum += array.values_iter().sum::<u64>();
            }

            sum
        }

        let count = Numbers::MORSEL_SIZE * 3;
        let root: Arc<dyn PhysicalOperator> = projection(numbers(count));
        let client_ctx = crate::common::client_context::tests::mock_client_context();
        let pipelines = Arc::new(Pipelines::try_new(&root, &client_ctx).unwrap());

        let sum = std::thread::scope(|s| {
            let pipelines_ = Arc::clone(&pipelines);
            let jh_0 = s.spawn(|| sum_numbers(pipelines_, &client_ctx));
            let jh_1 = s.spawn(|| sum_numbers(pipelines, &client_ctx));
            let mut sum = jh_0.join().unwrap();
            sum += jh_1.join().unwrap();
            sum
        });

        assert_eq!(sum, (0..count).sum::<u64>());
    }
}
