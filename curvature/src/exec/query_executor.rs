//! Executor that execute the query

use data_block::block::DataBlock;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use snafu::{ResultExt, Snafu};

use super::physical_operator::{OperatorError, PhysicalOperator};
use super::pipeline::{
    BuildPipelineError, Pipeline, PipelineExecutor, PipelineExecutorError, PipelineIndex,
    Pipelines, Sink,
};
use crate::common::client_context::ClientContext;
use crate::common::profiler::Instant;
use crate::common::types::ParallelismDegree;
use std::sync::Arc;

#[allow(missing_docs)]
#[derive(Debug, Snafu)]
pub enum QueryExecutorError {
    #[snafu(display("Failed to create a `QueryExecutor`"))]
    Create { source: BuildPipelineError },
    #[snafu(display("Failed to create pipeline executor for pipeline: `{pipeline}`"))]
    CreatePipelineExecutor {
        pipeline: String,
        source: PipelineExecutorError,
    },
    #[snafu(display("Failed to execute the pipeline: `{pipeline}`"))]
    ExecutePipeline {
        pipeline: String,
        source: PipelineExecutorError,
    },
    #[snafu(display("Failed to finalize the pipeline: `{pipeline}`"))]
    FinalizePipeline {
        pipeline: String,
        source: OperatorError,
    },
}

type Result<T> = std::result::Result<T, QueryExecutorError>;

/// Executor that execute the query
#[derive(Debug)]
pub struct QueryExecutor {
    /// Client context of this query
    client_ctx: Arc<ClientContext>,
    /// Pipelines of the query
    pipelines: Pipelines,
}

impl QueryExecutor {
    /// Try to create a new [`QueryExecutor`]
    pub fn try_new(
        root: &Arc<dyn PhysicalOperator>,
        client_ctx: Arc<ClientContext>,
    ) -> Result<Self> {
        let pipelines = Pipelines::try_new(root, &client_ctx).context(CreateSnafu)?;
        Ok(Self {
            client_ctx,
            pipelines,
        })
    }

    /// Execute the query. This function takes a handler function that take the [`DataBlock`]
    /// as input. A typic handler function is the print function that prints the query result
    #[tracing::instrument(skip_all, name = "QueryExecutor::execute")]
    pub fn execute<F>(&self, handler: F) -> Result<()>
    where
        F: Fn(&DataBlock) + Send + Sync,
    {
        // FIXME: Execute the root pipelines in parallel
        for root_pipeline in &self.pipelines.root_pipelines {
            // Execute all of the dependencies of the root pipeline
            root_pipeline
                .children
                .iter()
                .try_for_each(|&child_index| self.execute_pipeline(child_index))?;

            // We can execute the root pipeline now
            let parallelism =
                root_pipeline.parallelism_degree(self.client_ctx.exec_args.parallelism);

            if parallelism > ParallelismDegree::MIN {
                // FIXME: remove rayon, use monoio instead?
                let current_span = tracing::span::Span::current();
                (0..parallelism.get()).into_par_iter().try_for_each(|_| {
                    let span =
                        tracing::info_span!(parent: &current_span, "execute_root_pipeline", root_pipeline = %root_pipeline);
                    let _guard = span.enter();
                    execute_root_pipeline(root_pipeline, &handler, &self.client_ctx)
                })?;
            } else {
                let span =
                    tracing::info_span!("execute_root_pipeline", root_pipeline = %root_pipeline);
                let _guard = span.enter();
                execute_root_pipeline(root_pipeline, &handler, &self.client_ctx)?;
            }
        }

        Ok(())
    }

    /// Execute the pipeline in the given index
    fn execute_pipeline(&self, index: PipelineIndex) -> Result<()> {
        let pipeline = &self.pipelines.pipelines[index];
        pipeline
            .children
            .iter()
            .try_for_each(|&child_index| self.execute_pipeline(child_index))?;

        // We can execute this pipeline now

        let parallelism = pipeline.parallelism_degree(self.client_ctx.exec_args.parallelism);

        if parallelism > ParallelismDegree::MIN {
            // FIXME: remove rayon, use monoio instead?
            let current_span = tracing::span::Span::current();
            (0..parallelism.get()).into_par_iter().try_for_each(|_| {
                let span = tracing::info_span!(parent: &current_span, "execute_pipeline", pipeline = %pipeline);
                let _guard = span.enter();
                execute_pipeline(pipeline, &self.client_ctx)
            })?;
        } else {
            let span = tracing::info_span!("execute_pipeline", pipeline = %pipeline);
            let _guard = span.enter();
            execute_pipeline(pipeline, &self.client_ctx)?;
        }

        // SAFETY: Main thread here, finalize_sink is only called here
        unsafe {
            pipeline
                .finalize_sink()
                .with_context(|_| FinalizePipelineSnafu {
                    pipeline: format!("{}", pipeline),
                })?;
        }

        Ok(())
    }
}

#[inline]
fn execute_pipeline(pipeline: &Pipeline<Sink>, client_ctx: &ClientContext) -> Result<()> {
    let now = Instant::now();
    PipelineExecutor::try_new(pipeline, client_ctx)
        .with_context(|_| CreatePipelineExecutorSnafu {
            pipeline: format!("{}", pipeline),
        })?
        .execute()
        .with_context(|_| ExecutePipelineSnafu {
            pipeline: format!("{}", pipeline),
        })?;

    tracing::debug!(
        "Execute pipeline `{}` elapsed: `{:?}`",
        pipeline,
        now.elapsed()
    );
    Ok(())
}

#[inline]
fn execute_root_pipeline<F>(
    root_pipeline: &Pipeline<()>,
    handler: F,
    client_ctx: &ClientContext,
) -> Result<()>
where
    F: Fn(&DataBlock),
{
    let now = Instant::now();

    let mut pipeline_executor =
        PipelineExecutor::try_new(root_pipeline, client_ctx).with_context(|_| {
            CreatePipelineExecutorSnafu {
                pipeline: format!("{}", root_pipeline),
            }
        })?;

    while let Some(block) =
        pipeline_executor
            .execute_once()
            .with_context(|_| ExecutePipelineSnafu {
                pipeline: format!("{}", root_pipeline),
            })?
    {
        handler(block)
    }

    tracing::debug!(
        "Execute root pipeline `{}` elapsed: `{:?}`",
        root_pipeline,
        now.elapsed()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::common::client_context::ExecArgs;
    use crate::common::uuid::QueryId;
    use crate::exec::physical_expr::field_ref::FieldRef;
    use crate::exec::physical_operator::numbers::Numbers;
    use crate::exec::physical_operator::projection::Projection;
    use data_block::array::ArrayImpl;
    use data_block::types::{Array, LogicalType};
    use snafu::Report;
    use std::num::NonZeroU64;
    use std::sync::atomic::Ordering::Relaxed;

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
    fn test_execute() -> Report<QueryExecutorError> {
        Report::capture(|| {
            let count = Numbers::MORSEL_SIZE * 3;
            let root: Arc<dyn PhysicalOperator> = projection(numbers(count));
            let client_ctx = Arc::new(ClientContext::new(
                QueryId::from_u128(0),
                ExecArgs {
                    parallelism: ParallelismDegree::new(2).unwrap(),
                },
            ));

            let query_executor = QueryExecutor::try_new(&root, client_ctx)?;
            let sum = std::sync::atomic::AtomicU64::new(0);
            query_executor.execute(|block: &DataBlock| {
                let ArrayImpl::UInt64(array) = block.get_array(0).unwrap() else {
                    panic!("Output array should be `UInt64Array`")
                };

                sum.fetch_add(array.values_iter().sum::<u64>(), Relaxed);
            })?;

            assert_eq!(sum.load(Relaxed), (0..count).sum::<u64>());

            Ok(())
        })
    }
}
