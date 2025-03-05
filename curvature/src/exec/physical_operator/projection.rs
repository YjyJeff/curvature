//! Projection operator

use data_block::block::DataBlock;
use data_block::types::LogicalType;
use snafu::Snafu;
use std::collections::HashMap;
use std::fmt::Write;
use std::sync::Arc;
use std::time::Duration;

use crate::common::client_context::ClientContext;
use crate::exec::physical_expr::executor::ExprsExecutor;
use crate::exec::physical_expr::utils::{CompactExprDisplayWrapper, compact_display_expressions};
use crate::exec::physical_expr::{ExprError, PhysicalExpr};
use crate::exec::physical_operator::metric::MetricsSet;

use super::ext_traits::RegularOperatorExt;
use super::metric::{MetricValue, Time};
use super::utils::downcast_mut_local_state;
use super::{
    DummyGlobalOperatorState, GlobalOperatorState, LocalOperatorState, OperatorError,
    OperatorExecStatus, OperatorResult, PhysicalOperator, StateStringify, Stringify,
    impl_sink_for_non_sink, impl_source_for_non_source, use_types_for_impl_sink_for_non_sink,
    use_types_for_impl_source_for_non_source,
};

use_types_for_impl_sink_for_non_sink!();
use_types_for_impl_source_for_non_source!();

#[derive(Debug, Snafu)]
#[snafu(display("Failed to execute the `{}` operator", projection))]
struct ProjectionError {
    projection: String,
    source: ExprError,
}

#[derive(Debug)]
/// Projection operator, it selects the expression from input
pub struct Projection {
    /// Input of the projection, it can only have single input
    input: [Arc<dyn PhysicalOperator>; 1],
    /// Projection expression
    exprs: Vec<Arc<dyn PhysicalExpr>>,
    output_types: Vec<LogicalType>,
    /// Metrics
    metrics: ProjectionMetrics,
}

#[derive(Debug)]
struct ProjectionMetrics(Vec<Time>);

impl Projection {
    /// Create a new [`Projection`]
    pub fn new(input: Arc<dyn PhysicalOperator>, exprs: Vec<Arc<dyn PhysicalExpr>>) -> Self {
        Self {
            input: [input],
            output_types: exprs
                .iter()
                .map(|expr| expr.output_type().clone())
                .collect(),
            metrics: ProjectionMetrics((0..exprs.len()).map(|_| Time::default()).collect()),
            exprs,
        }
    }
}

/// Local state of the projection
#[derive(Debug)]
pub struct ProjectionLocalState(ExprsExecutor);

impl StateStringify for ProjectionLocalState {
    fn name(&self) -> &'static str {
        "ProjectionLocalState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl LocalOperatorState for ProjectionLocalState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Stringify for Projection {
    fn name(&self) -> &'static str {
        "Projection"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Projection: exprs=")?;
        compact_display_expressions(f, &self.exprs)
    }
}

impl PhysicalOperator for Projection {
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
        let mut metrics_set = HashMap::new();
        self.exprs
            .iter()
            .zip(&self.metrics.0)
            .for_each(|(expr, time)| {
                metrics_set.insert(
                    expr.to_string().into(),
                    MetricValue::Time(time.nanoseconds()),
                );
            });

        MetricsSet {
            name: "Projection",
            metrics: metrics_set,
        }
    }

    // Regular Operator

    fn is_regular_operator(&self) -> bool {
        true
    }

    fn execute(
        &self,
        input: &DataBlock,
        output: &mut DataBlock,
        _global_state: &dyn GlobalOperatorState,
        local_state: &mut dyn LocalOperatorState,
    ) -> OperatorResult<OperatorExecStatus> {
        let local_state =
            downcast_mut_local_state!(self, local_state, ProjectionLocalState, OPERATOR);

        local_state
            .0
            .execute(&self.exprs, input, output)
            .map_or_else(
                |e| {
                    Err(OperatorError::Execute {
                        source: Box::new(ProjectionError {
                            projection: (self as &dyn PhysicalOperator).to_string(),
                            source: e,
                        }),
                    })
                },
                |_| Ok(OperatorExecStatus::NeedMoreInput),
            )
    }

    fn global_operator_state(&self, _client_ctx: &ClientContext) -> Arc<dyn GlobalOperatorState> {
        Arc::new(DummyGlobalOperatorState)
    }

    fn local_operator_state(&self) -> Box<dyn LocalOperatorState> {
        Box::new(ProjectionLocalState(ExprsExecutor::new(&self.exprs)))
    }

    fn merge_local_operator_metrics(&self, local_state: &dyn LocalOperatorState) {
        let local_state = self.downcast_ref_local_operator_state(local_state);
        let mut local_exec_time = Duration::default();
        let metrics_string = self
            .exprs
            .iter()
            .zip(&self.metrics.0)
            .zip(local_state.0.execution_times())
            .fold(
                String::new(),
                |mut output, ((expr, global_exec_time), exec_time)| {
                    local_exec_time += exec_time;
                    global_exec_time.add_duration(exec_time);
                    let _ = write!(
                        output,
                        "expr `{}` takes `{:?}`. ",
                        CompactExprDisplayWrapper::new(&**expr),
                        exec_time
                    );
                    output
                },
            );

        tracing::debug!(
            "Total projection time `{:?}`. {}",
            local_exec_time,
            metrics_string
        )
    }

    impl_source_for_non_source!();

    impl_sink_for_non_sink!();
}

impl RegularOperatorExt for Projection {
    type GlobalOperatorState = DummyGlobalOperatorState;
    type LocalOperatorState = ProjectionLocalState;
}
