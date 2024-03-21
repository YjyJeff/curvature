//! Projection operator

use data_block::block::DataBlock;
use data_block::types::LogicalType;
use snafu::Snafu;
use std::sync::Arc;

use crate::common::client_context::ClientContext;
use crate::exec::physical_expr::utils::compact_display_expressions;
use crate::exec::physical_expr::{ExprError, ExprExecutor, PhysicalExpr};

use super::utils::downcast_mut_local_state;
use super::{
    impl_sink_for_non_sink, impl_source_for_non_source, use_types_for_impl_sink_for_non_sink,
    use_types_for_impl_source_for_non_source, DummyGlobalOperatorState, GlobalOperatorState,
    LocalOperatorState, OperatorError, OperatorExecStatus, OperatorResult, PhysicalOperator,
    StateStringify, Stringify,
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
    /// Input of the projection, it can only have single input!
    input: Vec<Arc<dyn PhysicalOperator>>,
    output_types: Vec<LogicalType>,
    /// Projection expression
    exprs: Vec<Arc<dyn PhysicalExpr>>,
}

impl Projection {
    /// Create a new [`Projection`]
    pub fn new(input: Arc<dyn PhysicalOperator>, exprs: Vec<Arc<dyn PhysicalExpr>>) -> Self {
        Self {
            input: vec![input],
            output_types: exprs
                .iter()
                .map(|expr| expr.output_type().clone())
                .collect(),
            exprs,
        }
    }
}

#[derive(Debug)]
struct ProjectionLocalState(ExprExecutor);

impl StateStringify for ProjectionLocalState {
    fn name(&self) -> &'static str {
        "ProjectionLocalState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl LocalOperatorState for ProjectionLocalState {
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
        Box::new(ProjectionLocalState(ExprExecutor::new(&self.exprs)))
    }

    impl_source_for_non_source!();

    impl_sink_for_non_sink!();
}
