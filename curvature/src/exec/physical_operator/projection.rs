//! Projection operator

use data_block::block::DataBlock;
use data_block::types::LogicalType;
use snafu::ResultExt;
use std::sync::Arc;

use crate::common::client_context::ClientContext;
use crate::error::SendableError;
use crate::exec::physical_expr::{ExprExecutor, PhysicalExpr};

use super::{
    impl_sink_for_non_sink, impl_source_for_non_source, use_types_for_impl_sink_for_non_sink,
    use_types_for_impl_source_for_non_source, DummyGlobalOperatorState, GlobalOperatorState,
    LocalOperatorState, OperatorError, OperatorExecStatus, OperatorResult, PhysicalOperator,
    StateStringify, Stringify,
};

use_types_for_impl_sink_for_non_sink!();
use_types_for_impl_source_for_non_source!();

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
        write!(f, "Projection: exprs=[")?;
        let mut iter = self.exprs.iter();
        let Some(expr) = iter.next() else {
            return write!(f, "]");
        };
        expr.display(f, true)?;
        iter.try_for_each(|expr| {
            write!(f, ", ")?;
            expr.display(f, true)
        })?;
        write!(f, "]")
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

    fn is_parallel_operator(&self) -> OperatorResult<bool> {
        Ok(true)
    }

    fn execute(
        &self,
        input: &DataBlock,
        output: &mut DataBlock,
        _global_state: &dyn GlobalOperatorState,
        local_state: &mut dyn LocalOperatorState,
    ) -> OperatorResult<OperatorExecStatus> {
        let Some(local_state) = local_state
            .as_mut_any()
            .downcast_mut::<ProjectionLocalState>()
        else {
            return Err(OperatorError::Execute {
                op: self.name(),
                source: format!("Projection operator accepts invalid `local_state`, it should be `ProjectionLocalState`, found `{}`", local_state.name()).into(),
            });
        };

        local_state
            .0
            .execute(&self.exprs, input, output)
            .map_or_else(
                |e| {
                    Err(OperatorError::Execute {
                        op: self.name(),
                        source: Box::new(e),
                    })
                },
                |_| Ok(OperatorExecStatus::NeedMoreInput),
            )
    }

    fn global_operator_state(
        &self,
        _client_ctx: &ClientContext,
    ) -> OperatorResult<Arc<dyn GlobalOperatorState>> {
        Ok(Arc::new(DummyGlobalOperatorState))
    }

    fn local_operator_state(&self) -> OperatorResult<Box<dyn LocalOperatorState>> {
        Ok(Box::new(ProjectionLocalState(ExprExecutor::new(
            &self.exprs,
        ))))
    }

    impl_source_for_non_source!();

    impl_sink_for_non_sink!();
}
