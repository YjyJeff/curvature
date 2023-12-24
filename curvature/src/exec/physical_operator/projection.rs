//! Projection operator

use data_block::block::DataBlock;
use data_block::types::LogicalType;
use snafu::ResultExt;
use std::sync::Arc;

use crate::error::SendableError;
use crate::exec::expr_executor::ExprExecutor;
use crate::exec::physical_expr::PhysicalExpr;

use super::{
    impl_sink_for_non_sink, impl_source_for_non_source, use_types_for_impl_sink_for_non_sink,
    use_types_for_impl_source_for_non_source, DummyGlobalState, GlobalOperatorState,
    LocalOperatorState, OperatorError, OperatorExecStatus, OperatorResult, PhysicalOperator,
    Stringify,
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

struct ProjectionLocalState(ExprExecutor);

impl LocalOperatorState for ProjectionLocalState {
    #[inline]
    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Stringify for Projection {
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
    fn name(&self) -> &'static str {
        "Projection"
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
                source: "Projection operator accepts invalid `local_state`, it should be `ProjectionLocalState`".into(),
            });
        };

        local_state
            .0
            .execute(self.exprs.iter(), input, output)
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

    fn global_operator_state(&self) -> OperatorResult<Box<dyn GlobalOperatorState>> {
        Ok(Box::new(DummyGlobalState))
    }

    fn local_operator_state(&self) -> OperatorResult<Box<dyn LocalOperatorState>> {
        Ok(Box::new(ProjectionLocalState(ExprExecutor::new(
            self.exprs.iter(),
        ))))
    }

    impl_source_for_non_source!();

    impl_sink_for_non_sink!();
}
