//! Projection operator

use data_block::block::DataBlock;
use data_block::types::LogicalType;
use snafu::ResultExt;
use std::sync::Arc;

use crate::exec::{expr_executor::ExprExecutor, physical_expr::PhysicalExpr};

use super::{
    impl_source_sink_for_regular_operator, DummyGlobalState, Error as OperatorError,
    FinishLocalSinkSnafu, GlobalOperatorState, GlobalSinkState, GlobalSinkStateSnafu,
    GlobalSourceState, GlobalSourceStateSnafu, IsParallelSinkSnafu, IsParallelSourceSnafu,
    LocalOperatorState, LocalSinkState, LocalSinkStateSnafu, LocalSourceState,
    LocalSourceStateSnafu, OperatorExecStatus, OperatorResult, PhysicalOperator, ProgressSnafu,
    ReadDataSnafu, SinkExecStatus, SourceExecStatus, Stringify, WriteDataSnafu,
};

#[derive(Debug)]
/// Projection operator, it selects the expression from input
pub struct Projection {
    /// Input of the projection, it can only have single input!
    input: Vec<Arc<dyn PhysicalOperator>>,
    output_types: Vec<LogicalType>,
    /// Projection expression
    exprs: Vec<Arc<dyn PhysicalExpr>>,
}

struct ProjectionLocalState(Vec<ExprExecutor>);

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
        todo!()
    }
}

impl PhysicalOperator for Projection {
    #[inline]
    fn name(&self) -> &'static str {
        "Projection"
    }

    #[inline]
    fn output_types(&self) -> &[LogicalType] {
        &self.output_types
    }

    #[inline]
    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.input
    }

    #[inline]
    fn is_regular_operator(&self) -> bool {
        true
    }

    #[inline]
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

        self.exprs
            .iter()
            .zip(local_state.0.iter_mut())
            .zip(output.mutable_arrays())
            .try_for_each(|((expr, executor), output)| executor.execute(&**expr, input, output))
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

    #[inline]
    fn global_operator_state(&self) -> OperatorResult<Box<dyn GlobalOperatorState + '_>> {
        Ok(Box::new(DummyGlobalState))
    }

    #[inline]
    fn local_operator_state(&self) -> OperatorResult<Box<dyn LocalOperatorState>> {
        Ok(Box::new(ProjectionLocalState(
            self.exprs
                .iter()
                .map(|expr| ExprExecutor::new(&**expr))
                .collect(),
        )))
    }

    impl_source_sink_for_regular_operator!();
}
