//! Projection operator

use data_block::block::DataBlock;
use data_block::types::LogicalType;
use snafu::ResultExt;
use std::sync::Arc;

use super::{
    impl_source_sink_for_regular_operator, DummyGlobalState, FinishLocalSinkSnafu,
    GlobalOperatorState, GlobalSinkState, GlobalSinkStateSnafu, GlobalSourceState,
    GlobalSourceStateSnafu, IsParallelSinkSnafu, IsParallelSourceSnafu, LocalOperatorState,
    LocalSinkState, LocalSinkStateSnafu, LocalSourceState, LocalSourceStateSnafu,
    OperatorExecStatus, OperatorResult, PhysicalOperator, ProgressSnafu, ReadDataSnafu,
    SinkExecStatus, SourceExecStatus, Stringify, WriteDataSnafu,
};

#[derive(Debug)]
/// Projection operator, it selects the expression from input
pub struct Projection {
    /// Input of the projection, it can only have single input!
    input: Vec<Arc<dyn PhysicalOperator>>,
    output_types: Vec<LogicalType>,
    dummy_gloal_state: DummyGlobalState,
}

impl Stringify for Projection {
    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
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
        todo!()
    }

    #[inline]
    fn global_operator_state(&self) -> OperatorResult<&dyn GlobalOperatorState> {
        Ok(&self.dummy_gloal_state)
    }

    #[inline]
    fn local_operator_state(&self) -> OperatorResult<Box<dyn LocalOperatorState>> {
        todo!()
    }

    impl_source_sink_for_regular_operator!();
}
