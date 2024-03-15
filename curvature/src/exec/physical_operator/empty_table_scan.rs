//! A table scan operator that does not have any data

use std::sync::Arc;

use crate::common::client_context::ClientContext;
use data_block::block::DataBlock;
use data_block::types::LogicalType;

use super::{
    impl_regular_for_non_regular, impl_sink_for_non_sink,
    use_types_for_impl_regular_for_non_regular, use_types_for_impl_sink_for_non_sink,
    GlobalSourceState, LocalSourceState, OperatorResult, ParallelismDegree, PhysicalOperator,
    SourceExecStatus, StateStringify, Stringify,
};

use_types_for_impl_regular_for_non_regular!();
use_types_for_impl_sink_for_non_sink!();

/// A table scan operator that does not have any data
#[derive(Debug)]
pub struct EmptyTableScan {
    output_types: Vec<LogicalType>,
    _children: Vec<Arc<dyn PhysicalOperator>>,
}

impl EmptyTableScan {
    /// Create a new [`EmptyTableScan`]
    pub fn new(output_types: Vec<LogicalType>) -> Self {
        Self {
            output_types,
            _children: Vec::new(),
        }
    }
}

/// Global state for [`EmptyTableScan`]
#[derive(Debug)]
pub struct EmptyTableScanGlobalState;

impl StateStringify for EmptyTableScanGlobalState {
    fn name(&self) -> &'static str {
        "EmptyTableScanGlobalState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl GlobalSourceState for EmptyTableScanGlobalState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Local state for [`EmptyTableScan`]
#[derive(Debug)]
pub struct EmptyTableScanLocalState;

impl StateStringify for EmptyTableScanLocalState {
    fn name(&self) -> &'static str {
        "EmptyTableScanLocalState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl LocalSourceState for EmptyTableScanLocalState {
    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Stringify for EmptyTableScan {
    fn name(&self) -> &'static str {
        "EmptyTableScan"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EmptyTableScan: output_types={:?}", self.output_types)
    }
}

impl PhysicalOperator for EmptyTableScan {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn output_types(&self) -> &[LogicalType] {
        &self.output_types
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self._children
    }

    impl_regular_for_non_regular!();

    fn is_source(&self) -> bool {
        true
    }

    fn source_parallelism_degree(
        &self,
        _global_state: &dyn GlobalSourceState,
    ) -> ParallelismDegree {
        ParallelismDegree::MIN
    }

    fn read_data(
        &self,
        _output: &mut data_block::block::DataBlock,
        _global_state: &dyn GlobalSourceState,
        _local_state: &mut dyn LocalSourceState,
    ) -> OperatorResult<SourceExecStatus> {
        Ok(SourceExecStatus::Finished)
    }

    fn global_source_state(&self, _client_ctx: &ClientContext) -> Arc<dyn GlobalSourceState> {
        Arc::new(EmptyTableScanGlobalState)
    }

    fn local_source_state(
        &self,
        _global_state: &dyn GlobalSourceState,
    ) -> Box<dyn LocalSourceState> {
        Box::new(EmptyTableScanLocalState)
    }

    fn progress(&self, _global_state: &dyn GlobalSourceState) -> f64 {
        1.0
    }

    impl_sink_for_non_sink!();
}
