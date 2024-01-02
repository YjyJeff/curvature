//! Aggregate based on hash

use data_block::block::DataBlock;
use data_block::types::LogicalType;
use snafu::{ResultExt, Snafu};

use crate::error::SendableError;
use crate::exec::physical_operator::{
    impl_regular_for_non_regular, use_types_for_impl_regular_for_non_regular, GlobalSinkState,
    GlobalSourceState, LocalSinkState, LocalSourceState, OperatorResult, ParallelismDegree,
    PhysicalOperator, SinkExecStatus, SinkFinalizeStatus, SourceExecStatus, StateStringify,
    Stringify,
};
use std::sync::Arc;
use_types_for_impl_regular_for_non_regular!();

#[allow(missing_docs)]
#[derive(Debug, Snafu)]
pub enum HashAggregateError {}

type Result<T> = std::result::Result<T, HashAggregateError>;

/// Aggregate based on hash
#[derive(Debug)]
pub struct HashAggregate {
    output_types: Vec<LogicalType>,
    children: Vec<Arc<dyn PhysicalOperator>>,
}

impl HashAggregate {
    /// FIXME
    pub fn try_new(input: Arc<dyn PhysicalOperator>) -> Result<Self> {
        Ok(Self {
            output_types: input.output_types().to_owned(),
            children: vec![input],
        })
    }
}

/// Global source state for [`HashAggregate`]
#[derive(Debug)]
pub struct HashAggregateGlobalSourceState;

impl StateStringify for HashAggregateGlobalSourceState {
    fn name(&self) -> &'static str {
        "HashAggregateGlobalSourceState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl GlobalSourceState for HashAggregateGlobalSourceState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Local source state for [`HashAggregate`]
#[derive(Debug)]
pub struct HashAggregateLocalSourceState;

impl StateStringify for HashAggregateLocalSourceState {
    fn name(&self) -> &'static str {
        "HashAggregateLocalSourceState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl LocalSourceState for HashAggregateLocalSourceState {
    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn read_data(&mut self, _output: &mut DataBlock) -> OperatorResult<SourceExecStatus> {
        todo!()
    }
}

/// Global sink state for [`HashAggregate`]
#[derive(Debug)]
pub struct HashAggregateGlobalSinkState;

impl StateStringify for HashAggregateGlobalSinkState {
    fn name(&self) -> &'static str {
        "HashAggregateGlobalSinkState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl GlobalSinkState for HashAggregateGlobalSinkState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Local sink state for [`HashAggregate`]
#[derive(Debug)]
pub struct HashAggregateLocalSinkState;

impl StateStringify for HashAggregateLocalSinkState {
    fn name(&self) -> &'static str {
        "HashAggregateLocalSinkState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl LocalSinkState for HashAggregateLocalSinkState {
    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Stringify for HashAggregate {
    fn name(&self) -> &'static str {
        "HashAggregate"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    /// FIXME: display
    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HashAggregate")
    }
}

impl PhysicalOperator for HashAggregate {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn output_types(&self) -> &[LogicalType] {
        &self.output_types
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    impl_regular_for_non_regular!();

    // Source

    fn is_source(&self) -> bool {
        true
    }

    fn source_parallelism_degree(
        &self,
        _global_state: &dyn GlobalSourceState,
    ) -> OperatorResult<ParallelismDegree> {
        todo!()
    }

    fn read_data(
        &self,
        _output: &mut DataBlock,
        _global_state: &dyn GlobalSourceState,
        _local_state: &mut dyn LocalSourceState,
    ) -> OperatorResult<SourceExecStatus> {
        todo!()
    }

    fn global_source_state(&self) -> OperatorResult<Arc<dyn GlobalSourceState>> {
        Ok(Arc::new(HashAggregateGlobalSourceState))
    }

    fn local_source_state(
        &self,
        _global_state: &dyn GlobalSourceState,
    ) -> OperatorResult<Box<dyn LocalSourceState>> {
        Ok(Box::new(HashAggregateLocalSourceState))
    }

    fn progress(&self, _global_state: &dyn GlobalSourceState) -> OperatorResult<f64> {
        todo!()
    }

    // Sink

    fn is_sink(&self) -> bool {
        true
    }

    fn is_parallel_sink(&self) -> OperatorResult<bool> {
        Ok(true)
    }

    fn write_data(
        &self,
        _input: &DataBlock,
        _global_state: &dyn GlobalSinkState,
        _local_state: &mut dyn LocalSinkState,
    ) -> OperatorResult<SinkExecStatus> {
        todo!()
    }

    fn finish_local_sink(
        &self,
        _global_state: &dyn GlobalSinkState,
        _local_state: &mut dyn LocalSinkState,
    ) -> OperatorResult<()> {
        todo!()
    }

    unsafe fn finalize_sink(
        &self,
        _global_state: &dyn GlobalSinkState,
    ) -> OperatorResult<SinkFinalizeStatus> {
        todo!()
    }

    fn global_sink_state(&self) -> OperatorResult<Arc<dyn GlobalSinkState>> {
        Ok(Arc::new(HashAggregateGlobalSinkState))
    }

    fn local_sink_state(
        &self,
        _global_state: &dyn GlobalSinkState,
    ) -> OperatorResult<Box<dyn LocalSinkState>> {
        Ok(Box::new(HashAggregateLocalSinkState))
    }
}
