//! Physical operators
//!
//! Heavily adapted from [`DuckDB`](https://duckdb.org/)

pub mod aggregate;
pub mod empty_table_scan;
pub mod memory_table_scan;
pub mod metric;
pub mod numbers;
pub mod projection;
mod source_ext;
pub mod union;
pub mod utils;

use self::source_ext::SourceOperatorExt;
use self::utils::{
    impl_regular_for_non_regular, impl_sink_for_non_sink, impl_source_for_non_source,
    use_types_for_impl_regular_for_non_regular, use_types_for_impl_sink_for_non_sink,
    use_types_for_impl_source_for_non_source,
};
use crate::common::client_context::ClientContext;
use crate::common::types::ParallelismDegree;
use crate::error::SendableError;
use crate::visit::{Visit, Visitor};
use data_block::block::DataBlock;
use data_block::types::LogicalType;
use snafu::Snafu;
use std::fmt::{Debug, Display};
use std::ops::ControlFlow;
use std::sync::Arc;

/// Maximum parallelism degree
///
/// FIXME: configure it via command/env?
const MAX_PARALLELISM_DEGREE: ParallelismDegree = unsafe { ParallelismDegree::new_unchecked(256) };

#[derive(Debug, Snafu)]
#[allow(missing_docs)]
pub enum OperatorError {
    #[snafu(display("Calling `is_parallel_operator` on a non regular operator: `{}`", op))]
    IsParallelOperator { op: &'static str },
    #[snafu(display("Failed to execute the regular operator: `{}`", op))]
    Execute {
        op: &'static str,
        source: SendableError,
    },
    #[snafu(display("Calling `global_operator_state` on a non regular operator: `{}`", op))]
    GlobalOperatorState { op: &'static str },
    #[snafu(display("Calling `local_operator_state` on a non regular operator: `{}`", op))]
    LocalOperatorState { op: &'static str },
    #[snafu(display("Failed to call `source_parallelism_degree` on the operator: `{}`", op))]
    SourceParallelismDegree {
        op: &'static str,
        source: SendableError,
    },
    #[snafu(display("Failed to read data from the source operator: `{}`", op))]
    ReadData {
        op: &'static str,
        source: SendableError,
    },
    #[snafu(display("Calling `global_source_state` on a non source operator: `{}`", op))]
    GlobalSourceState { op: &'static str },
    #[snafu(display("Failed to call `local_source_state` on the operator: `{}`", op))]
    LocalSourceState {
        op: &'static str,
        source: SendableError,
    },
    #[snafu(display("Failed to call `progress` on the operator: `{}`", op))]
    Progress {
        op: &'static str,
        source: SendableError,
    },
    #[snafu(display("Calling `is_parallel_sink` on a non sink operator: `{}`", op))]
    IsParallelSink { op: &'static str },
    #[snafu(display("Failed to write data to the sink operator: `{}`", op))]
    WriteData {
        op: &'static str,
        source: SendableError,
    },
    #[snafu(display("Failed to call `global_sink_state` on the operator: `{}`", op))]
    GlobalSinkState {
        op: &'static str,
        source: SendableError,
    },
    #[snafu(display("Calling `local_sink_state` on a non sink operator: `{}`", op))]
    LocalSinkState { op: &'static str },
    #[snafu(display(
        "Failed to call the `finish_local_sink` on the sink operator: `{}`",
        op
    ))]
    FinishLocalSink {
        op: &'static str,
        source: SendableError,
    },
    #[snafu(display("Failed to call `finalize_sink` on the sink operator: `{}`", op))]
    FinalizeSink {
        op: &'static str,
        source: SendableError,
    },
    #[snafu(display(
        "Source operator: `{op}` accepts invalid LocalSourceState: `{state}`.
         PipelineExecutor should guarantee it never happens, it has fatal bug 😭
    "
    ))]
    InvalidLocalSourceState {
        op: &'static str,
        state: &'static str,
    },
    #[snafu(display(
        "Source operator: `{op}` accepts invalid GlobalSourceState: `{state}`.
         PipelineExecutor should guarantee it never happens, it has fatal bug 😭
    "
    ))]
    InvalidGlobalSourceState {
        op: &'static str,
        state: &'static str,
    },
    #[snafu(display(
        "Sink operator: `{op}` accepts invalid LocalSinkState: `{state}`.
         PipelineExecutor should guarantee it never happens, it has fatal bug 😭
    "
    ))]
    InvalidLocalSinkState {
        op: &'static str,
        state: &'static str,
    },
    #[snafu(display(
        "Sink operator: `{op}` accepts invalid GlobalSinkState: `{state}`.
         PipelineExecutor should guarantee it never happens, it has fatal bug 😭
    "
    ))]
    InvalidGlobalSinkState {
        op: &'static str,
        state: &'static str,
    },
}

type Result<T> = std::result::Result<T, OperatorError>;
// Used by the children modules
type OperatorResult<T> = Result<T>;

/// Stringify the physical operator
pub trait Stringify {
    /// Get name of the physical operator
    fn name(&self) -> &'static str;

    /// Debug message
    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;

    /// Display the operator without the children info
    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
}

/// Physical operator is the node in the physical plan, it contains the information about
/// how to execute the query.
///
/// Note that we do not provide default implementation. Because we want to avoid the case
/// that compiler compiles but user forget to implement the method that should implement.
/// You can use some macros defined in the [utils] to avoid repeated code
pub trait PhysicalOperator: Send + Sync + Stringify + 'static {
    /// As any for dynamic casting
    fn as_any(&self) -> &dyn std::any::Any;

    /// Get the output logical type of this operator
    fn output_types(&self) -> &[LogicalType];

    /// Get children of this operator.
    ///
    /// If the children is empty, it **must be** TableScan operator, therefore it is
    /// a source. However, if the children is not empty, it **may be** source operator
    fn children(&self) -> &[Arc<dyn PhysicalOperator>];

    // Regular operator(not source and sink) methods

    /// Is the operator a regular operator? It is used to verify the correctness of
    /// the pipeline
    fn is_regular_operator(&self) -> bool;

    /// Can we execute the regular operator in parallel?
    ///
    /// Note that if this method is called on a non regular operator, it should return the
    /// [`OperatorError::IsParallelOperator`]
    fn is_parallel_operator(&self) -> Result<bool>;

    /// Execute the operator with input and write the result to output
    ///
    /// # Notes
    ///
    /// - Operator will try to cast the global/local state to concrete state
    /// according to the concrete type of the operator. Therefore, it is the executor's
    /// responsibility to pass the correct global/local state. The global/local state
    /// should be created with `self.global_operator_state`/`self.local_operator_state`
    /// method
    ///
    /// - All of the regular operator should return `Ok` even if the regular
    /// operator **do not** have regular operator
    fn execute(
        &self,
        input: &DataBlock,
        output: &mut DataBlock,
        global_state: &dyn GlobalOperatorState,
        local_state: &mut dyn LocalOperatorState,
    ) -> Result<OperatorExecStatus>;

    /// Create a global state for the physical operator. Executor calls it and
    /// passes it to the `self.execute` function
    ///
    /// Note that if this method is called on a non regular operator, it should return the
    /// [`OperatorError::GlobalOperatorState`]
    fn global_operator_state(
        &self,
        client_ctx: &ClientContext,
    ) -> Result<Arc<dyn GlobalOperatorState>>;

    /// Create a thread local state for the physical operator. Executor calls it and
    /// passes it to the `self.execute` function
    ///
    /// Note that if this method is called on a non regular operator, it should return the
    /// [`OperatorError::LocalOperatorState`]
    fn local_operator_state(&self) -> Result<Box<dyn LocalOperatorState>>;

    // Source operator methods

    /// Is the operator a source operator? It is used to verify the correctness of
    /// the pipeline
    fn is_source(&self) -> bool;

    /// Parallelism of the source, if the source can not be executed in parallel, it should
    /// return Ok(1)
    ///
    /// Note that if this method is called on a non source operator, it should return the
    /// [`OperatorError::SourceParallelismDegree`]
    fn source_parallelism_degree(
        &self,
        global_state: &dyn GlobalSourceState,
    ) -> Result<ParallelismDegree>;

    /// Read a [`DataBlock`] from the source and write it to output
    ///
    /// # Notes
    ///
    /// - Operator will try to cast the global/local state to concrete state
    /// according to the concrete type of the operator. Therefore, it is the executor's
    /// responsibility to pass the correct global/local state. The global/local state
    /// should be created with `self.global_source_state`/`self.local_source_state`
    /// method
    ///
    ///
    /// FIXME: Async
    fn read_data(
        &self,
        output: &mut DataBlock,
        global_state: &dyn GlobalSourceState,
        local_state: &mut dyn LocalSourceState,
    ) -> Result<SourceExecStatus>;

    /// Create a global source state for the physical operator. Executor calls it and
    /// passes it to the `self.read_data` function
    ///
    /// Note that if this method is called on a non source operator, it should return the
    /// [`OperatorError::GlobalSourceState`]
    fn global_source_state(&self, client_ctx: &ClientContext)
        -> Result<Arc<dyn GlobalSourceState>>;

    /// Create a thread local state, morsel assigned to the thread, for the physical operator.
    /// Executor calls it and passes it to the `self.read_data` function
    ///
    /// Note that if this method is called on a non source operator, it should return the
    /// [`OperatorError::LocalSourceState`]
    fn local_source_state(
        &self,
        global_state: &dyn GlobalSourceState,
    ) -> Result<Box<dyn LocalSourceState>>;

    /// Get progress of the source: [0.0 - 1.0]
    ///
    /// # Notes
    ///
    /// - If this method is called on a non source operator, it should return the
    /// [`OperatorError::Progress`]
    ///
    /// - If the source does not support progress, return negative number
    ///
    /// - The returned value represents the progress the morsel have been assigned
    /// to execute, which is almost equivalent to the progress the source has been
    /// processed because we assume executing the morsel is fast
    fn progress(&self, global_state: &dyn GlobalSourceState) -> Result<f64>;

    // Sink operator methods

    /// Is the operator a sink operator? It is used to verify the correctness of
    /// the pipeline
    fn is_sink(&self) -> bool;

    /// Can we execute the source parallel in parallel?
    ///
    /// Note that if this method is called on a non sink operator, it should return the
    /// [`OperatorError::IsParallelSink`]
    fn is_parallel_sink(&self) -> Result<bool>;

    /// Write the input [`DataBlock`] to the sink
    ///
    /// # Notes
    ///
    /// - Operator will try to cast the global/local state to concrete state
    /// according to the concrete type of the operator. Therefore, it is the executor's
    /// responsibility to pass the correct global/local state. The global/local state
    /// should be created with `self.global_source_state`/`self.local_source_state`
    /// method
    ///
    /// FIXME: Async
    fn write_data(
        &self,
        input: &DataBlock,
        global_state: &dyn GlobalSinkState,
        local_state: &mut dyn LocalSinkState,
    ) -> Result<SinkExecStatus>;

    /// The `finish_local_sink` is called when a single thread has completed execution
    /// of its part of the pipeline, it is the final time that a specific [`LocalSinkState`]
    /// is accessible. This method can be called in parallel while other `write_data` or
    /// `finish_local_sink` calls are active on the same [`GlobalSinkState`].
    ///
    /// The implementation should combine the [`LocalSinkState`] into the [`GlobalState`]
    /// that under the [`GlobalSinkState`]. Such that when the operator is used as the
    /// regular/source operator in the parent pipeline, it can access the data produced
    /// by the sink
    ///
    /// # Notes
    ///
    /// - Operator will try to cast the global/local state to concrete state
    /// according to the concrete type of the operator. Therefore, it is the executor's
    /// responsibility to pass the correct global/local state. The global/local state
    /// should be created with `self.global_source_state`/`self.local_source_state`
    /// method
    ///
    /// FIXME: Async
    fn finish_local_sink(
        &self,
        global_state: &dyn GlobalSinkState,
        local_state: &mut dyn LocalSinkState,
    ) -> Result<()>;

    /// Finalize the Sink state
    ///
    /// # Safety
    ///
    /// The finalize_sink is called when ALL threads are finished execution. It is called
    /// only once per sink, which means that for each sink, only one thread can call it.
    /// In our execution model, the main thread(thread handle the query and schedule tasks)
    /// will call this function
    ///
    /// # Note
    ///
    /// Finalize function can spawn threads and execute the function body in parallel
    unsafe fn finalize_sink(&self, global_state: &dyn GlobalSinkState) -> Result<()>;

    /// Create a global sink state for the physical operator. Executor calls it and
    /// passes it to the `self.write_data` function
    ///
    /// Note that if this method is called on a non sink operator, it should return the
    /// [`OperatorError::GlobalSinkState`]
    fn global_sink_state(&self, client_ctx: &ClientContext) -> Result<Arc<dyn GlobalSinkState>>;

    /// Create a thread local state for the physical operator. Executor calls it and
    /// passes it to the `self.write_data` function
    ///
    /// Note that if this method is called on a non sink operator, it should return the
    /// [`OperatorError::LocalSinkState`]
    fn local_sink_state(
        &self,
        global_state: &dyn GlobalSinkState,
    ) -> Result<Box<dyn LocalSinkState>>;
}

/// Global state associated with the [`PhysicalOperator`]. It will be shared across
/// different threads, therefore implementation should guarantee it can be used in
/// parallel. You need to use lock-free data structures or Mutex(lock) to guarantee
/// it.
///
/// [`GlobalOperatorState`]/[`GlobalSourceState`]/[`GlobalSinkState`] will be built
/// upon it. You can view the [`GlobalState`] as the memory that stores the data
/// used to build different operator state. For example, the `Aggregation` operator
/// stores the hash tables. [`GlobalSinkState`] is writes data to the hash tables and
/// [`GlobalSourceState`] fetches the data from it.
pub trait GlobalState {}

/// Trait for stringify states
pub trait StateStringify {
    /// Get name of the
    fn name(&self) -> &'static str;

    /// Debug message
    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
}

// Regular operator state

/// Global operator state of the [`PhysicalOperator`]. It should be built upon
/// the [`GlobalState`] associated with the the operator.
pub trait GlobalOperatorState: Send + Sync + StateStringify + 'static {
    /// As any such that we can perform dynamic cast
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Thread local state of each physical operator. It is !Send and each thread should
/// create a new one to execute the [`PhysicalOperator`].
///
/// The purpose of the local state is storing some states that are needed to execute
/// the physical operator and reuse it across different [`DataBlock`]s.
pub trait LocalOperatorState: StateStringify + 'static {
    /// As any such that we can perform dynamic cast
    fn as_mut_any(&mut self) -> &mut dyn std::any::Any;
}

#[derive(Debug)]
/// A dummy global operator state, operators do not have global state will use this state as
/// its global state
struct DummyGlobalOperatorState;

impl StateStringify for DummyGlobalOperatorState {
    fn name(&self) -> &'static str {
        "DummyGlobalOperatorState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DummyGlobalOperatorState")
    }
}

impl GlobalOperatorState for DummyGlobalOperatorState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// OperatorExecStatus indicates the status of the  operator for the `execute` call.
/// Executor should check this status and decide how to execute the pipeline
pub enum OperatorExecStatus {
    /// Operator is done with the current input, it produce empty result and
    /// can consume more input if available.
    ///
    /// If there is more input the operator will be called with more input, otherwise
    /// the operator will not be called again. When this status is returned, the output
    /// should be empty
    OutputEmptyAndNeedMoreInput,
    /// Operator is done with the current input, it produce non empty result and
    /// can consume more input if available
    ///
    /// If there is more input the operator will be called with more input, otherwise
    /// the operator will not be called again. When this status is returned, the output
    /// should **not** be empty
    NeedMoreInput,
    /// Operator is not finished yet with the current input. Executor should call the
    /// `execute` method on this executor again with the same input. When this status
    /// is returned, the output should **not** be empty
    ///
    /// Following operators may produce this status:
    /// - Join
    HaveMoreOutput,
    /// Operator has finished the entire pipeline and no more processing is necessary.
    /// The operator will not be called again, and neither will any other operators
    /// in this pipeline.
    ///
    /// Following operators may produce this status:
    /// - StreamingLimit: all of the [`DataBlock`]s have fetched
    Finished,
}

// Source operator state

/// Global source state of the [`PhysicalOperator`]. It should be built upon
/// the [`GlobalState`] associated with the the operator.
pub trait GlobalSourceState: Send + Sync + StateStringify + 'static {
    /// As any such that we can perform dynamic cast
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Thread local state of the source operator. It is !Send and each thread should
/// create a new one to execute the [`PhysicalOperator`].
///
/// The purpose of the local source state is storing some states that are needed to
/// read the data from the source operator and reuse it across different [`DataBlock`]s.
pub trait LocalSourceState: StateStringify + 'static {
    /// As any such that we can perform dynamic cast
    fn as_mut_any(&mut self) -> &mut dyn std::any::Any;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// SourceExecStatus indicates the status of the  operator for the `read_data` call.
/// Executor should check this status and decide how to execute the pipeline
pub enum SourceExecStatus {
    /// Source have more output, executor should pull it again.
    ///
    /// Note that if this status is returned, the output data chunk should **not** be empty
    HaveMoreOutput,
    /// The source is exhausted, no more data will be produced by this source
    ///
    /// Note that if this status is returned, the output data chunk should be empty
    Finished,
}

// Sink operator state

/// Global sink state of the [`PhysicalOperator`]. It should be built upon
/// the [`GlobalState`] associated with the the operator.
pub trait GlobalSinkState: Send + Sync + StateStringify + 'static {
    /// As any such that we can perform dynamic cast
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Thread local state of the sink operator. It is !Send and each thread should
/// create a new one to execute the [`PhysicalOperator`].
///
/// The purpose of the local source state is storing some states that are needed to
/// write the data to the sink operator and reuse it across different [`DataBlock`]s.
pub trait LocalSinkState: StateStringify + 'static {
    /// As any such that we can perform dynamic cast
    fn as_mut_any(&mut self) -> &mut dyn std::any::Any;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// SinkExecStatus indicates the status of the  operator for the `write_data` call.
/// Executor should check this status and decide how to execute the pipeline
pub enum SinkExecStatus {
    /// Sink needs more input, executor can write more data to it
    NeedMoreInput,
    /// The sink finished the executing, do not write data to it anymore
    Finished,
}

// Implement traits for dyn PhysicalOperator

impl Debug for dyn PhysicalOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.debug(f)
    }
}

impl Display for dyn PhysicalOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display(f)
    }
}

impl Visit for dyn PhysicalOperator {
    type Node = dyn PhysicalOperator;

    fn accept<V: Visitor<Self>>(&self, visitor: &mut V) -> ControlFlow<V::Break> {
        visitor.pre_visit(self)?;

        for child in self.children() {
            child.accept(visitor)?;
        }

        visitor.post_visit(self)
    }
}

macro_rules! impl_debug_for_state {
    ($ty:ty) => {
        impl Debug for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.debug(f)
            }
        }
    };
}

impl_debug_for_state!(dyn GlobalOperatorState);
impl_debug_for_state!(dyn LocalOperatorState);
impl_debug_for_state!(dyn GlobalSourceState);
impl_debug_for_state!(dyn LocalSourceState);
impl_debug_for_state!(dyn GlobalSinkState);
impl_debug_for_state!(dyn LocalSinkState);
