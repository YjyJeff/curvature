//! Physical operators
//!
//! Heavily adapted from [`DuckDB`](https://duckdb.org/)

pub mod aggregate;
pub mod empty_table_scan;
mod ext_traits;
pub mod memory_table_scan;
pub mod metric;
pub mod numbers;
pub mod projection;
pub mod union;
pub mod utils;

use self::ext_traits::SourceOperatorExt;
use self::metric::MetricsSet;
use self::utils::{
    impl_regular_for_non_regular, impl_sink_for_non_sink, impl_source_for_non_source,
    use_types_for_impl_regular_for_non_regular, use_types_for_impl_sink_for_non_sink,
    use_types_for_impl_source_for_non_source,
};
use crate::common::client_context::ClientContext;
use crate::common::types::ParallelismDegree;
use crate::error::SendableError;
use crate::tree_node::{handle_visit_recursion, TreeNode, TreeNodeRecursion, Visitor};
use data_block::block::DataBlock;
use data_block::types::LogicalType;
use snafu::Snafu;
use std::fmt::{Debug, Display};
use std::sync::Arc;

/// Maximum parallelism degree
///
/// FIXME: configure it via command/env?
const MAX_PARALLELISM_DEGREE: ParallelismDegree = unsafe { ParallelismDegree::new_unchecked(256) };

#[derive(Debug, Snafu)]
#[allow(missing_docs)]
pub enum OperatorError {
    #[snafu(display("Failed to execute the regular operator"))]
    Execute { source: SendableError },
    #[snafu(display("Failed to read data from the source operator"))]
    ReadData { source: SendableError },
    #[snafu(display("Failed to write data to the sink operator"))]
    WriteData { source: SendableError },
    #[snafu(display("Failed to call the `merge_sink` on the sink operator"))]
    MergeSink { source: SendableError },
    #[snafu(display("Failed to call `finalize_sink` on the sink operator"))]
    FinalizeSink { source: SendableError },
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

    /// Display the operator without the children info. We will use this function
    /// to implement the `Display` trait for the `dyn PhysicalOperator`
    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
}

/// Physical operator is the node in the physical plan, it contains the information about
/// how to execute the query.
///
/// Note that we do not provide default implementation. Because we want to avoid the case
/// that compiler compiles but user forget to implement the method that should implement.
/// You can use some macros defined in the [utils] to avoid repeated code
///
/// # Hacks
///
/// All of the local states must bounded by the `'static` lifetime, because the
/// [`std::any::Any`] trait requires it and the reason can be found [here]. However,
/// in the execution model, the [`PhysicalOperator`]s will always out-live its local
/// states, see [`PipelineExecutor`] for details. So when the local states need some
/// fields in the [`PhysicalOperator`] to make it self-contained, a simple solution
/// would be wrapping them into [`Arc`]. Pretty annoying 😂... Instead of using [`Arc`],
/// I would recommend to use `*const T`. The consequence is that we may run into the
/// undefined behavior if we use it incorrectly 🙃. Anyway, I know what I am doing now
/// and I will take responsibility for this hack!
///
/// [here]: https://internals.rust-lang.org/t/would-non-static-typeid-be-at-all-possible/14258
/// [`PipelineExecutor`]: crate::exec::pipeline::PipelineExecutor
pub trait PhysicalOperator: Send + Sync + Stringify + 'static {
    /// As any for dynamic casting
    fn as_any(&self) -> &dyn std::any::Any;

    /// Get the output logical type of this operator
    fn output_types(&self) -> &[LogicalType];

    /// Get children of this operator.
    fn children(&self) -> &[Arc<dyn PhysicalOperator>];

    /// Get the metrics of this operator. Caller should call it after the operator
    /// has finished the execution. Otherwise, all of the metrics are zeros
    fn metrics(&self) -> MetricsSet;

    // Regular operator(not source and sink) methods

    /// Is the operator a regular operator? It is used to verify the correctness of
    /// the pipeline
    fn is_regular_operator(&self) -> bool;

    /// Execute the operator with input and write the result to output
    ///
    /// # Notes
    ///
    /// All of the regular operator should return `Ok` even if the regular
    /// operator **do not** have regular operator
    ///
    /// # Panics
    ///
    /// [PipelineExecutor] should guarantee following invariants, otherwise implementation will panic:
    ///
    /// - output should match the operator's [`Self::output_types()`]
    ///
    /// - The global and local state should be created with [`Self::global_operator_state()`]/
    /// [`Self::local_operator_state()`] methods.
    ///
    /// [PipelineExecutor]: crate::exec::pipeline::PipelineExecutor
    fn execute(
        &self,
        input: &DataBlock,
        output: &mut DataBlock,
        global_state: &dyn GlobalOperatorState,
        local_state: &mut dyn LocalOperatorState,
    ) -> Result<OperatorExecStatus>;

    /// Create a global state for the physical operator. [`PipelineExecutor`] calls it and
    /// passes it to the [`Self::execute()`] function
    ///
    /// [`PipelineExecutor`]: crate::exec::pipeline::PipelineExecutor
    fn global_operator_state(&self, client_ctx: &ClientContext) -> Arc<dyn GlobalOperatorState>;

    /// Create a thread local state for the physical operator. [PipelineExecutor] calls it and
    /// passes it to the [`Self::execute()`] function
    ///
    /// [PipelineExecutor]: crate::exec::pipeline::PipelineExecutor
    fn local_operator_state(&self) -> Box<dyn LocalOperatorState>;

    /// Merge the local operator metrics into the global metrics in self. This function
    /// should be called when the pipeline executor has finished the execution
    ///
    /// # Panics
    ///
    /// The local state should be created with [`Self::local_operator_state()`] method,
    /// otherwise panic
    fn merge_local_operator_metrics(&self, local_state: &dyn LocalOperatorState);

    // Source operator methods

    /// Is the operator a source operator? It is used to verify the correctness of
    /// the pipeline
    fn is_source(&self) -> bool;

    /// Parallelism of the source, if the source can not be executed in parallel, it should
    /// return Ok(1)
    fn source_parallelism_degree(&self, global_state: &dyn GlobalSourceState) -> ParallelismDegree;

    /// Read a [`DataBlock`] from the source and write it to output
    ///
    /// # Panics
    ///
    /// [PipelineExecutor] should guarantee following invariants, otherwise implementation will panic:
    ///
    /// - output should match the operator's [`Self::output_types()`]
    ///
    /// - The global and local state should be created with [`Self::global_source_state()`]/
    /// [`Self::local_source_state()`] methods.
    ///
    /// [PipelineExecutor]: crate::exec::pipeline::PipelineExecutor
    ///
    /// FIXME: Async
    fn read_data(
        &self,
        output: &mut DataBlock,
        global_state: &dyn GlobalSourceState,
        local_state: &mut dyn LocalSourceState,
    ) -> Result<SourceExecStatus>;

    /// Create a global source state for the physical operator. [`PipelineExecutor`] calls it and
    /// passes it to the [`Self::read_data`] function
    ///
    /// [`PipelineExecutor`]: crate::exec::pipeline::PipelineExecutor
    fn global_source_state(&self, client_ctx: &ClientContext) -> Arc<dyn GlobalSourceState>;

    /// Create a thread local state, morsel assigned to the thread, for the physical operator.
    /// [`PipelineExecutor`] calls it and passes it to the [`Self::read_data()`] function
    ///
    /// [`PipelineExecutor`]: crate::exec::pipeline::PipelineExecutor
    fn local_source_state(&self, global_state: &dyn GlobalSourceState)
        -> Box<dyn LocalSourceState>;

    /// Merge the local source metrics into the global metrics in self. This function
    /// should be called when the pipeline executor has finished the execution
    ///
    /// # Panics
    ///
    /// The local state should be created with [`Self::local_source_state()`] method,
    /// otherwise panic
    fn merge_local_source_metrics(&self, local_state: &dyn LocalSourceState);

    /// Get progress of the source: [0.0 - 1.0]
    ///
    /// # Notes
    ///
    /// - If the source does not support progress, return negative number
    ///
    /// - The returned value represents the progress the morsel have been assigned
    /// to execute, which is almost equivalent to the progress the source has been
    /// processed because we assume executing the morsel is fast
    fn progress(&self, global_state: &dyn GlobalSourceState) -> f64;

    // Sink operator methods

    /// Is the operator a sink operator? It is used to verify the correctness of
    /// the pipeline
    fn is_sink(&self) -> bool;

    /// Write the input [`DataBlock`] to the sink
    ///
    /// # Panics
    ///
    /// [PipelineExecutor] should guarantee following invariants, otherwise implementation will panic:
    ///
    /// - The global and local state should be created with [`Self::global_sink_state()`]/
    /// [`Self::local_sink_state()`] methods.
    ///
    /// - The input should not be empty
    ///
    /// [PipelineExecutor]: crate::exec::pipeline::PipelineExecutor
    ///
    /// FIXME: Async
    fn write_data(
        &self,
        input: &DataBlock,
        global_state: &dyn GlobalSinkState,
        local_state: &mut dyn LocalSinkState,
    ) -> Result<SinkExecStatus>;

    /// The `merge_sink` is called when a single thread has completed the execution
    /// of its part for the pipeline, it is the final time that a specific [`LocalSinkState`]
    /// is accessible. This method can be called in parallel while other `write_data` or
    /// `merge_sink` calls are active on the same [`GlobalSinkState`].
    ///
    /// The implementation should combine the [`LocalSinkState`] into the [`GlobalState`]
    /// that under the [`GlobalSinkState`]. Such that when the operator is used as the
    /// regular/source operator in the parent pipeline, it can access the data produced
    /// by the sink. What's more, the implementation should also combine the local
    /// metric into the global metric
    ///
    /// # Panics
    ///
    /// The global and local state should be created with [`Self::global_sink_state()`]/
    /// [`Self::local_sink_state()`] methods.
    ///
    /// FIXME: Async
    fn merge_sink(
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
    ///
    /// # Panics
    ///
    /// The global state should be created with [`Self::global_sink_state()`]
    unsafe fn finalize_sink(&self, global_state: &dyn GlobalSinkState) -> Result<()>;

    /// Create a global sink state for the physical operator. [`PipelineExecutor`] calls it and
    /// passes it to the [`Self::write_data()`]/[`Self::merge_sink()`]/[`Self::finalize_sink()`]
    /// functions
    ///
    /// [`PipelineExecutor`]: crate::exec::pipeline::PipelineExecutor
    fn global_sink_state(&self, client_ctx: &ClientContext) -> Arc<dyn GlobalSinkState>;

    /// Create a local sink state for the physical operator. [`PipelineExecutor`] calls it and
    /// passes it to the [`Self::write_data()`]/[`Self::merge_sink()`] functions
    ///
    /// [`PipelineExecutor`]: crate::exec::pipeline::PipelineExecutor
    fn local_sink_state(&self, global_state: &dyn GlobalSinkState) -> Box<dyn LocalSinkState>;

    // Metrics
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
    fn as_any(&self) -> &dyn std::any::Any;

    /// As any such that we can perform dynamic cast
    fn as_mut_any(&mut self) -> &mut dyn std::any::Any;
}

#[derive(Debug)]
/// A dummy global operator state, operators do not have global state will use this state as
/// its global state
pub struct DummyGlobalOperatorState;

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
    /// Operator is done with the current input, it produce non empty result and
    /// can consume more input if available
    ///
    /// If there is more input the operator will be called with more input, otherwise
    /// the operator will not be called again
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
    fn as_any(&self) -> &dyn std::any::Any;

    /// As any such that we can perform dynamic cast
    fn as_mut_any(&mut self) -> &mut dyn std::any::Any;
}

#[derive(Debug)]
/// A dummy global sink state, operators do not have global sink state will use this state as
/// its global sink state
pub struct DummyGlobalSourceState;

impl StateStringify for DummyGlobalSourceState {
    fn name(&self) -> &'static str {
        "DummyGlobalSourceState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DummyGlobalSourceState")
    }
}

impl GlobalSourceState for DummyGlobalSourceState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
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

#[derive(Debug)]
/// A dummy global sink state, operators do not have global sink state will use this state as
/// its global sink state
pub struct DummyGlobalSinkState;

impl StateStringify for DummyGlobalSinkState {
    fn name(&self) -> &'static str {
        "DummyGlobalSinkState"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DummyGlobalSinkState")
    }
}

impl GlobalSinkState for DummyGlobalSinkState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
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

impl TreeNode for dyn PhysicalOperator {
    fn visit_children<V, F>(&self, f: &mut F) -> std::result::Result<TreeNodeRecursion, V::Error>
    where
        V: Visitor<Self>,
        F: FnMut(&Self) -> std::result::Result<TreeNodeRecursion, V::Error>,
    {
        for child in self.children() {
            handle_visit_recursion!(f(&**child)?);
        }

        Ok(TreeNodeRecursion::Continue)
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
