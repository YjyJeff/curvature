//! Pipeline and its executor

mod builder;
mod executor;

use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::sync::Arc;

use self::builder::PipelineBuilder;
pub use self::builder::PipelineBuilderError;
pub use self::executor::{PipelineExecutor, PipelineExecutorError};
use super::physical_operator::{
    GlobalOperatorState, GlobalSinkState, GlobalSourceState, LocalSinkState, OperatorError,
    PhysicalOperator,
};
use crate::common::client_context::ClientContext;
use crate::common::types::ParallelismDegree;
use crate::private::Sealed;
use crate::tree_node::display::IndentDisplayWrapper;

use snafu::{ResultExt, Snafu};

/// Error for building pipelines
#[derive(Debug, Snafu)]
#[snafu(display("Failed to build pipeline for the physical plan:\n{}", physical_plan))]
pub struct BuildPipelineError {
    source: PipelineBuilderError,
    physical_plan: String,
}

/// Index of the [`Pipeline`] in [`Pipelines`]
pub type PipelineIndex = usize;

/// Pipelines is a DAG that consists of multiple [`Pipeline`]s
#[derive(Debug)]
pub struct Pipelines {
    /// Invariance: the pipeline in index `i` is not depend on `j` if `i < j`.
    pub(super) pipelines: Vec<Pipeline<Sink>>,
    /// Root pipelines(pipeline without sink)
    pub(super) root_pipelines: Vec<Pipeline<()>>,
}

impl Pipelines {
    /// Try to create a [`Pipelines`] with given root operator.
    pub fn try_new(
        root: &Arc<dyn PhysicalOperator>,
        client_ctx: &ClientContext,
    ) -> Result<Self, BuildPipelineError> {
        let pipelines = RefCell::new(Vec::new());
        let mut builder = PipelineBuilder::new(&pipelines, client_ctx);
        builder
            .build_pipelines(root)
            .with_context(|_| BuildPipelineSnafu {
                physical_plan: format!("{}", IndentDisplayWrapper::new(&**root)),
            })?;

        let root_pipelines = builder.finish().with_context(|_| BuildPipelineSnafu {
            physical_plan: format!("{}", IndentDisplayWrapper::new(&**root)),
        })?;

        Ok(Self {
            pipelines: pipelines.into_inner(),
            root_pipelines,
        })
    }
}

/// Trait to constrain the sink in the Pipeline
pub trait SinkTrait: Debug + Sealed {
    /// Local sink state of the sink
    type LocalSinkState: Debug;

    /// Create a local sink state
    fn local_sink_state(&self) -> Self::LocalSinkState;
}

impl Sealed for () {}
impl Sealed for Sink {}

/// Implement SinkTrait for `()`, pipeline contains `()` as sink means the pipeline does not
/// have sink
impl SinkTrait for () {
    type LocalSinkState = ();

    #[inline]
    fn local_sink_state(&self) -> Self::LocalSinkState {}
}

impl SinkTrait for Sink {
    type LocalSinkState = Box<dyn LocalSinkState>;

    #[inline]
    fn local_sink_state(&self) -> Self::LocalSinkState {
        self.op.local_sink_state(&*self.global_state)
    }
}

/// A fragment of the `PhysicalPlan` that can be executed in parallel. It consists of
/// `Source`, `Regular` and `Sink` operators.
///
/// The generic parameter `S` represents the `Sink` operator in the pipeline. There are
/// two kinds of pipelines: with sink and without sink. The pipeline without sink is the
/// root pipeline(pipeline without parent). We use the `()` type to represent the pipeline
/// without sink and `Sink` type to represent the pipeline with sink.
///
/// # Generic instead of Option
///
/// As you can see, we can wrap the `Sink` type with [`Option`] to represent these two
/// kinds of pipelines, why do we use generic here? The advantages of generic:
///
/// - Separate these two pipelines with different types, avoid mixing them in the compiler
/// stage(in static)
///
/// - Different kinds of pipelines have different executors, we can avoid calling the
/// sink methods for the pipeline without sink in the compiler stage(in static)
#[derive(Debug)]
pub struct Pipeline<S> {
    /// Source of the pipeline
    source: Source,
    /// Chain of regular operators
    operators: Vec<Operator>,
    /// Sink of the pipeline.
    sink: S,
    /// Children of this Pipeline. This pipeline can be executed iff all of its children are
    /// finished
    ///
    /// Note that if S is (), the PipelineIndex represents the index of the array that contains
    /// `Pipeline<Sink>`
    pub(super) children: Vec<PipelineIndex>,
}

impl Display for Pipeline<Sink> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Source({}) --> ", self.source.op.name())?;
        self.operators
            .iter()
            .try_for_each(|operator| write!(f, "{} --> ", operator.op.name()))?;
        write!(f, "Sink({})", self.sink.op.name())
    }
}

impl Display for Pipeline<()> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Source({})", self.source.op.name())?;
        let mut iter = self.operators.iter();
        let Some(operator) = iter.next() else {
            return Ok(());
        };
        write!(f, " --> {}", operator.op.name())?;
        iter.try_for_each(|operator| write!(f, " --> {}", operator.op.name()))
    }
}

impl<S> Pipeline<S> {
    /// Compute the parallelism degree of the pipeline
    ///
    /// FIXME: Take care of the intermediate operator and sink operator
    pub(super) fn parallelism_degree(
        &self,
        max_parallelism: ParallelismDegree,
    ) -> ParallelismDegree {
        let source_parallelism = self
            .source
            .op
            .source_parallelism_degree(&*self.source.global_state);

        std::cmp::min(source_parallelism, max_parallelism)
    }
}

impl Pipeline<Sink> {
    /// Finalize the sink, called by the query executor
    pub(super) unsafe fn finalize_sink(&self) -> Result<(), OperatorError> {
        self.sink.op.finalize_sink(&*self.sink.global_state)
    }
}

/// Source operator and its global source state
#[derive(Debug)]
pub(super) struct Source {
    /// Source operator
    op: Arc<dyn PhysicalOperator>,
    /// Global source state
    global_state: Arc<dyn GlobalSourceState>,
}

/// Regular operator and its global operator state
#[derive(Debug)]
struct Operator {
    /// Regular operator
    op: Arc<dyn PhysicalOperator>,
    /// Global operator state
    global_state: Arc<dyn GlobalOperatorState>,
}

impl Clone for Operator {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            op: Arc::clone(&self.op),
            global_state: Arc::clone(&self.global_state),
        }
    }
}

/// Sink operator and its global sink state
#[derive(Debug)]
pub(super) struct Sink {
    /// Sink operator
    op: Arc<dyn PhysicalOperator>,
    /// Global sink state
    global_state: Arc<dyn GlobalSinkState>,
}

impl Clone for Sink {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            op: Arc::clone(&self.op),
            global_state: Arc::clone(&self.global_state),
        }
    }
}
