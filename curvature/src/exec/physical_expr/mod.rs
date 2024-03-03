//! PhysicalExpression that can be interpreted/executed

pub mod arith;
mod executor;
pub mod field_ref;
pub mod function;
pub mod utils;

use crate::error::SendableError;
use crate::visit::{Visit, Visitor};
use data_block::array::ArrayImpl;
use data_block::block::DataBlock;
use data_block::types::LogicalType;
pub use executor::ExprExecutor;
use snafu::Snafu;
use std::fmt::{Debug, Display};
use std::ops::ControlFlow;
use std::sync::Arc;

use self::executor::ExprExecCtx;

#[allow(missing_docs)]
#[derive(Debug, Snafu)]
pub enum ExprError {
    #[snafu(display("Failed to execute the expression"))]
    Execute { source: SendableError },
}

type Result<T> = std::result::Result<T, ExprError>;
pub(super) type ExprResult<T> = Result<T>;

/// Stringify the [`PhysicalExpr`]
pub trait Stringify {
    /// Name of the expression
    fn name(&self) -> &'static str;

    /// Debug message
    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;

    /// Display the expression without children info
    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;

    /// Display the expression with children info in one line.
    fn compact_display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
}

/// Trait for all of the physical expressions
pub trait PhysicalExpr: Stringify + Send + Sync {
    /// as_any for down cast
    fn as_any(&self) -> &dyn std::any::Any;

    /// Output type of the expression
    fn output_type(&self) -> &LogicalType;

    /// Get children of this expression
    fn children(&self) -> &[Arc<dyn PhysicalExpr>];

    /// Execute the expression, read from input and write the result to output
    ///
    /// # Note
    ///
    /// Implementation should execute the children internally. We design it because
    /// some exprs, like `AndConjunction`, do not need to execute all of its children
    /// before execute themselves
    ///
    /// # Arguments
    ///
    /// - `leaf_input`: Input of the [`DataBlock`] for the **leaf node** of the expression.
    ///
    /// - `exec_ctx`: Execution context for **self**. it should be created by `ExprExecCtx::new(self)`
    ///
    /// - `output`: The output array that stores the computation result of self. it should
    /// match `self.output_type`
    fn execute(
        &self,
        leaf_input: &DataBlock,
        exec_ctx: &mut ExprExecCtx,
        output: &mut ArrayImpl,
    ) -> Result<()>;
}

impl Debug for dyn PhysicalExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.debug(f)
    }
}

impl Display for dyn PhysicalExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display(f)
    }
}

impl Visit for dyn PhysicalExpr {
    type Node = dyn PhysicalExpr;

    fn accept<V: Visitor<Self>>(&self, visitor: &mut V) -> ControlFlow<V::Break> {
        visitor.pre_visit(self)?;

        for child in self.children() {
            child.accept(visitor)?;
        }

        visitor.post_visit(self)
    }
}
