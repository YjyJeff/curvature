//! PhysicalExpression that can be interpreted/executed

mod executor;
pub mod field_ref;
pub mod function;

use crate::error::SendableError;
use data_block::array::ArrayImpl;
use data_block::block::DataBlock;
use data_block::types::LogicalType;
pub use executor::ExprExecutor;
use snafu::Snafu;
use std::fmt::{Debug, Display};
use std::sync::Arc;

#[allow(missing_docs)]
#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Failed to execute the expression: `{}`", expr))]
    Execute {
        expr: &'static str,
        source: SendableError,
    },
}

type Result<T> = std::result::Result<T, Error>;
pub(super) type ExprResult<T> = Result<T>;

/// Stringify the [`PhysicalExpr`]
pub trait Stringify {
    /// Name of the expression
    fn name(&self) -> &'static str;

    /// Debug message
    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;

    /// Display message
    ///
    /// If `compact` is true, use one line representation for each expression.
    /// Otherwise, prints a tree of expressions one node per line
    fn display(&self, f: &mut std::fmt::Formatter<'_>, compact: bool) -> std::fmt::Result;
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
    /// Note that `ExprExecutor` should guarantee the output has the same type
    /// with `self.output_type`
    fn execute(&self, input: &DataBlock, output: &mut ArrayImpl) -> Result<()>;
}

impl Debug for dyn PhysicalExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.debug(f)
    }
}

impl Display for dyn PhysicalExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display(f, true)
    }
}
