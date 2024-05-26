//! PhysicalExpression that can be interpreted/executed

pub mod arith;
pub mod conjunction;
pub mod constant;
pub mod executor;
pub mod field_ref;
pub mod function;
pub mod is_null;
pub mod regex_match;
pub mod utils;

use crate::error::SendableError;
use crate::tree_node::{handle_visit_recursion, TreeNode, TreeNodeRecursion, Visitor};
use data_block::array::ArrayImpl;
use data_block::bitmap::Bitmap;
use data_block::block::DataBlock;
use data_block::types::LogicalType;
use snafu::Snafu;
use std::fmt::{Debug, Display};
use std::sync::Arc;

use self::executor::ExprExecCtx;

#[allow(missing_docs)]
#[derive(Debug, Snafu)]
pub enum ExprError {
    #[snafu(display("Failed to execute the `{}` expression", expr))]
    Execute { expr: String, source: SendableError },
}

type Result<T> = std::result::Result<T, ExprError>;
pub(super) type ExprResult<T> = Result<T>;

/// Stringify the [`PhysicalExpr`]
pub trait Stringify {
    /// Name of the expression
    fn name(&self) -> &'static str;

    /// Debug message
    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;

    /// Display the expression **without** children info. We will use this method
    /// to implement `Display` for `dyn PhysicalExpr`. And the `IndentDisplayWrapper`
    /// will use this method internally to display the `dyn PhysicalExpr` in tree
    /// structure
    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;

    /// Display the expression **with** children info in one line.
    ///
    /// It is used in two cases:
    ///
    /// 1. Error message: Reporting which expression is failed
    ///
    /// 2. Display the operator: Displaying expressions the operator contains
    fn compact_display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
}

/// Trait for all of the physical expressions
///
/// TODO: Add selection context
pub trait PhysicalExpr: Stringify + Send + Sync {
    /// `as_any` for downcast
    fn as_any(&self) -> &dyn std::any::Any;

    /// Output type of the expression
    fn output_type(&self) -> &LogicalType;

    /// Get children of this expression
    fn children(&self) -> &[Arc<dyn PhysicalExpr>];

    /// Execute the expression, read from input and write the result to output
    ///
    /// # Arguments
    ///
    /// - `leaf_input`: Input of the [`DataBlock`] for the **leaf node** of the expression.
    ///
    /// - `selection`: Selection bitmap, expr will only perform operation on the selected elements
    ///
    /// - `exec_ctx`: Execution context for **self**. it should be created by [ExprExecCtx::new(self)](ExprExecCtx::new)
    ///
    /// - `output`: The output array that stores the computation result of self. it should
    /// be created by [ArrayImpl::new](ArrayImpl::new) with [self.output_type()](Self::output_type)
    /// as its argument.
    ///
    /// # Notes
    ///
    /// - Implementation should execute the children internally. We design it because
    /// some exprs, like `AndConjunction`, do not need to execute all of its children
    /// before execute themselves
    ///
    /// - If the expression is a boolean expression, it will not write the result to the output.
    /// It should update the `selection` argument instead !!!!!
    ///
    /// - Except the boolean expression, after executing the expressions,
    /// the invariance: `output.len() == leaf_input.len()` should always hold.
    /// If the expression is a boolean expression, the selection should hold the invariance:
    /// `selection.is_empty() || selection.len() == leaf_input.len()`
    fn execute(
        &self,
        leaf_input: &DataBlock,
        selection: &mut Bitmap,
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

impl TreeNode for dyn PhysicalExpr {
    fn visit_children<V, F>(&self, f: &mut F) -> std::result::Result<TreeNodeRecursion, V::Error>
    where
        V: Visitor<Self>,
        F: FnMut(&Self) -> std::result::Result<TreeNodeRecursion, V::Error>,
    {
        for child in self.children() {
            handle_visit_recursion!(f(&**child)?)
        }

        Ok(TreeNodeRecursion::Continue)
    }
}

impl<'a> AsRef<dyn PhysicalExpr + 'a> for dyn PhysicalExpr + 'a {
    fn as_ref(&self) -> &(dyn PhysicalExpr + 'a) {
        self
    }
}
