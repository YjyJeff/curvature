//! Comparison between two expressions

use std::sync::Arc;

use data_block::array::ArrayImpl;
use data_block::bitmap::Bitmap;
use data_block::block::DataBlock;
use data_block::types::LogicalType;
use snafu::{ensure, Snafu};

use super::executor::ExprExecCtx;
use super::utils::CompactExprDisplayWrapper;
use super::{execute_binary_children, ExprResult, PhysicalExpr, Stringify};
use crate::common::expr_operator::comparison::{
    infer_comparison_func_set, CmpOperator, ComparisonFunctionSet,
};
use crate::exec::physical_expr::constant::Constant;

#[allow(missing_docs)]
/// Error returned by comparison expression
#[derive(Debug, Snafu)]
pub enum ComparisonError {
    #[snafu(display("Perform `{}` comparison between `{:?}` and `{:?}` is not supported", op.symbol_ident(), left,right))]
    UnsupportedInputs {
        left: LogicalType,
        right: LogicalType,
        op: CmpOperator,
    },
    #[snafu(display(
        "Perform `{}` between two constants(left: `{}`, right: `{}`) makes no sense. Planner/Optimizer should handle it in advance", 
        op,
        left,
        right
    ))]
    TwoConstants {
        left: String,
        right: String,
        op: CmpOperator,
    },
}

/// Comparison between two expressions
#[derive(Debug)]
pub struct Comparison {
    output_type: LogicalType,
    children: [Arc<dyn PhysicalExpr>; 2],
    op: CmpOperator,
    /// Function set used to perform comparison. It is determined by the children and op in
    /// the constructor phase
    ///
    /// Note that we introduce an extra indirection here by function pointer. Writing code
    /// this way is elegant and benchmark show that the impact of this extra indirection
    /// if negligible 😊
    function_set: ComparisonFunctionSet,
    alias: String,
}

type Result<T> = std::result::Result<T, ComparisonError>;

impl Comparison {
    /// Create a new [`Comparison`] expression
    pub fn try_new(
        left: Arc<dyn PhysicalExpr>,
        op: CmpOperator,
        right: Arc<dyn PhysicalExpr>,
    ) -> Result<Self> {
        Self::try_new_with_alias(left, op, right, String::new())
    }
    /// Create a new [`Comparison`] expression
    pub fn try_new_with_alias(
        left: Arc<dyn PhysicalExpr>,
        op: CmpOperator,
        right: Arc<dyn PhysicalExpr>,
        alias: String,
    ) -> Result<Self> {
        let function_set = if let Some(function_set) =
            infer_comparison_func_set(left.output_type(), right.output_type(), op)
        {
            function_set
        } else {
            return UnsupportedInputsSnafu {
                left: left.output_type().to_owned(),
                right: right.output_type().to_owned(),
                op,
            }
            .fail();
        };
        ensure!(
            left.as_any().downcast_ref::<Constant>().is_none()
                || right.as_any().downcast_ref::<Constant>().is_none(),
            TwoConstantsSnafu {
                left: CompactExprDisplayWrapper::new(&*left).to_string(),
                right: CompactExprDisplayWrapper::new(&*right).to_string(),
                op
            }
        );
        Ok(Self {
            output_type: LogicalType::Boolean,
            children: [left, right],
            op,
            function_set,
            alias,
        })
    }

    /// Get left child
    pub fn left_child(&self) -> &Arc<dyn PhysicalExpr> {
        &self.children[0]
    }

    /// Get right child
    pub fn right_child(&self) -> &Arc<dyn PhysicalExpr> {
        &self.children[1]
    }
}

impl Stringify for Comparison {
    fn name(&self) -> &'static str {
        self.op.ident()
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.op.ident())
    }

    fn compact_display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.left_child().compact_display(f)?;
        write!(f, " {} ", self.op.symbol_ident())?;
        self.right_child().compact_display(f)
    }

    fn alias(&self) -> &str {
        &self.alias
    }
}

impl PhysicalExpr for Comparison {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn output_type(&self) -> &LogicalType {
        &self.output_type
    }

    fn children(&self) -> &[Arc<dyn PhysicalExpr>] {
        &self.children
    }

    fn execute(
        &self,
        leaf_input: &DataBlock,
        selection: &mut Bitmap,
        exec_ctx: &mut ExprExecCtx,
        output: &mut ArrayImpl,
    ) -> ExprResult<()> {
        // Execute children
        execute_binary_children(self, leaf_input, selection, exec_ctx)?;

        // Execute comparison
        let left_array = unsafe { exec_ctx.intermediate_block.get_array_unchecked(0) };
        let right_array = unsafe { exec_ctx.intermediate_block.get_array_unchecked(1) };

        match (left_array.len(), right_array.len()) {
            (1, 1) => {
                if (!left_array.validity().all_valid() || !right_array.validity().all_valid())
                    || (!(self.function_set.scalar_cmp_scalar)(left_array, right_array))
                {
                    selection.mutate().set_all_invalid(leaf_input.len());
                }
            }
            (1, _) => {
                (self.function_set.scalar_cmp_array)(selection, right_array, left_array, output);
            }
            (_, 1) => {
                (self.function_set.array_cmp_scalar)(selection, left_array, right_array, output)
            }
            (_, _) => {
                (self.function_set.array_cmp_array)(selection, left_array, right_array, output)
            }
        }

        // Boolean expression, check the selection
        debug_assert!(selection.is_empty() || selection.len() == leaf_input.len());

        Ok(())
    }
}
