//! Not expression

use std::sync::Arc;

use data_block::array::ArrayImpl;
use data_block::bitmap::Bitmap;
use data_block::block::DataBlock;
use data_block::compute::logical::{and_inplace, and_not_inplace};
use data_block::types::{Array, LogicalType};
use snafu::{ensure, Snafu};

use crate::common::profiler::ScopedTimerGuard;
use crate::exec::physical_expr::{constant::Constant, utils::CompactExprDisplayWrapper};

use super::executor::ExprExecCtx;
use super::{execute_unary_child, ExprResult, PhysicalExpr, Stringify};

#[allow(missing_docs)]
#[derive(Debug, Snafu)]
pub enum NotError {
    #[snafu(display("`Not` expression requires the input should be a boolean expression. However the input `{}` returns `{:?}`", expr, expr_return_type))]
    NotBooleanExpr {
        expr: String,
        expr_return_type: LogicalType,
    },
    #[snafu(display("Input of the `not` expression is a constant expression: `{expr}`, it makes no sense. Handle it in the planner/optimizer"))]
    ConstantExpr { expr: String },
}

/// Not expression
#[derive(Debug)]
pub struct Not {
    output_type: LogicalType,
    children: [Arc<dyn PhysicalExpr>; 1],
    alias: String,
}

impl Not {
    /// Try to create a new [`Not`] expression
    pub fn try_new(input: Arc<dyn PhysicalExpr>) -> Result<Self, NotError> {
        Self::try_new_with_alias(input, String::new())
    }

    /// Try to create a new [`Not`] expression with alias
    pub fn try_new_with_alias(
        input: Arc<dyn PhysicalExpr>,
        alias: String,
    ) -> Result<Self, NotError> {
        ensure!(
            input.output_type() == &LogicalType::Boolean,
            NotBooleanExprSnafu {
                expr: CompactExprDisplayWrapper::new(&*input).to_string(),
                expr_return_type: input.output_type().to_owned(),
            }
        );

        if let Some(expr) = input.as_any().downcast_ref::<Constant>() {
            return ConstantExprSnafu {
                expr: format!("{:?}", expr),
            }
            .fail();
        }

        Ok(Self {
            output_type: LogicalType::Boolean,
            children: [input],
            alias,
        })
    }
}

impl Stringify for Not {
    fn name(&self) -> &'static str {
        "NOT"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NOT -> Boolean")
    }

    fn compact_display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NOT(")?;
        self.children[0].compact_display(f)?;
        write!(f, ")")
    }

    fn alias(&self) -> &str {
        &self.alias
    }
}

impl PhysicalExpr for Not {
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
        _output: &mut ArrayImpl,
    ) -> ExprResult<()> {
        execute_unary_child(self, leaf_input, selection, exec_ctx)?;

        let _profile_guard = ScopedTimerGuard::new(&mut exec_ctx.metric.exec_time);

        // Execute not
        let array = unsafe { exec_ctx.intermediate_block.get_array_unchecked(0) };
        let ArrayImpl::Boolean(array) = array else {
            panic!("Input array of the `NOT` expression must be `BooleanArray`")
        };

        // FIXME: Maybe we only need to flip the selected bits?
        unsafe {
            and_not_inplace(selection, array.data(), leaf_input.len());
            if selection.count_zeros() != leaf_input.len() {
                and_inplace(selection, array.validity());
            }
        }

        Ok(())
    }
}
