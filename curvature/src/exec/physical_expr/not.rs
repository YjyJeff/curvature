//! Not expression

use std::sync::Arc;

use data_block::array::ArrayImpl;
use data_block::bitmap::Bitmap;
use data_block::block::DataBlock;
use data_block::compute::logical::{and_inplace, and_not_inplace};
use data_block::types::{Array, LogicalType};
use snafu::{Snafu, ensure};

use crate::common::profiler::ScopedTimerGuard;
use crate::exec::physical_expr::{constant::Constant, utils::CompactExprDisplayWrapper};

use super::executor::ExprExecCtx;
use super::{ExprResult, PhysicalExpr, Stringify, execute_unary_child};

/// Error returned by creating the not expression
#[derive(Debug, Snafu)]
pub enum NotError {
    /// The input is not a boolean expression
    #[snafu(display(
        "`Not` expression requires the input should be a boolean expression. However the input `{}` returns `{:?}`",
        expr,
        expr_return_type
    ))]
    NotBooleanExpr {
        /// Input expression
        expr: String,
        /// Return type of the input expression
        expr_return_type: LogicalType,
    },
    /// Input is a constant expression
    #[snafu(display(
        "Input of the `not` expression is a constant expression: `{constant}`, it makes no sense. Handle it in the planner/optimizer"
    ))]
    ConstantExpr {
        /// The constant expression
        constant: String,
    },
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

        if let Some(constant) = input.as_any().downcast_ref::<Constant>() {
            return ConstantExprSnafu {
                constant: format!("{:?}", constant),
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

        // FIXME: Depend on the selectivity, maybe we only need to flip the selected bits
        unsafe {
            and_not_inplace(selection, array.data(), leaf_input.len());
            if selection.count_zeros() != leaf_input.len() {
                and_inplace(selection, array.validity());
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::physical_expr::field_ref::FieldRef;
    use data_block::array::BooleanArray;

    #[test]
    fn test_not() {
        let input = Arc::new(FieldRef::new(0, LogicalType::Boolean, "col0".to_string()));

        let not = Not::try_new(input).unwrap();

        let mut exec_ctx = ExprExecCtx::new(&not);
        let mut output = ArrayImpl::Boolean(BooleanArray::try_new(LogicalType::Boolean).unwrap());
        let mut selection = Bitmap::new();

        let leaf_input = DataBlock::try_new(
            vec![ArrayImpl::Boolean(BooleanArray::from_iter([
                Some(true),
                None,
                Some(false),
                None,
                None,
                Some(true),
                Some(true),
                Some(false),
                Some(false),
            ]))],
            9,
        )
        .unwrap();

        not.execute(&leaf_input, &mut selection, &mut exec_ctx, &mut output)
            .unwrap();

        assert_eq!(
            selection.iter().collect::<Vec<_>>(),
            [false, false, true, false, false, false, false, true, true]
        );
    }
}
