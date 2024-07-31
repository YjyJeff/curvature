//! Check the input is null

use std::sync::Arc;

use data_block::array::ArrayImpl;
use data_block::bitmap::Bitmap;
use data_block::block::DataBlock;
use data_block::compute::null::is_null;
use data_block::types::LogicalType;
use snafu::{ensure, Snafu};

use crate::common::profiler::ScopedTimerGuard;
use crate::exec::physical_expr::constant::Constant;

use super::executor::ExprExecCtx;
use super::field_ref::FieldRef;
use super::utils::CompactExprDisplayWrapper;
use super::{execute_unary_child, ExprResult, PhysicalExpr, Stringify};

#[allow(missing_docs)]
#[derive(Debug, Snafu)]
pub enum IsNullError {
    #[snafu(display("Input of the `is_null` expression is a boolean expression: `{expr}`, it makes no sense. Handle it in the planner/optimizer"))]
    BooleanExpr { expr: String },
    #[snafu(display("Input of the `is_null` expression is a constant expression: `{expr}`, it makes no sense. Handle it in the planner/optimizer"))]
    ConstantExpr { expr: String },
}

/// The expression used to check the input is (not) null
#[derive(Debug)]
pub struct IsNull<const NOT: bool> {
    output_type: LogicalType,
    children: [Arc<dyn PhysicalExpr>; 1],
    alias: String,
}

impl<const NOT: bool> IsNull<NOT> {
    /// Try to create a new [`IsNull`] expression
    pub fn try_new(input: Arc<dyn PhysicalExpr>) -> Result<Self, IsNullError> {
        Self::try_new_with_alias(input, String::new())
    }

    /// Try to create a new [`IsNull`] expression with alias
    pub fn try_new_with_alias(
        input: Arc<dyn PhysicalExpr>,
        alias: String,
    ) -> Result<Self, IsNullError> {
        ensure!(
            input.output_type() != &LogicalType::Boolean
                || input.as_any().downcast_ref::<FieldRef>().is_some(),
            BooleanExprSnafu {
                expr: CompactExprDisplayWrapper::new(&*input).to_string()
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

impl<const NOT: bool> Stringify for IsNull<NOT> {
    fn name(&self) -> &'static str {
        if NOT {
            "IS NOT NULL"
        } else {
            "IS NULL"
        }
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} -> Boolean", self.name())
    }

    fn compact_display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        self.children[0].compact_display(f)?;
        write!(f, ") {}", self.name())
    }

    fn alias(&self) -> &str {
        &self.alias
    }
}

impl<const NOT: bool> PhysicalExpr for IsNull<NOT> {
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
        let rows_count = selection.count_ones().unwrap_or(leaf_input.len()) as u64;
        exec_ctx.metric.rows_count += rows_count;
        // Execute is null
        let array = unsafe { exec_ctx.intermediate_block.get_array_unchecked(0) };

        if array.len() == 1 {
            let valid = array.validity().all_valid();
            if NOT {
                if !valid {
                    selection.mutate().set_all_invalid(leaf_input.len());
                    // Selection must have same length with leaf_input
                    exec_ctx.metric.filter_out_count += rows_count;
                }
            } else if valid {
                selection.mutate().set_all_invalid(leaf_input.len());
                // Selection must have same length with leaf_input
                exec_ctx.metric.filter_out_count += rows_count;
            }
        } else {
            // SAFETY: Expression executor guarantees the selection is either empty or has same
            // length with input array(execute expression guarantees the output has same length
            // with input)
            unsafe { is_null::<NOT>(selection, array) };

            // Selection must have same length with leaf_input
            exec_ctx.metric.filter_out_count += rows_count - selection.count_ones().unwrap() as u64;

            // Boolean expression, check the selection
            debug_assert!(selection.len() == leaf_input.len());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::physical_expr::field_ref::FieldRef;
    use data_block::array::{BooleanArray, Int32Array};

    #[test]
    fn test_is_null() {
        let input = Arc::new(FieldRef::new(0, LogicalType::Integer, "col0".to_string()));

        let is_null = IsNull::<false>::try_new(input).unwrap();
        let mut exec_ctx = ExprExecCtx::new(&is_null);
        let mut output = ArrayImpl::Boolean(BooleanArray::try_new(LogicalType::Boolean).unwrap());
        let mut selection = Bitmap::new();

        let leaf_input = DataBlock::try_new(
            vec![ArrayImpl::Int32(Int32Array::from_iter([
                Some(-1),
                None,
                Some(100),
                None,
            ]))],
            4,
        )
        .unwrap();

        is_null
            .execute(&leaf_input, &mut selection, &mut exec_ctx, &mut output)
            .unwrap();

        assert_eq!(
            selection.iter().collect::<Vec<_>>(),
            [false, true, false, true]
        )
    }
}
