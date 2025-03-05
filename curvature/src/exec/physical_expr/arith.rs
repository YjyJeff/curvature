//! Expression to perform basic arithmetic like `+`/`-`/`*`/`/`/`%`

use data_block::array::ArrayImpl;
use data_block::bitmap::Bitmap;
use data_block::block::DataBlock;
use data_block::types::LogicalType;
use snafu::{Snafu, ensure};
use std::fmt::Debug;
use std::sync::Arc;

use crate::common::expr_operator::arith::{
    ArithFunctionSet, ArithOperator, infer_arithmetic_func_set,
};
use crate::common::profiler::ScopedTimerGuard;
use crate::exec::physical_expr::constant::Constant;

use super::executor::ExprExecCtx;
use super::utils::CompactExprDisplayWrapper;
use super::{ExprResult, PhysicalExpr, Stringify, execute_binary_children};

#[allow(missing_docs)]
#[derive(Debug, Snafu)]
pub enum ArithError {
    #[snafu(display(
        "Perform `{}` arithmetic between `{:?}` and `{:?}` is not supported",
        op,
        left,
        right
    ))]
    UnsupportedInputs {
        left: LogicalType,
        right: LogicalType,
        op: ArithOperator,
    },
    #[snafu(display(
        "Perform `{}` arithmetic between two constants(left: `{}`, right: `{}`) makes no sense. Planner/Optimizer should handle it in advance",
        op,
        left,
        right
    ))]
    TwoConstants {
        left: String,
        right: String,
        op: ArithOperator,
    },
}

type Result<T> = std::result::Result<T, ArithError>;

/// Arithmetic expression
#[derive(Debug)]
pub struct Arith {
    children: [Arc<dyn PhysicalExpr>; 2],
    op: ArithOperator,
    /// Function set used to perform arithmetic. It is determined by the children and op in
    /// the constructor phase
    ///
    /// Note that we introduce an extra indirection here by function pointer. Writing code
    /// this way is elegant and benchmark show that the impact of this extra indirection
    /// if negligible 😊
    function_set: ArithFunctionSet,
    alias: String,
}

impl Arith {
    /// Create a new [`Arith`] expression
    pub fn try_new(
        left: Arc<dyn PhysicalExpr>,
        op: ArithOperator,
        right: Arc<dyn PhysicalExpr>,
    ) -> Result<Self> {
        Self::try_new_with_alias(left, op, right, String::new())
    }

    /// Create a new [`Arith`] expression with alias
    pub fn try_new_with_alias(
        left: Arc<dyn PhysicalExpr>,
        op: ArithOperator,
        right: Arc<dyn PhysicalExpr>,
        alias: String,
    ) -> Result<Self> {
        let function_set = if let Some(function_set) =
            infer_arithmetic_func_set(left.output_type(), right.output_type(), op)
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

impl Stringify for Arith {
    fn name(&self) -> &'static str {
        self.op.ident()
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} -> {:?}",
            self.op.ident(),
            self.function_set.output_type
        )
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

impl PhysicalExpr for Arith {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn output_type(&self) -> &LogicalType {
        &self.function_set.output_type
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

        // Execute arith
        let _profile_guard = ScopedTimerGuard::new(&mut exec_ctx.metric.exec_time);
        let rows_count = selection.count_ones().unwrap_or(leaf_input.len()) as u64;
        exec_ctx.metric.rows_count += rows_count;
        let left_array = unsafe { exec_ctx.intermediate_block.get_array_unchecked(0) };
        let right_array = unsafe { exec_ctx.intermediate_block.get_array_unchecked(1) };

        match (left_array.len(), right_array.len()) {
            (1, 1) => (self.function_set.scalar_arith_scalar)(left_array, right_array, output),
            (1, _) => {
                todo!()
            }
            (_, 1) => {
                (self.function_set.array_arith_scalar)(selection, left_array, right_array, output);
            }
            (_, _) => {
                // Both of them are array, they must have same length.
                // Actually, we do do need this check, `execute_binary_children` already checked it
                (self.function_set.array_arith_array)(selection, left_array, right_array, output);
            }
        }

        debug_assert!(output.len() == 1 || output.len() == leaf_input.len());

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use data_block::{array::Int32Array, types::Array};

    use crate::exec::physical_expr::field_ref::FieldRef;

    use super::*;

    fn create_arith() -> Arith {
        Arith::try_new(
            Arc::new(FieldRef::new(0, LogicalType::Integer, "f0".to_string())),
            ArithOperator::Add,
            Arc::new(FieldRef::new(1, LogicalType::Integer, "f1".to_string())),
        )
        .unwrap()
    }

    fn assert_integer_array(output: &ArrayImpl, gt: &[Option<i32>]) {
        if let ArrayImpl::Int32(output) = &output {
            assert_eq!(output.iter().collect::<Vec<_>>(), gt);
        } else {
            panic!("Output must be Int32")
        }
    }

    #[test]
    fn test_arith_scalars() {
        let arith = create_arith();
        let mut selection = Bitmap::new();
        let mut exec_ctx = ExprExecCtx::new(&arith);

        let input = DataBlock::try_new(
            vec![
                ArrayImpl::Int32(Int32Array::from_values_iter([-4])),
                ArrayImpl::Int32(Int32Array::from_values_iter([8])),
            ],
            100,
        )
        .unwrap();

        let mut output = ArrayImpl::new(LogicalType::Integer);

        arith
            .execute(&input, &mut selection, &mut exec_ctx, &mut output)
            .unwrap();

        assert_integer_array(&output, &[Some(4)]);

        // One of them is null
        let input = DataBlock::try_new(
            vec![
                ArrayImpl::Int32(Int32Array::from_values_iter([1])),
                ArrayImpl::Int32(Int32Array::from_iter([None])),
            ],
            3,
        )
        .unwrap();
        arith
            .execute(&input, &mut selection, &mut exec_ctx, &mut output)
            .unwrap();

        assert_integer_array(&output, &[None]);
    }

    #[test]
    fn test_array_arith_scalar() {
        let arith = create_arith();
        let mut selection = Bitmap::new();
        let mut exec_ctx = ExprExecCtx::new(&arith);

        let input = DataBlock::try_new(
            vec![
                ArrayImpl::Int32(Int32Array::from_values_iter([-1, 0, 1])),
                ArrayImpl::Int32(Int32Array::from_values_iter([1])),
            ],
            3,
        )
        .unwrap();

        let mut output = ArrayImpl::new(LogicalType::Integer);

        arith
            .execute(&input, &mut selection, &mut exec_ctx, &mut output)
            .unwrap();

        assert_integer_array(&output, &[Some(0), Some(1), Some(2)]);

        // Add null
        let input = DataBlock::try_new(
            vec![
                ArrayImpl::Int32(Int32Array::from_values_iter([-1, 0, 1])),
                ArrayImpl::Int32(Int32Array::from_iter([None])),
            ],
            3,
        )
        .unwrap();

        arith
            .execute(&input, &mut selection, &mut exec_ctx, &mut output)
            .unwrap();

        assert_integer_array(&output, &[None, None, None]);
    }
}
