//! Check the input is null

use data_block::array::ArrayImpl;
use data_block::bitmap::Bitmap;
use data_block::block::DataBlock;
use data_block::compute::null::is_null;
use data_block::types::{LogicalType, PhysicalType};
use snafu::{ensure, ResultExt, Snafu};
use std::ops::DerefMut;
use std::sync::Arc;

use super::executor::ExprExecCtx;
use super::field_ref::FieldRef;
use super::utils::CompactExprDisplayWrapper;
use super::{ExecuteSnafu, ExprResult, PhysicalExpr, Stringify};

/// Error returned by creating the [`IsNull`]
#[derive(Debug, Snafu)]
#[snafu(display("Input of the `IsNull` is a boolean expression: `{input}`, it makes no sense because it always returns True/False. Handle it in the planner/optimizer instead"))]
pub struct BooleanExprInputError {
    /// Input expression
    input: String,
}

/// The expression used to check the input is (not) null
#[derive(Debug)]
pub struct IsNull<const NOT: bool> {
    output_type: LogicalType,
    children: Vec<Arc<dyn PhysicalExpr>>,
}

impl<const NOT: bool> IsNull<NOT> {
    /// Create a new [`IsNull`]
    pub fn try_new(input: Arc<dyn PhysicalExpr>) -> Result<Self, BooleanExprInputError> {
        ensure!(
            input.output_type().physical_type() != PhysicalType::Boolean
                || input.as_any().downcast_ref::<FieldRef>().is_some(),
            BooleanExprInputSnafu {
                input: CompactExprDisplayWrapper::new(&*input).to_string()
            }
        );

        Ok(Self {
            output_type: LogicalType::Boolean,
            children: vec![input],
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
        write!(f, "{}", self.name())
    }

    fn compact_display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        self.children[0].compact_display(f)?;
        write!(f, ") {}", self.name())
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
        // Execute child
        let mut guard = exec_ctx.intermediate_block.mutate_single_array();
        self.children[0]
            .execute(
                leaf_input,
                selection,
                &mut exec_ctx.children[0],
                guard.deref_mut(),
            )
            .boxed()
            .with_context(|_| ExecuteSnafu {
                expr: CompactExprDisplayWrapper::new(self).to_string(),
            })?;

        // Execute is null
        // SAFETY: Expression executor guarantees the selection is either empty or has same
        // length with input array(execute expression guarantees the output has same length
        // with input)
        unsafe { is_null::<NOT>(selection, &guard) };

        // Boolean expression, check the selection
        debug_assert!(selection.is_empty() || selection.len() == leaf_input.len());

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

        let leaf_input = DataBlock::try_new(vec![ArrayImpl::Int32(Int32Array::from_iter([
            Some(-1),
            None,
            Some(100),
            None,
        ]))])
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
