//! Check the input is null

use data_block::array::{ArrayImpl, BooleanArray};
use data_block::block::DataBlock;
use data_block::compute::null::is_null;
use data_block::types::LogicalType;
use snafu::ResultExt;
use std::ops::DerefMut;
use std::sync::Arc;

use super::executor::ExprExecCtx;
use super::utils::CompactExprDisplayWrapper;
use super::{ExecuteSnafu, PhysicalExpr, Result, Stringify};

/// The expression used to check the input is (not) null
#[derive(Debug)]
pub struct IsNull<const NOT: bool> {
    output_type: LogicalType,
    children: Vec<Arc<dyn PhysicalExpr>>,
}

impl<const NOT: bool> IsNull<NOT> {
    /// Create a new [`IsNull`]
    pub fn new(input: Arc<dyn PhysicalExpr>) -> Self {
        Self {
            output_type: LogicalType::Boolean,
            children: vec![input],
        }
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
        exec_ctx: &mut ExprExecCtx,
        output: &mut ArrayImpl,
    ) -> Result<()> {
        let output: &mut BooleanArray = output
            .try_into()
            .expect("Output array of the is_null should be `BooleanArray`");

        // Execute child
        let mut guard = exec_ctx.intermediate_block.mutate_single_array();
        self.children[0]
            .execute(leaf_input, &mut exec_ctx.children[0], guard.deref_mut())
            .boxed()
            .with_context(|_| ExecuteSnafu {
                expr: CompactExprDisplayWrapper::new(self).to_string(),
            })?;

        unsafe { is_null::<NOT>(&guard, output) };

        Ok(())
    }
}
