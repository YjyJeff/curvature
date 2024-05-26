//! Expression that contains the user literally specified constant

use std::sync::Arc;

use data_block::array::ArrayImpl;
use data_block::bitmap::Bitmap;
use data_block::block::DataBlock;
use data_block::element::ElementImpl;
use data_block::types::LogicalType;

use super::executor::ExprExecCtx;
use super::{ExecuteSnafu, ExprResult, PhysicalExpr, Stringify};

/// Expression that contains the user literally specified constant
#[derive(Debug)]
pub struct Constant {
    logical_type: LogicalType,
    value: ElementImpl,
    children: Vec<Arc<dyn PhysicalExpr>>,
}

// SAFETY: It is read only. We do not provide any method to modify it
unsafe impl Send for Constant {}
unsafe impl Sync for Constant {}

impl Stringify for Constant {
    fn name(&self) -> &'static str {
        "Constant"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.value)
    }

    fn compact_display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display(f)
    }
}

impl PhysicalExpr for Constant {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn children(&self) -> &[Arc<dyn PhysicalExpr>] {
        &self.children
    }

    fn output_type(&self) -> &LogicalType {
        &self.logical_type
    }

    fn execute(
        &self,
        leaf_input: &DataBlock,
        _selection: &mut Bitmap,
        _exec_ctx: &mut ExprExecCtx,
        output: &mut ArrayImpl,
    ) -> ExprResult<()> {
        todo!()
    }
}
