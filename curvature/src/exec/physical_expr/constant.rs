//! Expression that contains the user literally specified constant

use std::sync::Arc;

use data_block::array::ArrayImpl;
use data_block::bitmap::Bitmap;
use data_block::block::DataBlock;
use data_block::types::LogicalType;
use snafu::{ResultExt, Snafu, ensure};

use super::executor::ExprExecCtx;
use super::{ExecuteSnafu, ExprResult, PhysicalExpr, Stringify};
use crate::common::profiler::ScopedTimerGuard;

/// Error returned by creating the [`Constant`] expression
#[derive(Debug, Snafu)]
#[snafu(display(
    "Constant expression expects a `ConstantArray`, however, the `FlatArray`: `{:?}` is accepted",
    array
))]
pub struct ConstantError {
    array: ArrayImpl,
}

/// Expression that contains the user literally specified constant
#[derive(Debug)]
pub struct Constant {
    logical_type: LogicalType,
    constant: ArrayImpl,
    children: [Arc<dyn PhysicalExpr>; 0],
    alias: String,
}

/// SAFETY:
///
/// - The constant array is not referenced by any array and it does
///   no reference other array. It is the unique owner of the memory
///
/// - It is read only
unsafe impl Send for Constant {}
unsafe impl Sync for Constant {}

impl Constant {
    /// Create a [`Constant expression`]
    ///
    /// # Safety
    ///
    /// The `constant` should not reference to other array and should not
    /// referenced by any array. It is the unique owner of the memory
    pub unsafe fn try_new(constant: ArrayImpl) -> Result<Self, ConstantError> {
        unsafe { Self::try_new_with_alias(constant, String::new()) }
    }

    /// Create a [`Constant expression`] with alias
    ///
    /// # Safety
    ///
    /// The `constant` should not reference to other array and should not
    /// referenced by any array. It is the unique owner of the memory
    pub unsafe fn try_new_with_alias(
        constant: ArrayImpl,
        alias: String,
    ) -> Result<Self, ConstantError> {
        ensure!(constant.len() == 1, ConstantSnafu { array: constant });
        let logical_type = constant.logical_type().to_owned();
        Ok(Self {
            logical_type,
            constant,
            children: [],
            alias,
        })
    }
}

impl Stringify for Constant {
    fn name(&self) -> &'static str {
        "Constant"
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Constant({:?}) -> {:?}",
            unsafe { self.constant.get_value_unchecked(0) },
            self.logical_type
        )
    }

    fn compact_display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Constant({:?})", unsafe {
            self.constant.get_value_unchecked(0)
        },)
    }

    fn alias(&self) -> &str {
        &self.alias
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
        exec_ctx: &mut ExprExecCtx,
        output: &mut ArrayImpl,
    ) -> ExprResult<()> {
        let _profile_guard = ScopedTimerGuard::new(&mut exec_ctx.metric.exec_time);
        exec_ctx.metric.rows_count += leaf_input.len() as u64;
        output
            .reference(&self.constant)
            .boxed()
            .with_context(|_| ExecuteSnafu {
                expr: format!("{:?}", self),
            })
    }
}
