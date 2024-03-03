//! Executor that execute the physical expression

use std::sync::Arc;

use data_block::array::ArrayImpl;
use data_block::block::DataBlock;

use super::{ExprResult, PhysicalExpr};
use crate::common::profiler::Profiler;

/// Executor that executes set of expressions
///
/// It can only execute the expressions that create it! In ideal,
/// it should contain the reference to the expressions that create
/// it. However, it will cause the self-referential structure
#[derive(Debug)]
pub struct ExprExecutor {
    ctx: Vec<ExprExecCtx>,
}

impl ExprExecutor {
    /// Create a new [`ExprExecutor`] that can execute the input exprs
    pub fn new(exprs: &[Arc<dyn PhysicalExpr>]) -> Self {
        Self {
            ctx: exprs.iter().map(|expr| ExprExecCtx::new(&**expr)).collect(),
        }
    }

    /// Execute the expression
    #[inline]
    pub fn execute(
        &mut self,
        exprs: &[Arc<dyn PhysicalExpr>],
        input: &DataBlock,
        output: &mut DataBlock,
    ) -> ExprResult<()> {
        debug_assert_eq!(self.ctx.len(), exprs.len());
        debug_assert_eq!(self.ctx.len(), output.num_arrays());

        let guard = output.mutate_arrays();
        let mutate_func = |output: &mut [ArrayImpl]| {
            exprs
                .iter()
                .zip(self.ctx.iter_mut())
                .zip(output)
                .try_for_each(|((expr, exec_ctx), output)| expr.execute(input, exec_ctx, output))
        };

        // SAFETY: expressions process arrays with same length will produce arrays with same length
        unsafe { guard.mutate(mutate_func) }
    }
}

/// Context that holds the intermediate arrays of the expression
///
/// This context has the same structure with the expression that
/// create it. In ideal, we should contain a reference to the
/// expression that create it. However, this will cause the
/// self-referential structure. Therefore, this context is not
/// self-contained.  It is the users responsibility to only
/// use the instance with the the expression that create it!
///
/// FIXME: Check we should profile or not dynamically
#[cfg_attr(not(feature = "profile"), allow(dead_code))]
#[derive(Debug)]
pub struct ExprExecCtx {
    /// The data block that **children** of this expression will
    /// write to. It is also used as input of this expression
    pub intermediate_block: DataBlock,
    /// Children contexts
    pub children: Vec<ExprExecCtx>,
    /// Profiler for profiling the cpu time used in the expression
    pub profiler: Profiler,
}

impl ExprExecCtx {
    /// Create a new [`ExprExecCtx`]
    pub fn new(expr: &dyn PhysicalExpr) -> Self {
        // Create intermediate array for the children
        let (children, arrays) = expr
            .children()
            .iter()
            .map(|child_expr| {
                (
                    Self::new(&**child_expr),
                    ArrayImpl::new(child_expr.output_type().clone()),
                )
            })
            .unzip();

        // SAFETY: ArrayImpl::new return empty array
        Self {
            intermediate_block: unsafe { DataBlock::new_unchecked(arrays, 0) },
            children,
            profiler: Profiler::new(),
        }
    }
}
