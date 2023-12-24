//! Executor that execute the physical expression

use data_block::array::ArrayImpl;
use data_block::block::DataBlock;

use super::physical_expr::{ExprResult, PhysicalExpr};
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
    pub fn new<I: Iterator<Item = impl AsRef<dyn PhysicalExpr>>>(exprs: I) -> Self {
        Self {
            ctx: exprs.map(|expr| ExprExecCtx::new(expr.as_ref())).collect(),
        }
    }

    /// Execute the expression
    #[inline]
    pub fn execute<I: Iterator<Item = impl AsRef<dyn PhysicalExpr>>>(
        &mut self,
        exprs: I,
        input: &DataBlock,
        output: &mut DataBlock,
    ) -> ExprResult<()> {
        exprs
            .zip(self.ctx.iter_mut())
            .zip(output.mutable_arrays())
            .try_for_each(|((expr, ctx), output)| execute(expr.as_ref(), ctx, input, output))
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
struct ExprExecCtx {
    /// The data block that **children** of this expression will
    /// write to. It is also used as input of this expression
    intermediate_block: DataBlock,
    /// Children contexts
    children: Vec<ExprExecCtx>,
    /// Profiler for profiling the cpu time used in the expression
    profiler: Profiler,
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

fn execute(
    expr: &dyn PhysicalExpr,
    ctx: &mut ExprExecCtx,
    leaf_input: &DataBlock,
    output: &mut ArrayImpl,
) -> ExprResult<()> {
    // Execute children
    expr.children()
        .iter()
        .zip(ctx.children.iter_mut())
        .zip(ctx.intermediate_block.mutable_arrays())
        .try_for_each(|((expr, ctx), output)| execute(&**expr, ctx, leaf_input, output))?;

    // execute expr
    #[cfg(feature = "profile")]
    let _guard = ctx.profiler.start_profile(leaf_input.length() as _);

    if ctx.children.is_empty() {
        // Leaf expression, take leaf input as input
        expr.execute(leaf_input, output)
    } else {
        // Non leaf expression, take ctx.block as input
        expr.execute(&ctx.intermediate_block, output)
    }
}
