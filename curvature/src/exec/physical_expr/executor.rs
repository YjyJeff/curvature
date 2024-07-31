//! Executor that execute the physical expression

use std::fmt::Display;
use std::sync::Arc;
use std::time::Duration;

use data_block::array::ArrayImpl;
use data_block::bitmap::{BitStore, Bitmap};
use data_block::block::{DataBlock, MutateArrayError};
use data_block::types::{Array, LogicalType};

use super::{is_boolean_expr, ExprError, ExprResult, PhysicalExpr};
use crate::exec::physical_expr::conjunction::OrConjunction;
use crate::exec::physical_expr::field_ref::FieldRef;
use crate::exec::physical_expr::utils::CompactExprDisplayWrapper;
use crate::tree_node::{TreeNode, TreeNodeRecursion};

/// Executor that executes set of expressions
///
/// It can only execute the expressions that create it! In ideal,
/// it should contain the reference to the expressions that create
/// it. However, it will cause the self-referential structure
#[derive(Debug)]
pub struct ExprsExecutor {
    /// Executors
    executors: Vec<Executor>,
}

/// Executor that execute a single expression
#[derive(Debug)]
pub struct Executor {
    /// Execution context
    ctx: ExprExecCtx,
    /// Selection map for the expression
    selection: Bitmap,
}

impl Executor {
    /// Create an [`Executor`]
    pub fn new(expr: &dyn PhysicalExpr) -> Self {
        Executor {
            ctx: ExprExecCtx::new(expr),
            selection: Bitmap::new(),
        }
    }

    /// Execute a boolean expression. Called by the [Filter] operator. It guarantees
    /// the expr is **not** a [Constant] expression
    ///
    /// [Filter]: crate::exec::physical_operator::filter::Filter
    /// [Constant]: crate::exec::physical_expr::constant::Constant
    pub fn execute_predicate(
        &mut self,
        input: &DataBlock,
        expr: &dyn PhysicalExpr,
        output: &mut ArrayImpl,
    ) -> ExprResult<&Bitmap> {
        debug_assert_eq!(expr.output_type(), &LogicalType::Boolean);
        // Clear the selection. Such that all of the elements are selected
        self.selection.mutate().clear();

        if let Some(field_ref) = expr.as_any().downcast_ref::<FieldRef>() {
            // SAFETY: Selection is empty
            unsafe {
                field_ref.copy_into_selection(input, &mut self.selection);
            }
        } else {
            expr.execute(input, &mut self.selection, &mut self.ctx, output)?;
        }

        Ok(&self.selection)
    }

    /// Get the execution time of the expression
    #[inline]
    pub fn exec_time(&self) -> Duration {
        self.ctx.cumulative_exec_time()
    }

    pub(crate) fn displayable_expr_with_metric<'a, const IN_FILTER: bool>(
        &'a self,
        expr: &'a dyn PhysicalExpr,
    ) -> DisplayExprWithMetric<'a, IN_FILTER> {
        DisplayExprWithMetric {
            expr,
            ctx: &self.ctx,
        }
    }
}

impl ExprsExecutor {
    /// Create a new [`ExprsExecutor`] that can execute the input exprs
    pub fn new(exprs: &[Arc<dyn PhysicalExpr>]) -> Self {
        Self {
            executors: exprs.iter().map(|expr| Executor::new(&**expr)).collect(),
        }
    }

    /// Execute the expression
    ///
    /// Note that executing the expression does not change the length. Therefore, the
    /// `output` has same length with `input`
    pub fn execute(
        &mut self,
        exprs: &[Arc<dyn PhysicalExpr>],
        input: &DataBlock,
        output: &mut DataBlock,
    ) -> ExprResult<()> {
        debug_assert_eq!(self.executors.len(), exprs.len());
        debug_assert_eq!(self.executors.len(), output.num_arrays());

        let guard = output.mutate_arrays(input.len());
        let mutate_func = |output: &mut [ArrayImpl]| {
            exprs
                .iter()
                .zip(self.executors.iter_mut())
                .zip(output)
                .try_for_each(|((expr, executor), output)| {
                    // Clear the selection. Such that all of the elements are selected
                    executor.selection.mutate().clear();
                    expr.execute(input, &mut executor.selection, &mut executor.ctx, output)?;
                    // Boolean expressions, except the FieldRef and Constant, will store the result in the selection.
                    // Copy it to the output
                    if is_boolean_expr(&**expr) {
                        let ArrayImpl::Boolean(output) = output else {
                            panic!("`{}` is a boolean expression. However the output array is `{}`Array", CompactExprDisplayWrapper::new(&**expr), output.ident())
                        };
                        unsafe{
                            output.validity_mut().mutate().clear();
                            let mut guard = output.data_mut().mutate();
                            let uninitialized = guard.clear_and_resize(input.len());
                            if executor.selection.all_valid(){
                                // All valid, the selection array maybe empty
                                uninitialized.iter_mut().for_each(|v| *v = BitStore::MAX);
                            }else{
                                // Compiler will optimize it to memory copy
                                debug_assert_eq!(executor.selection.len(), input.len());
                                uninitialized.iter_mut().zip(executor.selection.as_raw_slice()).for_each(|(dst, &src)| {
                                    *dst = src;
                                })
                            }
                        }
                    }
                    Ok::<_, ExprError>(())
                })
        };

        guard.mutate(mutate_func).map_err(|err| match err {
            MutateArrayError::Inner { source } => source,
            MutateArrayError::Length { inner } => {
                let expr = &exprs[inner.index];
                ExprError::Execute {
                    expr: CompactExprDisplayWrapper::new(&**expr).to_string(),
                    source: format!(
                        "Expression produce FlatArray with length: `{}`, it does not equal to the input data block length: `{}`",
                        inner.array_len,
                        inner.length
                    )
                    .into(),
                }
            }
        })
    }

    /// Get the execution times for each expression
    pub fn execution_times(&self) -> impl Iterator<Item = Duration> + '_ {
        self.executors
            .iter()
            .map(|executor| executor.ctx.cumulative_exec_time())
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
#[derive(Debug)]
pub struct ExprExecCtx {
    /// The data block that **children** of this expression will
    /// write to. It is also used as input of this expression
    pub intermediate_block: DataBlock,
    /// Selection arrays used by the [`OrConjunction`], it requires each
    /// child has its own selection array
    pub selections: Vec<Bitmap>,
    /// Children contexts
    pub children: Vec<ExprExecCtx>,
    /// Metric
    pub metric: ExprMetric,
}

/// Metric of the expression
#[derive(Debug, Default)]
pub struct ExprMetric {
    /// Duration used to execute the expression. Expression should
    /// profile it by itself.
    ///
    /// Note: It should only contains the execution time of the current expression,
    /// does not include the execution time of its children
    pub exec_time: Duration,
    /// Number of rows the expression has processed
    pub rows_count: u64,
    /// Number or rows the expression has filtered out. Used by the boolean expression
    pub filter_out_count: u64,
}

impl ExprExecCtx {
    /// Create a new [`ExprExecCtx`]
    pub fn new(expr: &dyn PhysicalExpr) -> Self {
        // Create intermediate array for the children

        let selections = if expr.as_any().downcast_ref::<OrConjunction>().is_some() {
            (0..expr.children().len())
                .map(|_| Bitmap::new())
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let (arrays, children) = expr
            .children()
            .iter()
            .map(|child_expr| {
                (
                    ArrayImpl::new(child_expr.output_type().clone()),
                    Self::new(&**child_expr),
                )
            })
            .unzip();

        // SAFETY: ArrayImpl::new return empty array
        Self {
            intermediate_block: unsafe { DataBlock::new_unchecked(arrays, 0) },
            selections,
            children,
            metric: ExprMetric::default(),
        }
    }

    /// Get the cumulative execution time of the expression, including its children execution
    /// time
    pub fn cumulative_exec_time(&self) -> Duration {
        let mut cumulative = self.metric.exec_time;
        for child in &self.children {
            cumulative += child.cumulative_exec_time();
        }
        cumulative
    }
}

/// Displayable expression with its metric
pub(crate) struct DisplayExprWithMetric<'a, const IN_FILTER: bool> {
    expr: &'a dyn PhysicalExpr,
    ctx: &'a ExprExecCtx,
}

impl<const IN_FILTER: bool> Display for DisplayExprWithMetric<'_, IN_FILTER> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.expr.display(f)?;
        write!(
            f,
            ": cumulative_exec_time: `{:?}`. exec_time: `{:?}`. process rows: `{}`",
            self.ctx.cumulative_exec_time(),
            self.ctx.metric.exec_time,
            self.ctx.metric.rows_count,
        )?;

        if IN_FILTER && is_boolean_expr(self.expr) {
            write!(
                f,
                ". filter_out_rows: `{}`",
                self.ctx.metric.filter_out_count
            )
        } else {
            Ok(())
        }
    }
}

impl<const IN_FILTER: bool> TreeNode for DisplayExprWithMetric<'_, IN_FILTER> {
    fn apply_children<E, F: FnMut(&Self) -> std::result::Result<TreeNodeRecursion, E>>(
        &self,
        mut f: F,
    ) -> std::result::Result<TreeNodeRecursion, E> {
        let mut tnr = TreeNodeRecursion::Continue;
        for (child, child_ctx) in self.expr.children().iter().zip(self.ctx.children.iter()) {
            tnr = f(&Self {
                expr: &**child,
                ctx: child_ctx,
            })?;
            match tnr {
                TreeNodeRecursion::Continue | TreeNodeRecursion::Jump => {}
                TreeNodeRecursion::Stop => return Ok(TreeNodeRecursion::Stop),
            }
        }
        Ok(tnr)
    }
}

#[cfg(test)]
mod tests {
    use crate::exec::physical_expr::regex_match::RegexMatch;

    use super::*;
    use data_block::{
        array::{BooleanArray, Int32Array, StringArray},
        types::LogicalType,
    };

    #[test]
    fn test_exprs_executor() {
        let exprs: Vec<Arc<dyn PhysicalExpr>> = vec![
            Arc::new(FieldRef::new(0, LogicalType::Boolean, "col0".to_string())),
            Arc::new(
                RegexMatch::<false, false>::try_new(
                    Arc::new(FieldRef::new(1, LogicalType::VarChar, "col1".to_string())),
                    ".s".to_string(),
                )
                .unwrap(),
            ),
            Arc::new(FieldRef::new(2, LogicalType::Integer, "col2".to_string())),
        ];

        let input = DataBlock::try_new(
            vec![
                ArrayImpl::Boolean(BooleanArray::from_iter([
                    Some(false),
                    None,
                    Some(true),
                    None,
                ])),
                ArrayImpl::String(StringArray::from_values_iter([
                    "yes", "no", "expr", "executor",
                ])),
                ArrayImpl::Int32(Int32Array::from_iter([Some(-1), Some(2), None, None])),
            ],
            4,
        )
        .unwrap();

        let mut output = DataBlock::with_logical_types(vec![
            LogicalType::Boolean,
            LogicalType::Boolean,
            LogicalType::Integer,
        ]);

        let mut executor = ExprsExecutor::new(&exprs);
        executor.execute(&exprs, &input, &mut output).unwrap();
        let expected = expect_test::expect![[r#"
            DataBlock {
                arrays: [
                    Boolean(
                        BooleanArray { logical_type: Boolean, len: 4, data: [
                            Some(
                                false,
                            ),
                            None,
                            Some(
                                true,
                            ),
                            None,
                        ]},
                    ),
                    Boolean(
                        BooleanArray { logical_type: Boolean, len: 4, data: [
                            Some(
                                true,
                            ),
                            Some(
                                false,
                            ),
                            Some(
                                false,
                            ),
                            Some(
                                false,
                            ),
                        ]},
                    ),
                    Int32(
                        Int32Array { logical_type: Integer, len: 4, data: [
                            Some(
                                -1,
                            ),
                            Some(
                                2,
                            ),
                            None,
                            None,
                        ]},
                    ),
                ],
                length: 4,
            }
        "#]];
        expected.assert_debug_eq(&output);

        let input = DataBlock::try_new(
            vec![
                ArrayImpl::Boolean(BooleanArray::from_iter([None, Some(true), Some(false)])),
                ArrayImpl::String(StringArray::from_values_iter(["exprs", "executors", "yes"])),
                ArrayImpl::Int32(Int32Array::from_iter([None, Some(-1), Some(2)])),
            ],
            3,
        )
        .unwrap();

        let mut executor = ExprsExecutor::new(&exprs);
        executor.execute(&exprs, &input, &mut output).unwrap();

        let expected = expect_test::expect![[r#"
            DataBlock {
                arrays: [
                    Boolean(
                        BooleanArray { logical_type: Boolean, len: 3, data: [
                            None,
                            Some(
                                true,
                            ),
                            Some(
                                false,
                            ),
                        ]},
                    ),
                    Boolean(
                        BooleanArray { logical_type: Boolean, len: 3, data: [
                            Some(
                                true,
                            ),
                            Some(
                                true,
                            ),
                            Some(
                                true,
                            ),
                        ]},
                    ),
                    Int32(
                        Int32Array { logical_type: Integer, len: 3, data: [
                            None,
                            Some(
                                -1,
                            ),
                            Some(
                                2,
                            ),
                        ]},
                    ),
                ],
                length: 3,
            }
        "#]];
        expected.assert_debug_eq(&output);
    }
}
