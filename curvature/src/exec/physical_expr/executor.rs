//! Executor that execute the physical expression

use std::sync::Arc;
use std::time::Duration;

use data_block::bitmap::{BitStore, Bitmap};
use data_block::block::DataBlock;
use data_block::types::Array;
use data_block::{array::ArrayImpl, types::PhysicalType};

use super::{ExprResult, PhysicalExpr};
use crate::common::profiler::ScopedTimerGuard;
use crate::exec::physical_expr::conjunction::OrConjunction;
use crate::exec::physical_expr::field_ref::FieldRef;
use crate::exec::physical_expr::utils::CompactExprDisplayWrapper;

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

#[derive(Debug)]
struct Executor {
    /// Execution context
    ctx: ExprExecCtx,
    /// Selection map
    selection: Bitmap,
    /// Metrics to profile the execution time
    metric: Duration,
}

impl ExprsExecutor {
    /// Create a new [`ExprsExecutor`] that can execute the input exprs
    pub fn new(exprs: &[Arc<dyn PhysicalExpr>]) -> Self {
        Self {
            executors: exprs
                .iter()
                .map(|expr| Executor {
                    ctx: ExprExecCtx::new(&**expr),
                    selection: Bitmap::new(),
                    metric: Duration::default(),
                })
                .collect(),
        }
    }

    /// Execute the expression
    ///
    /// Note that executing the expression does not change the length. Therefore, the
    /// `output` has same length with `input`
    #[inline]
    pub fn execute(
        &mut self,
        exprs: &[Arc<dyn PhysicalExpr>],
        input: &DataBlock,
        output: &mut DataBlock,
    ) -> ExprResult<()> {
        debug_assert_eq!(self.executors.len(), exprs.len());
        debug_assert_eq!(self.executors.len(), output.num_arrays());

        let guard = output.mutate_arrays();
        let mutate_func = |output: &mut [ArrayImpl]| {
            exprs
                .iter()
                .zip(self.executors.iter_mut())
                .zip(output)
                .try_for_each(|((expr, executor), output)| {
                    let _guard = ScopedTimerGuard::new(&mut executor.metric);
                    // Clear the selection. Such that all of the elements are selected
                    executor.selection.mutate().clear();
                    expr.execute(input, &mut executor.selection, &mut executor.ctx, output)?;
                    // Boolean expressions, except the FieldRef, will store the result in the selection.
                    // Copy it to the output
                    if expr.output_type().physical_type() == PhysicalType::Boolean && expr.as_any().downcast_ref::<FieldRef>().is_none(){
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
                                uninitialized.iter_mut().zip(executor.selection.as_raw_slice()).for_each(|(dst, &src)| {
                                    *dst = src;
                                })
                            }
                        }
                    }
                    Ok(())
                })
        };

        // SAFETY: expressions process arrays with same length will produce arrays with same length
        unsafe {
            guard.mutate(mutate_func)?;
        }

        // Output and input must have same length
        debug_assert_eq!(output.len(), input.len());

        Ok(())
    }

    /// Get the execution times for each expression
    pub fn execution_times(&self) -> impl Iterator<Item = Duration> + '_ {
        self.executors.iter().map(|executor| executor.metric)
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
        }
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

        let input = DataBlock::try_new(vec![
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
        ])
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

        let input = DataBlock::try_new(vec![
            ArrayImpl::Boolean(BooleanArray::from_iter([None, Some(true), Some(false)])),
            ArrayImpl::String(StringArray::from_values_iter(["exprs", "executors", "yes"])),
            ArrayImpl::Int32(Int32Array::from_iter([None, Some(-1), Some(2)])),
        ])
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
