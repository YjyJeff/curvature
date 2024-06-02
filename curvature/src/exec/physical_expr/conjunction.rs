//! Conjunct boolean expressions

use std::sync::Arc;

use data_block::array::ArrayImpl;
use data_block::bitmap::Bitmap;
use data_block::block::DataBlock;
use data_block::compute::logical::{and_inplace, or_inplace};
use data_block::types::{LogicalType, PhysicalType};
use snafu::{ensure, ResultExt, Snafu};

use crate::exec::physical_expr::field_ref::FieldRef;

use super::constant::Constant as ConstantExpr;
use super::executor::ExprExecCtx;
use super::utils::{compact_display_expressions, CompactExprDisplayWrapper};
use super::{ExecuteSnafu, ExprResult, PhysicalExpr, Stringify};

/// Conjunct boolean expressions with `And` operation
pub type AndConjunction = Conjunction<true>;
/// Conjunct boolean expressions with `Or` operation
pub type OrConjunction = Conjunction<false>;

#[allow(missing_docs)]
#[derive(Debug, Snafu)]
pub enum ConjunctionError {
    #[snafu(display("Conjunction requires at least two input expressions. However, it accepts `{len}` inputs: `{inputs}`"))]
    TooFewInputs { inputs: String, len: usize },
    #[snafu(display("Conjunction requires the input should be boolean expression. However, the input `{input}` has logical type: `{logical_type:?}`"))]
    NotBooleanExpr {
        input: String,
        logical_type: LogicalType,
    },
    #[snafu(display(
        "`{}Conjunction` requires the input is flattened. However the input: `{}` is not flattened. Planner/Optimizer should flatten the inputs!",
        conjunction,
        input
    ))]
    NotFlattenedExpr {
        conjunction: &'static str,
        input: String,
    },
    #[snafu(display("Conjunction requires that input should not be constant expression, because it can be optimized away. Planner/Optimizer should handle it in advance"))]
    ConstantExpr,
}

/// Conjunct boolean expressions
///
/// TODO:
/// - Add metric and reorder the children expressions dynamically
#[derive(Debug)]
pub struct Conjunction<const IS_AND: bool> {
    output_type: LogicalType,
    children: Vec<Arc<dyn PhysicalExpr>>,
}

impl<const IS_AND: bool> Conjunction<IS_AND> {
    /// Create a new [`Conjunction`]
    ///
    /// # Notes:
    ///
    /// The inputs should be flattened: `And(a, And(b, c))` is not allowed. In this case,
    /// the inputs should be: `And(a, b, c)`. The reason is that:
    ///
    /// - We can efficiently reorder the expressions dynamically if the inputs is flattened
    ///
    /// - It is the planner/optimizers responsibility to flatten the inputs. The [`Conjunction`]
    /// only execute it efficiently
    pub fn try_new(inputs: Vec<Arc<dyn PhysicalExpr>>) -> Result<Self, ConjunctionError> {
        ensure!(
            inputs.len() >= 2,
            TooFewInputsSnafu {
                len: inputs.len(),
                inputs: if inputs.is_empty() {
                    "[]".to_string()
                } else {
                    format!("[{}]", CompactExprDisplayWrapper::new(&*inputs[0]))
                }
            }
        );

        inputs.iter().try_for_each(|expr| {
            ensure!(
                expr.output_type().physical_type() == PhysicalType::Boolean,
                NotBooleanExprSnafu {
                    input: CompactExprDisplayWrapper::new(&**expr).to_string(),
                    logical_type: expr.output_type().to_owned()
                }
            );
            ensure!(
                expr.as_any().downcast_ref::<Self>().is_none(),
                NotFlattenedExprSnafu {
                    conjunction: if IS_AND { "And" } else { "Or" },
                    input: CompactExprDisplayWrapper::new(&**expr).to_string()
                }
            );

            ensure!(
                expr.as_any().downcast_ref::<ConstantExpr>().is_none(),
                ConstantExprSnafu
            );

            Ok::<(), ConjunctionError>(())
        })?;

        Ok(Self {
            output_type: LogicalType::Boolean,
            children: inputs,
        })
    }
}

impl<const IS_AND: bool> Stringify for Conjunction<IS_AND> {
    fn name(&self) -> &'static str {
        if IS_AND {
            "And"
        } else {
            "Or"
        }
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }

    fn compact_display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}(", self.name())?;
        compact_display_expressions(f, &self.children)?;
        write!(f, ")")
    }
}

impl<const IS_AND: bool> PhysicalExpr for Conjunction<IS_AND> {
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
        debug_assert_eq!(
            exec_ctx.children.len(),
            exec_ctx.intermediate_block.num_arrays()
        );

        // Boolean expression, check the selection
        debug_assert!(selection.is_empty() || selection.len() == leaf_input.len());

        // SAFETY: The input block of the conjunction expression do not need to hold the
        // length variance. The reason is that: children expressions will write the result
        // to selection array. It is only used to store the temporal result!
        let arrays = unsafe { exec_ctx.intermediate_block.arrays_mut_unchecked() };

        if IS_AND {
            // And conjunction. It is pretty simple, all of the children expressions, except `FieldRef`,
            // write the result into the `selection` argument. The `selection` argument will
            // be passed to all of its children
            debug_assert!(exec_ctx.selections.is_empty());

            for ((child_expr, child_output), child_exec_ctx) in self
                .children
                .iter()
                .zip(arrays)
                .zip(exec_ctx.children.iter_mut())
            {
                if let Some(field_ref) = child_expr.as_any().downcast_ref::<FieldRef>() {
                    // SAFETY: expression executor guarantees the the selection is either
                    // all valid or has same length with output
                    unsafe { field_ref.copy_into_selection(leaf_input, selection) }
                } else {
                    child_expr
                        .execute(leaf_input, selection, child_exec_ctx, child_output)
                        .boxed()
                        .with_context(|_| ExecuteSnafu {
                            expr: CompactExprDisplayWrapper::new(self).to_string(),
                        })?;
                }

                // Now, selection array contains the result of the children expressions till now
                if selection.count_zeros() == leaf_input.len() {
                    // All of them is false. Early return, do not execute other children
                    return Ok(());
                }
            }
        } else {
            // Or conjunction, a little bit complicated. For each child expression, we need to
            // use a new all valid `Bitmap` as selection array. It is stored in the `exec_ctx`
            debug_assert_eq!(exec_ctx.children.len(), exec_ctx.selections.len());

            // SAFETY: all of them have same length
            unsafe {
                // The selection array that used to store the result of or. We choose the
                // first selection array as the final result. Extend the lifetime to cheat
                // the borrower checker
                let or_selection = &mut *(exec_ctx.selections.get_unchecked_mut(0) as *mut Bitmap);
                for index in 0..self.children.len() {
                    let child_expr = self.children().get_unchecked(index);
                    let child_selection = exec_ctx.selections.get_unchecked_mut(index);
                    let child_exec_ctx = exec_ctx.children.get_unchecked_mut(index);
                    let child_output = arrays.get_unchecked_mut(index);

                    // For each block, we need to set it to all valid first
                    child_selection.mutate().clear();

                    if let Some(field_ref) = child_expr.as_any().downcast_ref::<FieldRef>() {
                        // SAFETY: expression executor guarantees the the selection is either
                        // all valid or has same length with output
                        field_ref.copy_into_selection(leaf_input, child_selection)
                    } else {
                        child_expr
                            .execute(leaf_input, child_selection, child_exec_ctx, child_output)
                            .boxed()
                            .with_context(|_| ExecuteSnafu {
                                expr: CompactExprDisplayWrapper::new(self).to_string(),
                            })?;
                    }

                    // Now, child_selection array contains the result of the child expression
                    if child_selection.all_valid() {
                        // All of them are true. Early return, do not execute other children
                        selection.mutate().clear();
                        return Ok(());
                    }

                    if index != 0 {
                        // Combine the child selection into the or_selection.
                        // child selection is not empty here. If it is empty, it will perform
                        // early return above
                        debug_assert_eq!(or_selection.len(), child_selection.len());
                        or_inplace(or_selection, child_selection);

                        // Now, or_selection contains the or result till now.
                        if or_selection.all_valid() {
                            // All of them are true. Early return, do not execute other children
                            selection.mutate().clear();
                            return Ok(());
                        }
                    }
                }

                // Do not early return. And the selection with or_selection
                and_inplace(selection, or_selection);
            }
        }

        // Boolean expression, check the selection
        debug_assert!(selection.is_empty() || selection.len() == leaf_input.len());

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use data_block::array::{BooleanArray, StringArray};

    use super::*;
    use crate::exec::physical_expr::regex_match::RegexMatch;

    #[test]
    fn test_and_conjunction() {
        let inputs: Vec<Arc<dyn PhysicalExpr>> = vec![
            Arc::new(FieldRef::new(0, LogicalType::Boolean, "col0".to_string())),
            Arc::new(FieldRef::new(1, LogicalType::Boolean, "col1".to_string())),
            Arc::new(
                RegexMatch::<false, false>::try_new(
                    Arc::new(FieldRef::new(2, LogicalType::VarChar, "col2".to_string())),
                    "hi?".to_string(),
                )
                .unwrap(),
            ),
        ];
        let and_conjunction = AndConjunction::try_new(inputs).unwrap();
        let mut exec_ctx = ExprExecCtx::new(&and_conjunction);
        let mut selection = Bitmap::new();
        let mut output = ArrayImpl::Boolean(BooleanArray::try_new(LogicalType::Boolean).unwrap());

        let data_block = DataBlock::try_new(
            vec![
                ArrayImpl::Boolean(BooleanArray::from_iter([
                    Some(true),
                    Some(false),
                    Some(true),
                    Some(false),
                    None,
                    Some(true),
                ])),
                ArrayImpl::Boolean(BooleanArray::from_iter([
                    Some(true),
                    Some(true),
                    Some(true),
                    Some(false),
                    None,
                    None,
                ])),
                ArrayImpl::String(StringArray::from_values_iter([
                    "abc", "hi", "h", "haha", "yes", "wtf",
                ])),
            ],
            6,
        )
        .unwrap();

        and_conjunction
            .execute(&data_block, &mut selection, &mut exec_ctx, &mut output)
            .unwrap();

        assert_eq!(
            selection.iter().collect::<Vec<_>>(),
            [false, false, true, false, false, false]
        );

        // Early return
        let data_block = DataBlock::try_new(
            vec![
                ArrayImpl::Boolean(BooleanArray::from_iter([
                    Some(false),
                    Some(false),
                    Some(false),
                    Some(false),
                    None,
                    Some(true),
                ])),
                ArrayImpl::Boolean(BooleanArray::from_iter([
                    Some(true),
                    Some(true),
                    Some(true),
                    Some(false),
                    None,
                    None,
                ])),
                ArrayImpl::String(StringArray::from_values_iter([
                    "abc", "hi", "h", "haha", "yes", "wtf",
                ])),
            ],
            6,
        )
        .unwrap();

        let mut selection = Bitmap::new();

        and_conjunction
            .execute(&data_block, &mut selection, &mut exec_ctx, &mut output)
            .unwrap();

        assert_eq!(
            selection.iter().collect::<Vec<_>>(),
            [false, false, false, false, false, false]
        )
    }

    #[test]
    fn test_or_conjunction() {
        let inputs: Vec<Arc<dyn PhysicalExpr>> = vec![
            Arc::new(FieldRef::new(0, LogicalType::Boolean, "col0".to_string())),
            Arc::new(
                RegexMatch::<false, false>::try_new(
                    Arc::new(FieldRef::new(1, LogicalType::VarChar, "col1".to_string())),
                    "hi?".to_string(),
                )
                .unwrap(),
            ),
        ];
        let or_conjunction = OrConjunction::try_new(inputs).unwrap();
        let mut exec_ctx = ExprExecCtx::new(&or_conjunction);
        let mut selection = Bitmap::new();
        let mut output = ArrayImpl::Boolean(BooleanArray::try_new(LogicalType::Boolean).unwrap());

        let input = DataBlock::try_new(
            vec![
                ArrayImpl::Boolean(BooleanArray::from_iter([Some(false), None, Some(true)])),
                ArrayImpl::String(StringArray::from_values_iter(["hi", "conjunction", "haha"])),
            ],
            3,
        )
        .unwrap();

        or_conjunction
            .execute(&input, &mut selection, &mut exec_ctx, &mut output)
            .unwrap();

        assert_eq!(selection.iter().collect::<Vec<_>>(), [true, false, true]);

        // Early return
        selection.mutate().clear();

        let input = DataBlock::try_new(
            vec![
                ArrayImpl::Boolean(BooleanArray::from_iter([
                    Some(true),
                    Some(true),
                    Some(true),
                ])),
                ArrayImpl::String(StringArray::from_values_iter(["hi", "conjunction", "haha"])),
            ],
            3,
        )
        .unwrap();

        or_conjunction
            .execute(&input, &mut selection, &mut exec_ctx, &mut output)
            .unwrap();
        assert!(selection.is_empty());
    }
}
