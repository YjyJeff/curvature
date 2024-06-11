//! Comparison between two expressions

use std::sync::Arc;

use data_block::array::{Array, ArrayImpl, BooleanArray};
use data_block::bitmap::Bitmap;
use data_block::block::DataBlock;
use data_block::element::ElementImplRef;
use data_block::types::LogicalType;
use snafu::{ensure, ResultExt, Snafu};

use super::executor::ExprExecCtx;
use super::utils::CompactExprDisplayWrapper;
use super::{execute_binary_children, ExecuteSnafu, ExprResult, PhysicalExpr, Stringify};
use crate::common::expr_operator::comparison::{can_compare, CmpOperator};
use crate::exec::physical_expr::constant::Constant;

#[allow(missing_docs)]
/// Error returned by comparison expression
#[derive(Debug, Snafu)]
pub enum ComparisonError {
    #[snafu(display("Perform `{}` comparison between `{:?}` and `{:?}` is not supported", op.symbol_ident(), left,right))]
    UnsupportedInputs {
        left: LogicalType,
        right: LogicalType,
        op: CmpOperator,
    },
    #[snafu(display(
        "Perform `{}` between two constants(left: `{}`, right: `{}`) makes no sense. Planner/Optimizer should handle it in advance", 
        op,
        left,
        right
    ))]
    TwoConstants {
        left: String,
        right: String,
        op: CmpOperator,
    },
    #[snafu(display(
        "Perform `{}` between {:?} and {:?} is supported but not implemented",
        op,
        left,
        right
    ))]
    NotImplemented {
        left: LogicalType,
        right: LogicalType,
        op: CmpOperator,
    },
}

/// Comparison between two expressions
#[derive(Debug)]
pub struct Comparison {
    output_type: LogicalType,
    children: [Arc<dyn PhysicalExpr>; 2],
    op: CmpOperator,
}

type Result<T> = std::result::Result<T, ComparisonError>;

impl Comparison {
    /// Create a new [`Comparison`] expression
    ///
    /// FIXME: We knew the variants and functions that should be called in this phase.
    /// Thus, we can avoid the redundant code and the inconsistent implementation between
    /// them
    pub fn try_new(
        left: Arc<dyn PhysicalExpr>,
        op: CmpOperator,
        right: Arc<dyn PhysicalExpr>,
    ) -> Result<Self> {
        ensure!(
            can_compare(left.output_type(), right.output_type(), op),
            UnsupportedInputsSnafu {
                left: left.output_type().to_owned(),
                right: right.output_type().to_owned(),
                op,
            }
        );
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
            output_type: LogicalType::Boolean,
            children: [left, right],
            op,
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

impl Stringify for Comparison {
    fn name(&self) -> &'static str {
        self.op.ident()
    }

    fn debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }

    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.op.ident())
    }

    fn compact_display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.left_child().compact_display(f)?;
        write!(f, " {} ", self.op.symbol_ident())?;
        self.right_child().compact_display(f)
    }
}

impl PhysicalExpr for Comparison {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn output_type(&self) -> &LogicalType {
        &self.output_type
    }

    fn children(&self) -> &[Arc<dyn PhysicalExpr>] {
        &self.children
    }

    // FIXME: The equality comparison between `DayTime` can be optimized by by comparing `i64`
    fn execute(
        &self,
        leaf_input: &DataBlock,
        selection: &mut Bitmap,
        exec_ctx: &mut ExprExecCtx,
        output: &mut ArrayImpl,
    ) -> ExprResult<()> {
        // Execute children
        execute_binary_children(self, leaf_input, selection, exec_ctx)?;

        // Execute comparison
        let left_array = unsafe { exec_ctx.intermediate_block.get_array_unchecked(0) };
        let right_array = unsafe { exec_ctx.intermediate_block.get_array_unchecked(1) };
        // Used to store the temporal result of the comparison for some expressions
        let temp: &mut BooleanArray = match output.try_into() {
            Ok(output) => output,
            Err(_) => {
                panic!(
                    "`{}` expect the input have `StringArray`, however the input array `{}Array` has logical type: `{:?}` with `{}`",
                    CompactExprDisplayWrapper::new(self),
                    output.ident(),
                    output.logical_type(),
                    output.logical_type().physical_type(),
                );
            }
        };

        match (left_array.len(), right_array.len()) {
            (1, 1) => compare_scalars(
                selection,
                left_array,
                self.op,
                right_array,
                leaf_input.len(),
            ),
            (1, _) => {
                // Reverse the partial order operator
                let op = match self.op {
                    op @ (CmpOperator::Equal | CmpOperator::NotEqual) => op,
                    CmpOperator::GreaterThan => CmpOperator::LessThanOrEqualTo,
                    CmpOperator::GreaterThanOrEqualTo => CmpOperator::LessThan,
                    CmpOperator::LessThan => CmpOperator::GreaterThanOrEqualTo,
                    CmpOperator::LessThanOrEqualTo => CmpOperator::GreaterThan,
                };
                compare_array_with_scalar(
                    selection,
                    right_array,
                    op,
                    left_array,
                    temp,
                    leaf_input.len(),
                )
            }
            (_, 1) => compare_array_with_scalar(
                selection,
                left_array,
                self.op,
                right_array,
                temp,
                leaf_input.len(),
            ),

            (_, _) => compare_arrays(selection, left_array, self.op, right_array, temp),
        }
        .boxed()
        .with_context(|_| ExecuteSnafu {
            expr: CompactExprDisplayWrapper::new(self as _).to_string(),
        })?;

        // Boolean expression, check the selection
        debug_assert!(selection.is_empty() || selection.len() == leaf_input.len());

        Ok(())
    }
}

fn compare_scalars(
    selection: &mut Bitmap,
    left: &ArrayImpl,
    op: CmpOperator,
    right: &ArrayImpl,
    len: usize,
) -> Result<()> {
    let left_valid = left.validity().all_valid();
    let right_valid = right.validity().all_valid();

    if !left_valid || !right_valid {
        // One of them is none. We do not need to perform comparison
        selection.mutate().set_all_invalid(len);
        return Ok(());
    }

    // Both of them are valid
    let left_scalar = unsafe { left.get_unchecked(0).unwrap() };
    let right_scalar = unsafe { right.get_unchecked(0).unwrap() };

    macro_rules! partial_ord {
        ($op:tt) => {
            match (left_scalar, right_scalar) {
                (ElementImplRef::Int8(left), ElementImplRef::Int8(right)) => left $op right,
                (ElementImplRef::UInt8(left), ElementImplRef::UInt8(right)) => left $op right,
                (ElementImplRef::Int16(left), ElementImplRef::Int16(right)) => left $op right,
                (ElementImplRef::UInt16(left), ElementImplRef::UInt16(right)) => left $op right,
                (ElementImplRef::Int32(left), ElementImplRef::Int32(right)) => left $op right,
                (ElementImplRef::UInt32(left), ElementImplRef::UInt32(right)) => left $op right,
                (ElementImplRef::Int64(left), ElementImplRef::Int64(right)) => left $op right,
                (ElementImplRef::UInt64(left), ElementImplRef::UInt64(right)) => left $op right,
                (ElementImplRef::Float32(left), ElementImplRef::Float32(right)) => left $op right,
                (ElementImplRef::Float64(left), ElementImplRef::Float64(right)) => left $op right,
                (ElementImplRef::Int128(left), ElementImplRef::Int128(right)) => left $op right,
                (ElementImplRef::DayTime(left), ElementImplRef::DayTime(right)) => left $op right,
                (ElementImplRef::String(left), ElementImplRef::String(right)) => left $op right,
                (ElementImplRef::Binary(left), ElementImplRef::Binary(right)) => left $op right,
                (ElementImplRef::Boolean(left), ElementImplRef::Boolean(right)) => left $op right,
                _ => {
                    return NotImplementedSnafu {
                        left: left.logical_type().to_owned(),
                        right: right.logical_type().to_owned(),
                        op,
                    }
                    .fail()
                },

            }
        };
    }

    match op {
        CmpOperator::Equal => {
            // Currently, we just compare the element
            if left_scalar != right_scalar {
                selection.mutate().set_all_invalid(len);
            }
        }
        CmpOperator::NotEqual => {
            // Currently, we just compare the element
            if left_scalar == right_scalar {
                selection.mutate().set_all_invalid(len);
            }
        }
        CmpOperator::GreaterThan => {
            let cmp_result = partial_ord!(>);
            if !cmp_result {
                selection.mutate().set_all_invalid(len);
            }
        }
        CmpOperator::GreaterThanOrEqualTo => {
            let cmp_result = partial_ord!(>=);
            if !cmp_result {
                selection.mutate().set_all_invalid(len);
            }
        }

        CmpOperator::LessThan => {
            let cmp_result = partial_ord!(<);
            if !cmp_result {
                selection.mutate().set_all_invalid(len);
            }
        }
        CmpOperator::LessThanOrEqualTo => {
            let cmp_result = partial_ord!(<=);
            if !cmp_result {
                selection.mutate().set_all_invalid(len);
            }
        }
    }

    Ok(())
}

fn compare_array_with_scalar(
    selection: &mut Bitmap,
    array: &ArrayImpl,
    op: CmpOperator,
    constant_array: &ArrayImpl,
    temp: &mut BooleanArray,
    len: usize,
) -> Result<()> {
    let constant_valid = constant_array.validity().all_valid();
    if !constant_valid {
        selection.mutate().set_all_invalid(len);
        return Ok(());
    }

    macro_rules! cmp {
        ($intrinsic_cmp_func:path, $primitive_cmp_func:path, $string_cmp_func:path, $boolean_cmp_func:path, $default_cmp_func:path) => {
            match (array, constant_array) {
                (ArrayImpl::Int8(array), ArrayImpl::Int8(constant_array)) => {
                    let scalar = constant_array.get_value_unchecked(0);
                    $intrinsic_cmp_func(selection, array, scalar, temp);
                }
                (ArrayImpl::UInt8(array), ArrayImpl::UInt8(constant_array)) => {
                    let scalar = constant_array.get_value_unchecked(0);
                    $intrinsic_cmp_func(selection, array, scalar, temp);
                }
                (ArrayImpl::Int16(array), ArrayImpl::Int16(constant_array)) => {
                    let scalar = constant_array.get_value_unchecked(0);
                    $intrinsic_cmp_func(selection, array, scalar, temp);
                }
                (ArrayImpl::UInt16(array), ArrayImpl::UInt16(constant_array)) => {
                    let scalar = constant_array.get_value_unchecked(0);
                    $intrinsic_cmp_func(selection, array, scalar, temp);
                }
                (ArrayImpl::Int32(array), ArrayImpl::Int32(constant_array)) => {
                    let scalar = constant_array.get_value_unchecked(0);
                    $intrinsic_cmp_func(selection, array, scalar, temp);
                }
                (ArrayImpl::UInt32(array), ArrayImpl::UInt32(constant_array)) => {
                    let scalar = constant_array.get_value_unchecked(0);
                    $intrinsic_cmp_func(selection, array, scalar, temp);
                }
                (ArrayImpl::Int64(array), ArrayImpl::Int64(constant_array)) => {
                    let scalar = constant_array.get_value_unchecked(0);
                    $intrinsic_cmp_func(selection, array, scalar, temp);
                }
                (ArrayImpl::UInt64(array), ArrayImpl::UInt64(constant_array)) => {
                    let scalar = constant_array.get_value_unchecked(0);
                    $intrinsic_cmp_func(selection, array, scalar, temp);
                }
                (ArrayImpl::Float32(array), ArrayImpl::Float32(constant_array)) => {
                    let scalar = constant_array.get_value_unchecked(0);
                    $intrinsic_cmp_func(selection, array, scalar, temp);
                }
                (ArrayImpl::Float64(array), ArrayImpl::Float64(constant_array)) => {
                    let scalar = constant_array.get_value_unchecked(0);
                    $intrinsic_cmp_func(selection, array, scalar, temp);
                }
                (ArrayImpl::Int128(array), ArrayImpl::Int128(constant_array)) => {
                    let scalar = constant_array.get_value_unchecked(0);
                    $default_cmp_func(selection, array, scalar);
                }
                (ArrayImpl::DayTime(array), ArrayImpl::DayTime(constant_array)) => {
                    let scalar = constant_array.get_value_unchecked(0);
                    $default_cmp_func(selection, array, scalar);
                }
                (ArrayImpl::String(array), ArrayImpl::String(constant_array)) => {
                    let scalar = constant_array.get_value_unchecked(0);
                    $string_cmp_func(selection, array, scalar);
                }
                (ArrayImpl::Binary(array), ArrayImpl::Binary(constant_array)) => {
                    let scalar = constant_array.get_value_unchecked(0);
                    $default_cmp_func(selection, array, scalar);
                }
                (ArrayImpl::Boolean(array), ArrayImpl::Boolean(constant_array)) => {
                    let scalar = constant_array.get_value_unchecked(0);
                    $boolean_cmp_func(selection, array, scalar);
                }
                _ => {
                    return NotImplementedSnafu {
                        left: array.logical_type().to_owned(),
                        right: constant_array.logical_type().to_owned(),
                        op,
                    }
                    .fail()
                }
            }
        };
    }

    unsafe {
        match op {
            CmpOperator::Equal => cmp!(
                data_block::compute::comparison::primitive::intrinsic::eq_scalar,
                data_block::compute::comparison::primitive::eq_scalar,
                data_block::compute::comparison::string::eq_scalar,
                data_block::compute::comparison::boolean::eq_scalar,
                data_block::compute::comparison::eq_scalar
            ),
            CmpOperator::NotEqual => cmp!(
                data_block::compute::comparison::primitive::intrinsic::ne_scalar,
                data_block::compute::comparison::primitive::ne_scalar,
                data_block::compute::comparison::string::ne_scalar,
                data_block::compute::comparison::boolean::ne_scalar,
                data_block::compute::comparison::ne_scalar
            ),
            CmpOperator::GreaterThan => cmp!(
                data_block::compute::comparison::primitive::intrinsic::gt_scalar,
                data_block::compute::comparison::primitive::gt_scalar,
                data_block::compute::comparison::string::gt_scalar,
                data_block::compute::comparison::boolean::gt_scalar,
                data_block::compute::comparison::gt_scalar
            ),
            CmpOperator::GreaterThanOrEqualTo => cmp!(
                data_block::compute::comparison::primitive::intrinsic::ge_scalar,
                data_block::compute::comparison::primitive::ge_scalar,
                data_block::compute::comparison::string::ge_scalar,
                data_block::compute::comparison::boolean::ge_scalar,
                data_block::compute::comparison::ge_scalar
            ),
            CmpOperator::LessThan => cmp!(
                data_block::compute::comparison::primitive::intrinsic::lt_scalar,
                data_block::compute::comparison::primitive::lt_scalar,
                data_block::compute::comparison::string::lt_scalar,
                data_block::compute::comparison::boolean::lt_scalar,
                data_block::compute::comparison::lt_scalar
            ),
            CmpOperator::LessThanOrEqualTo => cmp!(
                data_block::compute::comparison::primitive::intrinsic::le_scalar,
                data_block::compute::comparison::primitive::le_scalar,
                data_block::compute::comparison::string::le_scalar,
                data_block::compute::comparison::boolean::le_scalar,
                data_block::compute::comparison::le_scalar
            ),
        }
    }

    Ok(())
}

fn compare_arrays(
    _selection: &mut Bitmap,
    _left: &ArrayImpl,
    _op: CmpOperator,
    _right: &ArrayImpl,
    _temp: &mut BooleanArray,
) -> Result<()> {
    todo!()
}
