//! Arithmetic operator

use std::fmt::Display;

use data_block::array::{Array, ArrayImpl, PrimitiveArray};
use data_block::bitmap::Bitmap;
use data_block::compute::arith::intrinsic::{
    add, add_scalar, div, div_scalar, mul, mul_scalar, rem, rem_scalar, sub, sub_scalar,
};
use data_block::compute::arith::{
    scalar_add_scalar, scalar_div_scalar, scalar_mul_scalar, scalar_rem_scalar, scalar_sub_scalar,
};
use data_block::types::{LogicalType, TimeUnit};

/// Arithmetic operator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArithOperator {
    /// +
    Add,
    /// -
    Sub,
    /// *
    Mul,
    /// /
    Div,
    /// %
    Rem,
}

impl ArithOperator {
    /// Get the symbol
    pub fn symbol_ident(&self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Rem => "%",
        }
    }

    /// Get the ident of the op
    pub fn ident(&self) -> &'static str {
        match self {
            Self::Add => "Add",
            Self::Sub => "Sub",
            Self::Mul => "Mul",
            Self::Div => "Div",
            Self::Rem => "Rem",
        }
    }
}

impl Display for ArithOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.symbol_ident())
    }
}

/// Function set for arithmetic. It has an extra dynamic dispatch cost
#[derive(Debug)]
pub struct ArithFunctionSet {
    /// Output type of the arithmetic result
    pub output_type: LogicalType,
    /// Function that perform arithmetic between two scalars
    pub scalar_arith_scalar: fn(&ArrayImpl, &ArrayImpl, &mut ArrayImpl),
    /// Function that perform arithmetic between array and scalar
    pub array_arith_scalar: fn(&Bitmap, &ArrayImpl, &ArrayImpl, &mut ArrayImpl),
    /// Function that perform arithmetic between two arrays that have same length
    pub array_arith_array: fn(&Bitmap, &ArrayImpl, &ArrayImpl, &mut ArrayImpl),
}

macro_rules! arithmetic_scalars_func {
    ($left_pat:pat, $left_ty:ty, $right_pat:pat, $right_ty:ty, $output_ty:expr, $dst_ty:ty, $func:path) => {
        |left: &ArrayImpl, right: &ArrayImpl, dst: &mut ArrayImpl| {
            let left: &$left_ty = left.try_into().unwrap_or_else(|e| {
                panic!(
                    "Left ScalarArray with `LogicalType::{}` must be `{}`. Error: `{}`",
                    stringify!($left_pat),
                    stringify!($left_ty),
                    e
                )
            });
            let right: &$right_ty = right.try_into().unwrap_or_else(|e| {
                panic!(
                    "Right ScalarArray with `LogicalType::{}` must be `{}`. Error: `{}`",
                    stringify!($right_pat),
                    stringify!($right_ty),
                    e
                )
            });
            let dst: &mut $dst_ty = dst.try_into().unwrap_or_else(|e| {
                panic!(
                    "Output Array with `LogicalType::{}` must be `{}`. Error: `{}`",
                    stringify!($output_ty),
                    stringify!($dst_ty),
                    e
                )
            });
            // Result is a scalar array
            unsafe {
                if !left.validity().all_valid() || !right.validity().all_valid() {
                    // Either of them is NULL, the result is NULL
                    dst.set_all_invalid(1);
                } else {
                    let arith_result =
                        $func(left.get_value_unchecked(0), right.get_value_unchecked(0));
                    dst.replace_with_trusted_len_values_iterator(1, std::iter::once(arith_result));
                }
            }
        }
    };
}

macro_rules! arithmetic_array_scalar_func {
    ($array_pat:pat, $array_ty:ty, $scalar_pat:pat, $scalar_ty:ty, $output_ty:expr, $dst_ty:ty, $func:path) => {
        |selection: &Bitmap, array: &ArrayImpl, scalar: &ArrayImpl, dst: &mut ArrayImpl| {
            let array: &$array_ty = array.try_into().unwrap_or_else(|e| {
                panic!(
                    "FlattenArray with `LogicalType::{}` must be `{}`. Error: `{}`",
                    stringify!($array_pat),
                    stringify!($array_ty),
                    e
                )
            });
            let scalar: &$scalar_ty = scalar.try_into().unwrap_or_else(|e| {
                panic!(
                    "ScalarArray with `LogicalType::{}` must be `{}`. Error: `{}`",
                    stringify!($scalar_pat),
                    stringify!($scalar_ty),
                    e
                )
            });
            let dst: &mut $dst_ty = dst.try_into().unwrap_or_else(|e| {
                panic!(
                    "Output Array with `LogicalType::{}` must be `{}`. Error: `{}`",
                    stringify!($output_ty),
                    stringify!($dst_ty),
                    e
                )
            });

            unsafe {
                if !scalar.validity().all_valid() {
                    // Scalar is NULL, all of the result is NULL
                    dst.set_all_invalid(array.len());
                } else {
                    $func(selection, array, scalar.get_value_unchecked(0), dst)
                }
            }
        }
    };
}

macro_rules! arithmetic_arrays_func {
    ($left_pat:pat, $left_ty:ty, $right_pat:pat, $right_ty:ty, $output_ty:expr, $dst_ty:ty, $func:path) => {
        |selection: &Bitmap, left: &ArrayImpl, right: &ArrayImpl, dst: &mut ArrayImpl| {
            let left: &$left_ty = left.try_into().unwrap_or_else(|e| {
                panic!(
                    "Left FlattenArray with `LogicalType::{}` must be `{}`. Error: `{}`",
                    stringify!($left_pat),
                    stringify!($left_ty),
                    e
                )
            });
            let right: &$right_ty = right.try_into().unwrap_or_else(|e| {
                panic!(
                    "Right FlattenArray with `LogicalType::{}` must be `{}`. Error: `{}`",
                    stringify!($right_pat),
                    stringify!($right_ty),
                    e
                )
            });
            let dst: &mut $dst_ty = dst.try_into().unwrap_or_else(|e| {
                panic!(
                    "Output Array with `LogicalType::{}` must be `{}`. Error: `{}`",
                    stringify!($output_ty),
                    stringify!($dst_ty),
                    e
                )
            });
            unsafe { $func(selection, left, right, dst) }
        }
    };
}

// All supported match arm
// TODO: Support timestamp, timestamptz, Time, Timetz, Date
macro_rules! for_all_arith {
    ($macro:ident) => {
        $macro! {
            {TinyInt, TinyInt, Add, TinyInt, PrimitiveArray<i8>, PrimitiveArray<i8>, PrimitiveArray<i8>, scalar_add_scalar, add_scalar, add},
            {TinyInt, TinyInt, Sub, TinyInt, PrimitiveArray<i8>, PrimitiveArray<i8>, PrimitiveArray<i8>, scalar_sub_scalar, sub_scalar, sub},
            {TinyInt, TinyInt, Mul, TinyInt, PrimitiveArray<i8>, PrimitiveArray<i8>, PrimitiveArray<i8>, scalar_mul_scalar, mul_scalar, mul},
            {TinyInt, TinyInt, Div, TinyInt, PrimitiveArray<i8>, PrimitiveArray<i8>, PrimitiveArray<i8>, scalar_div_scalar, div_scalar, div},
            {TinyInt, TinyInt, Rem, TinyInt, PrimitiveArray<i8>, PrimitiveArray<i8>, PrimitiveArray<i8>, scalar_rem_scalar, rem_scalar, rem},
            {SmallInt, SmallInt, Add, SmallInt, PrimitiveArray<i16>, PrimitiveArray<i16>, PrimitiveArray<i16>, scalar_add_scalar, add_scalar, add},
            {SmallInt, SmallInt, Sub, SmallInt, PrimitiveArray<i16>, PrimitiveArray<i16>, PrimitiveArray<i16>, scalar_sub_scalar, sub_scalar, sub},
            {SmallInt, SmallInt, Mul, SmallInt, PrimitiveArray<i16>, PrimitiveArray<i16>, PrimitiveArray<i16>, scalar_mul_scalar, mul_scalar, mul},
            {SmallInt, SmallInt, Div, SmallInt, PrimitiveArray<i16>, PrimitiveArray<i16>, PrimitiveArray<i16>, scalar_div_scalar, div_scalar, div},
            {SmallInt, SmallInt, Rem, SmallInt, PrimitiveArray<i16>, PrimitiveArray<i16>, PrimitiveArray<i16>, scalar_rem_scalar, rem_scalar, rem},
            {Integer, Integer, Add, Integer, PrimitiveArray<i32>, PrimitiveArray<i32>, PrimitiveArray<i32>, scalar_add_scalar, add_scalar, add},
            {Integer, Integer, Sub, Integer, PrimitiveArray<i32>, PrimitiveArray<i32>, PrimitiveArray<i32>, scalar_sub_scalar, sub_scalar, sub},
            {Integer, Integer, Mul, Integer, PrimitiveArray<i32>, PrimitiveArray<i32>, PrimitiveArray<i32>, scalar_mul_scalar, mul_scalar, mul},
            {Integer, Integer, Div, Integer, PrimitiveArray<i32>, PrimitiveArray<i32>, PrimitiveArray<i32>, scalar_div_scalar, div_scalar, div},
            {Integer, Integer, Rem, Integer, PrimitiveArray<i32>, PrimitiveArray<i32>, PrimitiveArray<i32>, scalar_rem_scalar, rem_scalar, rem},
            {BigInt, BigInt, Add, BigInt, PrimitiveArray<i64>, PrimitiveArray<i64>, PrimitiveArray<i64>, scalar_add_scalar, add_scalar, add},
            {BigInt, BigInt, Sub, BigInt, PrimitiveArray<i64>, PrimitiveArray<i64>, PrimitiveArray<i64>, scalar_sub_scalar, sub_scalar, sub},
            {BigInt, BigInt, Mul, BigInt, PrimitiveArray<i64>, PrimitiveArray<i64>, PrimitiveArray<i64>, scalar_mul_scalar, mul_scalar, mul},
            {BigInt, BigInt, Div, BigInt, PrimitiveArray<i64>, PrimitiveArray<i64>, PrimitiveArray<i64>, scalar_div_scalar, div_scalar, div},
            {BigInt, BigInt, Rem, BigInt, PrimitiveArray<i64>, PrimitiveArray<i64>, PrimitiveArray<i64>, scalar_rem_scalar, rem_scalar, rem},
            {UnsignedTinyInt, UnsignedTinyInt, Add, UnsignedTinyInt, PrimitiveArray<u8>, PrimitiveArray<u8>, PrimitiveArray<u8>, scalar_add_scalar, add_scalar, add},
            {UnsignedTinyInt, UnsignedTinyInt, Sub, UnsignedTinyInt, PrimitiveArray<u8>, PrimitiveArray<u8>, PrimitiveArray<u8>, scalar_sub_scalar, sub_scalar, sub},
            {UnsignedTinyInt, UnsignedTinyInt, Mul, UnsignedTinyInt, PrimitiveArray<u8>, PrimitiveArray<u8>, PrimitiveArray<u8>, scalar_mul_scalar, mul_scalar, mul},
            {UnsignedTinyInt, UnsignedTinyInt, Div, UnsignedTinyInt, PrimitiveArray<u8>, PrimitiveArray<u8>, PrimitiveArray<u8>, scalar_div_scalar, div_scalar, div},
            {UnsignedTinyInt, UnsignedTinyInt, Rem, UnsignedTinyInt, PrimitiveArray<u8>, PrimitiveArray<u8>, PrimitiveArray<u8>, scalar_rem_scalar, rem_scalar, rem},
            {UnsignedSmallInt, UnsignedSmallInt, Add, UnsignedSmallInt, PrimitiveArray<u16>, PrimitiveArray<u16>, PrimitiveArray<u16>, scalar_add_scalar, add_scalar, add},
            {UnsignedSmallInt, UnsignedSmallInt, Sub, UnsignedSmallInt, PrimitiveArray<u16>, PrimitiveArray<u16>, PrimitiveArray<u16>, scalar_sub_scalar, sub_scalar, sub},
            {UnsignedSmallInt, UnsignedSmallInt, Mul, UnsignedSmallInt, PrimitiveArray<u16>, PrimitiveArray<u16>, PrimitiveArray<u16>, scalar_mul_scalar, mul_scalar, mul},
            {UnsignedSmallInt, UnsignedSmallInt, Div, UnsignedSmallInt, PrimitiveArray<u16>, PrimitiveArray<u16>, PrimitiveArray<u16>, scalar_div_scalar, div_scalar, div},
            {UnsignedSmallInt, UnsignedSmallInt, Rem, UnsignedSmallInt, PrimitiveArray<u16>, PrimitiveArray<u16>, PrimitiveArray<u16>, scalar_rem_scalar, rem_scalar, rem},
            {UnsignedInteger, UnsignedInteger, Add, UnsignedInteger, PrimitiveArray<u32>, PrimitiveArray<u32>, PrimitiveArray<u32>, scalar_add_scalar, add_scalar, add},
            {UnsignedInteger, UnsignedInteger, Sub, UnsignedInteger, PrimitiveArray<u32>, PrimitiveArray<u32>, PrimitiveArray<u32>, scalar_sub_scalar, sub_scalar, sub},
            {UnsignedInteger, UnsignedInteger, Mul, UnsignedInteger, PrimitiveArray<u32>, PrimitiveArray<u32>, PrimitiveArray<u32>, scalar_mul_scalar, mul_scalar, mul},
            {UnsignedInteger, UnsignedInteger, Div, UnsignedInteger, PrimitiveArray<u32>, PrimitiveArray<u32>, PrimitiveArray<u32>, scalar_div_scalar, div_scalar, div},
            {UnsignedInteger, UnsignedInteger, Rem, UnsignedInteger, PrimitiveArray<u32>, PrimitiveArray<u32>, PrimitiveArray<u32>, scalar_rem_scalar, rem_scalar, rem},
            {UnsignedBigInt, UnsignedBigInt, Add, UnsignedBigInt, PrimitiveArray<u64>, PrimitiveArray<u64>, PrimitiveArray<u64>, scalar_add_scalar, add_scalar, add},
            {UnsignedBigInt, UnsignedBigInt, Sub, UnsignedBigInt, PrimitiveArray<u64>, PrimitiveArray<u64>, PrimitiveArray<u64>, scalar_sub_scalar, sub_scalar, sub},
            {UnsignedBigInt, UnsignedBigInt, Mul, UnsignedBigInt, PrimitiveArray<u64>, PrimitiveArray<u64>, PrimitiveArray<u64>, scalar_mul_scalar, mul_scalar, mul},
            {UnsignedBigInt, UnsignedBigInt, Div, UnsignedBigInt, PrimitiveArray<u64>, PrimitiveArray<u64>, PrimitiveArray<u64>, scalar_div_scalar, div_scalar, div},
            {UnsignedBigInt, UnsignedBigInt, Rem, UnsignedBigInt, PrimitiveArray<u64>, PrimitiveArray<u64>, PrimitiveArray<u64>, scalar_rem_scalar, rem_scalar, rem},
            {Float, Float, Add, Float, PrimitiveArray<f32>, PrimitiveArray<f32>, PrimitiveArray<f32>, scalar_add_scalar, add_scalar, add},
            {Float, Float, Sub, Float, PrimitiveArray<f32>, PrimitiveArray<f32>, PrimitiveArray<f32>, scalar_sub_scalar, sub_scalar, sub},
            {Float, Float, Mul, Float, PrimitiveArray<f32>, PrimitiveArray<f32>, PrimitiveArray<f32>, scalar_mul_scalar, mul_scalar, mul},
            {Float, Float, Div, Float, PrimitiveArray<f32>, PrimitiveArray<f32>, PrimitiveArray<f32>, scalar_div_scalar, div_scalar, div},
            {Float, Float, Rem, Float, PrimitiveArray<f32>, PrimitiveArray<f32>, PrimitiveArray<f32>, scalar_rem_scalar, rem_scalar, rem},
            {Double, Double, Add, Double, PrimitiveArray<f64>, PrimitiveArray<f64>, PrimitiveArray<f64>, scalar_add_scalar, add_scalar, add},
            {Double, Double, Sub, Double, PrimitiveArray<f64>, PrimitiveArray<f64>, PrimitiveArray<f64>, scalar_sub_scalar, sub_scalar, sub},
            {Double, Double, Mul, Double, PrimitiveArray<f64>, PrimitiveArray<f64>, PrimitiveArray<f64>, scalar_mul_scalar, mul_scalar, mul},
            {Double, Double, Div, Double, PrimitiveArray<f64>, PrimitiveArray<f64>, PrimitiveArray<f64>, scalar_div_scalar, div_scalar, div},
            {Double, Double, Rem, Double, PrimitiveArray<f64>, PrimitiveArray<f64>, PrimitiveArray<f64>, scalar_rem_scalar, rem_scalar, rem},
            // Rem between different integer types
            {SmallInt, TinyInt, Rem, TinyInt, PrimitiveArray<i16>, PrimitiveArray<i8>, PrimitiveArray<i8>, scalar_rem_scalar, rem_scalar, rem},
            {Integer, TinyInt, Rem, TinyInt, PrimitiveArray<i32>, PrimitiveArray<i8>, PrimitiveArray<i8>, scalar_rem_scalar, rem_scalar, rem},
            {BigInt, TinyInt, Rem, TinyInt, PrimitiveArray<i64>, PrimitiveArray<i8>, PrimitiveArray<i8>, scalar_rem_scalar, rem_scalar, rem},
            {Integer, SmallInt, Rem, SmallInt, PrimitiveArray<i32>, PrimitiveArray<i16>, PrimitiveArray<i16>, scalar_rem_scalar, rem_scalar, rem},
            {BigInt, SmallInt, Rem, SmallInt, PrimitiveArray<i64>, PrimitiveArray<i16>, PrimitiveArray<i16>, scalar_rem_scalar, rem_scalar, rem},
            {BigInt, Integer, Rem, Integer, PrimitiveArray<i64>, PrimitiveArray<i32>, PrimitiveArray<i32>, scalar_rem_scalar, rem_scalar, rem},
            {UnsignedSmallInt, UnsignedTinyInt, Rem, UnsignedTinyInt, PrimitiveArray<u16>, PrimitiveArray<u8>, PrimitiveArray<u8>, scalar_rem_scalar, rem_scalar, rem},
            {UnsignedInteger, UnsignedTinyInt, Rem, UnsignedTinyInt, PrimitiveArray<u32>, PrimitiveArray<u8>, PrimitiveArray<u8>, scalar_rem_scalar, rem_scalar, rem},
            {UnsignedBigInt, UnsignedTinyInt, Rem, UnsignedTinyInt, PrimitiveArray<u64>, PrimitiveArray<u8>, PrimitiveArray<u8>, scalar_rem_scalar, rem_scalar, rem},
            {UnsignedInteger, UnsignedSmallInt, Rem, UnsignedSmallInt, PrimitiveArray<u32>, PrimitiveArray<u16>, PrimitiveArray<u16>, scalar_rem_scalar, rem_scalar, rem},
            {UnsignedBigInt, UnsignedSmallInt, Rem, UnsignedSmallInt, PrimitiveArray<u64>, PrimitiveArray<u16>, PrimitiveArray<u16>, scalar_rem_scalar, rem_scalar, rem},
            {UnsignedBigInt, UnsignedInteger, Rem, UnsignedInteger, PrimitiveArray<u64>, PrimitiveArray<u32>, PrimitiveArray<u32>, scalar_rem_scalar, rem_scalar, rem},
            // Timestamp
            {Timestamp(TimeUnit::Second), Timestamp(TimeUnit::Second), Sub, Duration(TimeUnit::Second), PrimitiveArray<i64>, PrimitiveArray<i64>, PrimitiveArray<i64>, scalar_sub_scalar, sub_scalar, sub},
            {Timestamp(TimeUnit::Microsecond), Timestamp(TimeUnit::Millisecond), Sub, Duration(TimeUnit::Millisecond), PrimitiveArray<i64>, PrimitiveArray<i64>, PrimitiveArray<i64>, scalar_sub_scalar, sub_scalar, sub},
            {Timestamp(TimeUnit::Microsecond), Timestamp(TimeUnit::Microsecond), Sub, Duration(TimeUnit::Microsecond), PrimitiveArray<i64>, PrimitiveArray<i64>, PrimitiveArray<i64>, scalar_sub_scalar, sub_scalar, sub},
            {Timestamp(TimeUnit::Nanosecond), Timestamp(TimeUnit::Nanosecond), Sub, Duration(TimeUnit::Nanosecond), PrimitiveArray<i64>, PrimitiveArray<i64>, PrimitiveArray<i64>, scalar_sub_scalar, sub_scalar, sub}
        }
    };
}

/// Infer the arithmetic function set
///
/// Returns `Some(ArithFunctionSet)` if the arithmetic between two logical types is supported.
/// Otherwise, returns `None`
pub fn infer_arithmetic_func_set(
    left: &LogicalType,
    right: &LogicalType,
    op: ArithOperator,
) -> Option<ArithFunctionSet> {
    use ArithOperator::*;
    use LogicalType::*;

    macro_rules! impl_arith {
        ($({$left:pat, $right:pat, $op:ident, $output_type:expr, $left_ty:ty, $right_ty:ty, $dst_ty:ty, $scalars_func:path, $array_scalar_func:path, $arrays_func:path}),+) => {
            match (left, right, op) {
                $(
                    ($left, $right, $op) => Some(ArithFunctionSet {
                        output_type: $output_type,
                        scalar_arith_scalar: arithmetic_scalars_func!($left, $left_ty, $right, $right_ty, $output_type, $dst_ty, $scalars_func),
                        array_arith_scalar: arithmetic_array_scalar_func!($left, $left_ty, $right, $right_ty, $output_type, $dst_ty, $array_scalar_func),
                        array_arith_array: arithmetic_arrays_func!($left, $left_ty, $right, $right_ty, $output_type, $dst_ty, $arrays_func),
                    }),
                )+

                _ => None,
            }
        };
    }

    for_all_arith!(impl_arith)
}
