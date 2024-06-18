//! Comparison operator

use std::fmt::Display;

use data_block::array::{Array, ArrayImpl, BinaryArray, BooleanArray, PrimitiveArray, StringArray};
use data_block::bitmap::Bitmap;
use data_block::compute::comparison::boolean::{
    eq as boolean_eq, ge as boolean_ge, gt as boolean_gt, le as boolean_le, lt as boolean_lt,
    ne as boolean_ne,
};
use data_block::compute::comparison::primitive::intrinsic::{
    eq as intrinsic_eq, eq_scalar as intrinsic_eq_scalar, ge as intrinsic_ge,
    ge_scalar as intrinsic_ge_scalar, gt as intrinsic_gt, gt_scalar as intrinsic_gt_scalar,
    le as intrinsic_le, le_scalar as intrinsic_le_scalar, lt as intrinsic_lt,
    lt_scalar as intrinsic_lt_scalar, ne as intrinsic_ne, ne_scalar as intrinsic_ne_scalar,
    timestamp_eq, timestamp_eq_scalar, timestamp_ge, timestamp_ge_scalar, timestamp_gt,
    timestamp_gt_scalar, timestamp_le, timestamp_le_scalar, timestamp_lt, timestamp_lt_scalar,
    timestamp_ne, timestamp_ne_scalar,
};
use data_block::compute::comparison::primitive::{
    eq as primitive_eq, eq_scalar as primitive_eq_scalar, ge as primitive_ge,
    ge_scalar as primitive_ge_scalar, gt as primitive_gt, gt_scalar as primitive_gt_scalar,
    le as primitive_le, le_scalar as primitive_le_scalar, lt as primitive_lt,
    lt_scalar as primitive_lt_scalar, ne as primitive_ne, ne_scalar as primitive_ne_scalar,
};
use data_block::compute::comparison::{
    scalar_eq_scalar, scalar_ge_scalar, scalar_gt_scalar, scalar_le_scalar, scalar_lt_scalar,
    scalar_ne_scalar, timestamp_scalar_eq_scalar, timestamp_scalar_ge_scalar,
    timestamp_scalar_gt_scalar, timestamp_scalar_le_scalar, timestamp_scalar_lt_scalar,
    timestamp_scalar_ne_scalar,
};
use data_block::element::string::StringView;
use data_block::element::Element;
use data_block::types::{LogicalType, TimeUnit};

/// Comparison operator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CmpOperator {
    /// ==
    Equal,
    /// !=
    NotEqual,
    /// >
    GreaterThan,
    /// >=
    GreaterThanOrEqualTo,
    /// <
    LessThan,
    /// <=
    LessThanOrEqualTo,
}

impl CmpOperator {
    /// Get the symbol
    pub fn symbol_ident(&self) -> &'static str {
        match self {
            Self::Equal => "==",
            Self::NotEqual => "!=",
            Self::GreaterThan => ">",
            Self::GreaterThanOrEqualTo => ">=",
            Self::LessThan => "<",
            Self::LessThanOrEqualTo => "<=",
        }
    }

    /// Get the ident of the op
    pub fn ident(&self) -> &'static str {
        match self {
            Self::Equal => "Equal",
            Self::NotEqual => "NotEqual",
            Self::GreaterThan => "GreaterThan",
            Self::GreaterThanOrEqualTo => "GreaterThanOrEqualTo",
            Self::LessThan => "LessThan",
            Self::LessThanOrEqualTo => "LessThanOrEqualTo",
        }
    }
}

impl Display for CmpOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.symbol_ident())
    }
}

/// Function set for comparison. It has extra dynamic dispatch cost
#[derive(Debug)]
pub struct ComparisonFunctionSet {
    /// Function that perform comparison between two scalars
    pub scalar_cmp_scalar: fn(&ArrayImpl, &ArrayImpl) -> bool,
    /// Function that perform comparison between array and scalar
    pub array_cmp_scalar: fn(&mut Bitmap, &ArrayImpl, &ArrayImpl, &mut ArrayImpl),
    /// Function that perform comparison between scalar and array.
    ///
    /// # Note
    ///
    /// Although its name is scalar_cmp_array, the first `ArrayImpl` should be
    /// FlattenArray. It use `array_cmp_scalar` internally
    pub scalar_cmp_array: fn(&mut Bitmap, array: &ArrayImpl, scalar: &ArrayImpl, &mut ArrayImpl),
    /// Function that perform comparison between two arrays
    pub array_cmp_array: fn(&mut Bitmap, array: &ArrayImpl, scalar: &ArrayImpl, &mut ArrayImpl),
}

macro_rules! cmp_scalars_func {
    ($lhs_pat:pat, $lhs_ty:ty, $rhs_pat:pat, $rhs_ty:ty, $func:path) => {
        |lhs: &ArrayImpl, rhs: &ArrayImpl| {
            let lhs: &$lhs_ty = lhs.try_into().unwrap_or_else(|e| {
                panic!(
                    "ScalarArray with `LogicalType::{}` must be `{}`. Error: `{}`",
                    stringify!($lhs_pat),
                    stringify!($lhs_ty),
                    e
                )
            });
            let rhs: &$rhs_ty = rhs.try_into().unwrap_or_else(|e| {
                panic!(
                    "ScalarArray with `LogicalType::{}` must be `{}`. Error: `{}`",
                    stringify!($rhs_pat),
                    stringify!($rhs_ty),
                    e
                )
            });

            unsafe { $func(lhs.get_value_unchecked(0), rhs.get_value_unchecked(0)) }
        }
    };
}

macro_rules! cmp_array_scalar_func {
    ($array_pat:pat, $array_ty:ty, $scalar_pat:pat, $scalar_ty:ty, $func:path) => {
        |selection: &mut Bitmap, array: &ArrayImpl, scalar: &ArrayImpl, temp: &mut ArrayImpl| {
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

            unsafe {
                if !scalar.validity().all_valid() {
                    // Scalar is NULL, all of the result is NULL
                    selection.mutate().set_all_invalid(array.len());
                } else {

                    let temp: &mut BooleanArray = match temp.try_into(){
                        Ok(dst) => dst,
                        Err(_) => {
                            panic!(
                                "Output of the comparison expression must be boolean array, however `{}Array` is found",
                                temp.ident()
                            )
                        }
                    };

                    $func(selection, array, scalar.get_value_unchecked(0), temp)
                }
            }
        }
    };
}

macro_rules! cmp_arrays_func {
    ($lhs_pat:pat, $lhs_ty:ty, $rhs_pat:pat, $rhs_ty:ty, $func:path) => {
        |selection: &mut Bitmap, lhs: &ArrayImpl, rhs: &ArrayImpl, temp: &mut ArrayImpl| {
            let lhs: &$lhs_ty = lhs.try_into().unwrap_or_else(|e| {
                panic!(
                    "FlattenArray with `LogicalType::{}` must be `{}`. Error: `{}`",
                    stringify!($lhs_pat),
                    stringify!($lsh_ty),
                    e
                )
            });
            let rhs: &$rhs_ty = rhs.try_into().unwrap_or_else(|e| {
                panic!(
                    "FlattenArray with `LogicalType::{}` must be `{}`. Error: `{}`",
                    stringify!($rhs_pat),
                    stringify!($rhs_ty),
                    e
                )
            });

            let temp: &mut BooleanArray = match temp.try_into(){
                Ok(dst) => dst,
                Err(_) => {
                    panic!(
                        "Output of the comparison expression must be boolean array, however `{}Array` is found",
                        temp.ident()
                    )
                }
            };

            unsafe{ $func(selection, lhs, rhs, temp); }
        }
    };
}

macro_rules! cmp_wrapper {
    ($cmp:ident) => {
        paste::paste! {
            /// Wrapper of the cmp_scalar for string, such that it matches the function signature
            /// in function set
            #[inline]
            unsafe fn [<string_ $cmp _scalar>](
                selection: &mut Bitmap,
                array: &StringArray,
                scalar: StringView<'_>,
                _temp: &BooleanArray,
            ) {
                data_block::compute::comparison::string::[<$cmp _scalar>](selection, array, scalar);
            }

            /// Wrapper of the cmp for string, such that it matches the function signature
            /// in function set
            #[inline]
            unsafe fn [<string_ $cmp>](
                selection: &mut Bitmap,
                lhs: &StringArray,
                rhs: &StringArray,
                _temp: &BooleanArray,
            ) {
                data_block::compute::comparison::string::$cmp(selection, lhs, rhs);
            }

            /// Wrapper of the cmp_scalar for boolean, such that it matches the function signature
            /// in function set
            #[inline]
            unsafe fn [<boolean_ $cmp _scalar>](
                selection: &mut Bitmap,
                array: &BooleanArray,
                scalar: bool,
                _temp: &BooleanArray,
            ) {
                data_block::compute::comparison::boolean::[<$cmp _scalar>](selection, array, scalar);
            }

            /// Wrapper of the cmp_scalar for generic array, such that it matches the function signature
            /// in function set
            #[inline]
            unsafe fn [<$cmp _scalar_default>]<A: Array>(
                selection: &mut Bitmap,
                array: &A,
                scalar: <A::Element as Element>::ElementRef<'_>,
                _temp: &BooleanArray,
            ) where
                for<'a, 'b> <A::Element as Element>::ElementRef<'a>:
                    PartialOrd<<A::Element as Element>::ElementRef<'b>>,
            {
                data_block::compute::comparison::[<$cmp _scalar>](selection, array, scalar);
            }

            /// Wrapper of the cmp for generic array, such that it matches the function signature
            /// in function set
            #[inline]
            unsafe fn [<$cmp _default>]<A: Array>(
                selection: &mut Bitmap,
                lhs: &A,
                rhs: &A,
                _temp: &BooleanArray,
            ) where
                for<'a, 'b> <A::Element as Element>::ElementRef<'a>:
                    PartialOrd<<A::Element as Element>::ElementRef<'b>>,
            {
                data_block::compute::comparison::$cmp(selection, lhs, rhs);
            }
        }
    };
}

cmp_wrapper!(eq);
cmp_wrapper!(ne);
cmp_wrapper!(gt);
cmp_wrapper!(ge);
cmp_wrapper!(lt);
cmp_wrapper!(le);

// TODO:
// - Reduce pattern
// - Support more types:
//      - Interval
//      - Duration
//      - Decimal
//      - List
macro_rules! for_all_comparison {
    ($macro:ident) => {
        $macro! {
            {TinyInt, TinyInt, Equal, PrimitiveArray<i8>, PrimitiveArray<i8>, intrinsic_eq_scalar, intrinsic_eq_scalar, scalar_eq_scalar, intrinsic_eq},
            {TinyInt, TinyInt, NotEqual, PrimitiveArray<i8>, PrimitiveArray<i8>, intrinsic_ne_scalar, intrinsic_ne_scalar, scalar_ne_scalar, intrinsic_ne},
            {TinyInt, TinyInt, GreaterThan, PrimitiveArray<i8>, PrimitiveArray<i8>, intrinsic_gt_scalar, intrinsic_lt_scalar, scalar_gt_scalar, intrinsic_gt},
            {TinyInt, TinyInt, GreaterThanOrEqualTo, PrimitiveArray<i8>, PrimitiveArray<i8>, intrinsic_ge_scalar, intrinsic_le_scalar, scalar_ge_scalar, intrinsic_ge},
            {TinyInt, TinyInt, LessThan, PrimitiveArray<i8>, PrimitiveArray<i8>, intrinsic_lt_scalar, intrinsic_gt_scalar, scalar_lt_scalar, intrinsic_lt},
            {TinyInt, TinyInt, LessThanOrEqualTo, PrimitiveArray<i8>, PrimitiveArray<i8>, intrinsic_le_scalar, intrinsic_ge_scalar, scalar_le_scalar, intrinsic_le},
            {SmallInt, SmallInt, Equal, PrimitiveArray<i16>, PrimitiveArray<i16>, intrinsic_eq_scalar, intrinsic_eq_scalar, scalar_eq_scalar, intrinsic_eq},
            {SmallInt, SmallInt, NotEqual, PrimitiveArray<i16>, PrimitiveArray<i16>, intrinsic_ne_scalar, intrinsic_ne_scalar, scalar_ne_scalar, intrinsic_ne},
            {SmallInt, SmallInt, GreaterThan, PrimitiveArray<i16>, PrimitiveArray<i16>, intrinsic_gt_scalar, intrinsic_lt_scalar, scalar_gt_scalar, intrinsic_gt},
            {SmallInt, SmallInt, GreaterThanOrEqualTo, PrimitiveArray<i16>, PrimitiveArray<i16>, intrinsic_ge_scalar, intrinsic_le_scalar, scalar_ge_scalar, intrinsic_ge},
            {SmallInt, SmallInt, LessThan, PrimitiveArray<i16>, PrimitiveArray<i16>, intrinsic_lt_scalar, intrinsic_gt_scalar, scalar_lt_scalar, intrinsic_lt},
            {SmallInt, SmallInt, LessThanOrEqualTo, PrimitiveArray<i16>, PrimitiveArray<i16>, intrinsic_le_scalar, intrinsic_ge_scalar, scalar_le_scalar, intrinsic_le},
            {Integer, Integer, Equal, PrimitiveArray<i32>, PrimitiveArray<i32>, intrinsic_eq_scalar, intrinsic_eq_scalar, scalar_eq_scalar, intrinsic_eq},
            {Integer, Integer, NotEqual, PrimitiveArray<i32>, PrimitiveArray<i32>, intrinsic_ne_scalar, intrinsic_ne_scalar, scalar_ne_scalar, intrinsic_ne},
            {Integer, Integer, GreaterThan, PrimitiveArray<i32>, PrimitiveArray<i32>, intrinsic_gt_scalar, intrinsic_lt_scalar, scalar_gt_scalar, intrinsic_gt},
            {Integer, Integer, GreaterThanOrEqualTo, PrimitiveArray<i32>, PrimitiveArray<i32>, intrinsic_ge_scalar, intrinsic_le_scalar, scalar_ge_scalar, intrinsic_ge},
            {Integer, Integer, LessThan, PrimitiveArray<i32>, PrimitiveArray<i32>, intrinsic_lt_scalar, intrinsic_gt_scalar, scalar_lt_scalar, intrinsic_lt},
            {Integer, Integer, LessThanOrEqualTo, PrimitiveArray<i32>, PrimitiveArray<i32>, intrinsic_le_scalar, intrinsic_ge_scalar, scalar_le_scalar, intrinsic_le},
            {BigInt, BigInt, Equal, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_eq_scalar, intrinsic_eq_scalar, scalar_eq_scalar, intrinsic_eq},
            {BigInt, BigInt, NotEqual, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_ne_scalar, intrinsic_ne_scalar, scalar_ne_scalar, intrinsic_ne},
            {BigInt, BigInt, GreaterThan, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_gt_scalar, intrinsic_lt_scalar, scalar_gt_scalar, intrinsic_gt},
            {BigInt, BigInt, GreaterThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_ge_scalar, intrinsic_le_scalar, scalar_ge_scalar, intrinsic_ge},
            {BigInt, BigInt, LessThan, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_lt_scalar, intrinsic_gt_scalar, scalar_lt_scalar, intrinsic_lt},
            {BigInt, BigInt, LessThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_le_scalar, intrinsic_ge_scalar, scalar_le_scalar, intrinsic_le},
            {UnsignedTinyInt, UnsignedTinyInt, Equal, PrimitiveArray<u8>, PrimitiveArray<u8>, intrinsic_eq_scalar, intrinsic_eq_scalar, scalar_eq_scalar, intrinsic_eq},
            {UnsignedTinyInt, UnsignedTinyInt, NotEqual, PrimitiveArray<u8>, PrimitiveArray<u8>, intrinsic_ne_scalar, intrinsic_ne_scalar, scalar_ne_scalar, intrinsic_ne},
            {UnsignedTinyInt, UnsignedTinyInt, GreaterThan, PrimitiveArray<u8>, PrimitiveArray<u8>, intrinsic_gt_scalar, intrinsic_lt_scalar, scalar_gt_scalar, intrinsic_gt},
            {UnsignedTinyInt, UnsignedTinyInt, GreaterThanOrEqualTo, PrimitiveArray<u8>, PrimitiveArray<u8>, intrinsic_ge_scalar, intrinsic_le_scalar, scalar_ge_scalar, intrinsic_ge},
            {UnsignedTinyInt, UnsignedTinyInt, LessThan, PrimitiveArray<u8>, PrimitiveArray<u8>, intrinsic_lt_scalar, intrinsic_gt_scalar, scalar_lt_scalar, intrinsic_lt},
            {UnsignedTinyInt, UnsignedTinyInt, LessThanOrEqualTo, PrimitiveArray<u8>, PrimitiveArray<u8>, intrinsic_le_scalar, intrinsic_ge_scalar, scalar_le_scalar, intrinsic_le},
            {UnsignedSmallInt, UnsignedSmallInt, Equal, PrimitiveArray<u16>, PrimitiveArray<u16>, intrinsic_eq_scalar, intrinsic_eq_scalar, scalar_eq_scalar, intrinsic_eq},
            {UnsignedSmallInt, UnsignedSmallInt, NotEqual, PrimitiveArray<u16>, PrimitiveArray<u16>, intrinsic_ne_scalar, intrinsic_ne_scalar, scalar_ne_scalar, intrinsic_ne},
            {UnsignedSmallInt, UnsignedSmallInt, GreaterThan, PrimitiveArray<u16>, PrimitiveArray<u16>, intrinsic_gt_scalar, intrinsic_lt_scalar, scalar_gt_scalar, intrinsic_gt},
            {UnsignedSmallInt, UnsignedSmallInt, GreaterThanOrEqualTo, PrimitiveArray<u16>, PrimitiveArray<u16>, intrinsic_ge_scalar, intrinsic_le_scalar, scalar_ge_scalar, intrinsic_ge},
            {UnsignedSmallInt, UnsignedSmallInt, LessThan, PrimitiveArray<u16>, PrimitiveArray<u16>, intrinsic_lt_scalar, intrinsic_gt_scalar, scalar_lt_scalar, intrinsic_lt},
            {UnsignedSmallInt, UnsignedSmallInt, LessThanOrEqualTo, PrimitiveArray<u16>, PrimitiveArray<u16>, intrinsic_le_scalar, intrinsic_ge_scalar, scalar_le_scalar, intrinsic_le},
            {UnsignedInteger, UnsignedInteger, Equal, PrimitiveArray<u32>, PrimitiveArray<u32>, intrinsic_eq_scalar, intrinsic_eq_scalar, scalar_eq_scalar, intrinsic_eq},
            {UnsignedInteger, UnsignedInteger, NotEqual, PrimitiveArray<u32>, PrimitiveArray<u32>, intrinsic_ne_scalar, intrinsic_ne_scalar, scalar_ne_scalar, intrinsic_ne},
            {UnsignedInteger, UnsignedInteger, GreaterThan, PrimitiveArray<u32>, PrimitiveArray<u32>, intrinsic_gt_scalar, intrinsic_lt_scalar, scalar_gt_scalar, intrinsic_gt},
            {UnsignedInteger, UnsignedInteger, GreaterThanOrEqualTo, PrimitiveArray<u32>, PrimitiveArray<u32>, intrinsic_ge_scalar, intrinsic_le_scalar, scalar_ge_scalar, intrinsic_ge},
            {UnsignedInteger, UnsignedInteger, LessThan, PrimitiveArray<u32>, PrimitiveArray<u32>, intrinsic_lt_scalar, intrinsic_gt_scalar, scalar_lt_scalar, intrinsic_lt},
            {UnsignedInteger, UnsignedInteger, LessThanOrEqualTo, PrimitiveArray<u32>, PrimitiveArray<u32>, intrinsic_le_scalar, intrinsic_ge_scalar, scalar_le_scalar, intrinsic_le},
            {UnsignedBigInt, UnsignedBigInt, Equal, PrimitiveArray<u64>, PrimitiveArray<u64>, intrinsic_eq_scalar, intrinsic_eq_scalar, scalar_eq_scalar, intrinsic_eq},
            {UnsignedBigInt, UnsignedBigInt, NotEqual, PrimitiveArray<u64>, PrimitiveArray<u64>, intrinsic_ne_scalar, intrinsic_ne_scalar, scalar_ne_scalar, intrinsic_ne},
            {UnsignedBigInt, UnsignedBigInt, GreaterThan, PrimitiveArray<u64>, PrimitiveArray<u64>, intrinsic_gt_scalar, intrinsic_lt_scalar, scalar_gt_scalar, intrinsic_gt},
            {UnsignedBigInt, UnsignedBigInt, GreaterThanOrEqualTo, PrimitiveArray<u64>, PrimitiveArray<u64>, intrinsic_ge_scalar, intrinsic_le_scalar, scalar_ge_scalar, intrinsic_ge},
            {UnsignedBigInt, UnsignedBigInt, LessThan, PrimitiveArray<u64>, PrimitiveArray<u64>, intrinsic_lt_scalar, intrinsic_gt_scalar, scalar_lt_scalar, intrinsic_lt},
            {UnsignedBigInt, UnsignedBigInt, LessThanOrEqualTo, PrimitiveArray<u64>, PrimitiveArray<u64>, intrinsic_le_scalar, intrinsic_ge_scalar, scalar_le_scalar, intrinsic_le},
            {Float, Float, Equal, PrimitiveArray<f32>, PrimitiveArray<f32>, intrinsic_eq_scalar, intrinsic_eq_scalar, scalar_eq_scalar, intrinsic_eq},
            {Float, Float, NotEqual, PrimitiveArray<f32>, PrimitiveArray<f32>, intrinsic_ne_scalar, intrinsic_ne_scalar, scalar_ne_scalar, intrinsic_ne},
            {Float, Float, GreaterThan, PrimitiveArray<f32>, PrimitiveArray<f32>, intrinsic_gt_scalar, intrinsic_lt_scalar, scalar_gt_scalar, intrinsic_gt},
            {Float, Float, GreaterThanOrEqualTo, PrimitiveArray<f32>, PrimitiveArray<f32>, intrinsic_ge_scalar, intrinsic_le_scalar, scalar_ge_scalar, intrinsic_ge},
            {Float, Float, LessThan, PrimitiveArray<f32>, PrimitiveArray<f32>, intrinsic_lt_scalar, intrinsic_gt_scalar, scalar_lt_scalar, intrinsic_lt},
            {Float, Float, LessThanOrEqualTo, PrimitiveArray<f32>, PrimitiveArray<f32>, intrinsic_le_scalar, intrinsic_ge_scalar, scalar_le_scalar, intrinsic_le},
            {Double, Double, Equal, PrimitiveArray<f64>, PrimitiveArray<f64>, intrinsic_eq_scalar, intrinsic_eq_scalar, scalar_eq_scalar, intrinsic_eq},
            {Double, Double, NotEqual, PrimitiveArray<f64>, PrimitiveArray<f64>, intrinsic_ne_scalar, intrinsic_ne_scalar, scalar_ne_scalar, intrinsic_ne},
            {Double, Double, GreaterThan, PrimitiveArray<f64>, PrimitiveArray<f64>, intrinsic_gt_scalar, intrinsic_lt_scalar, scalar_gt_scalar, intrinsic_gt},
            {Double, Double, GreaterThanOrEqualTo, PrimitiveArray<f64>, PrimitiveArray<f64>, intrinsic_ge_scalar, intrinsic_le_scalar, scalar_ge_scalar, intrinsic_ge},
            {Double, Double, LessThan, PrimitiveArray<f64>, PrimitiveArray<f64>, intrinsic_lt_scalar, intrinsic_gt_scalar, scalar_lt_scalar, intrinsic_lt},
            {Double, Double, LessThanOrEqualTo, PrimitiveArray<f64>, PrimitiveArray<f64>, intrinsic_le_scalar, intrinsic_ge_scalar, scalar_le_scalar, intrinsic_le},
            {HugeInt, HugeInt, Equal, PrimitiveArray<i128>, PrimitiveArray<i128>, primitive_eq_scalar, primitive_eq_scalar, scalar_eq_scalar, primitive_eq},
            {HugeInt, HugeInt, NotEqual, PrimitiveArray<i128>, PrimitiveArray<i128>, primitive_ne_scalar, primitive_ne_scalar, scalar_ne_scalar, primitive_ne},
            {HugeInt, HugeInt, GreaterThan, PrimitiveArray<i128>, PrimitiveArray<i128>, primitive_gt_scalar, primitive_lt_scalar, scalar_gt_scalar, primitive_gt},
            {HugeInt, HugeInt, GreaterThanOrEqualTo, PrimitiveArray<i128>, PrimitiveArray<i128>, primitive_ge_scalar, primitive_le_scalar, scalar_ge_scalar, primitive_ge},
            {HugeInt, HugeInt, LessThan, PrimitiveArray<i128>, PrimitiveArray<i128>, primitive_lt_scalar, primitive_gt_scalar, scalar_lt_scalar, primitive_lt},
            {HugeInt, HugeInt, LessThanOrEqualTo, PrimitiveArray<i128>, PrimitiveArray<i128>, primitive_le_scalar, primitive_ge_scalar, scalar_le_scalar, primitive_le},

            // Timestamp with same unit
            {Timestamp(TimeUnit::Second), Timestamp(TimeUnit::Second), Equal, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_eq_scalar, intrinsic_eq_scalar, scalar_eq_scalar, intrinsic_eq},
            {Timestamp(TimeUnit::Second), Timestamp(TimeUnit::Second), NotEqual, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_ne_scalar, intrinsic_ne_scalar, scalar_ne_scalar, intrinsic_ne},
            {Timestamp(TimeUnit::Second), Timestamp(TimeUnit::Second), GreaterThan, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_gt_scalar, intrinsic_lt_scalar, scalar_gt_scalar, intrinsic_gt},
            {Timestamp(TimeUnit::Second), Timestamp(TimeUnit::Second), GreaterThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_ge_scalar, intrinsic_le_scalar, scalar_ge_scalar, intrinsic_ge},
            {Timestamp(TimeUnit::Second), Timestamp(TimeUnit::Second), LessThan, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_lt_scalar, intrinsic_gt_scalar, scalar_lt_scalar, intrinsic_lt},
            {Timestamp(TimeUnit::Second), Timestamp(TimeUnit::Second), LessThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_le_scalar, intrinsic_ge_scalar, scalar_le_scalar, intrinsic_le},
            {Timestamp(TimeUnit::Millisecond), Timestamp(TimeUnit::Millisecond), Equal, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_eq_scalar, intrinsic_eq_scalar, scalar_eq_scalar, intrinsic_eq},
            {Timestamp(TimeUnit::Millisecond), Timestamp(TimeUnit::Millisecond), NotEqual, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_ne_scalar, intrinsic_ne_scalar, scalar_ne_scalar, intrinsic_ne},
            {Timestamp(TimeUnit::Millisecond), Timestamp(TimeUnit::Millisecond), GreaterThan, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_gt_scalar, intrinsic_lt_scalar, scalar_gt_scalar, intrinsic_gt},
            {Timestamp(TimeUnit::Millisecond), Timestamp(TimeUnit::Millisecond), GreaterThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_ge_scalar, intrinsic_le_scalar, scalar_ge_scalar, intrinsic_ge},
            {Timestamp(TimeUnit::Millisecond), Timestamp(TimeUnit::Millisecond), LessThan, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_lt_scalar, intrinsic_gt_scalar, scalar_lt_scalar, intrinsic_lt},
            {Timestamp(TimeUnit::Millisecond), Timestamp(TimeUnit::Millisecond), LessThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_le_scalar, intrinsic_ge_scalar, scalar_le_scalar, intrinsic_le},
            {Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Equal, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_eq_scalar, intrinsic_eq_scalar, scalar_eq_scalar, intrinsic_eq},
            {Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, NotEqual, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_ne_scalar, intrinsic_ne_scalar, scalar_ne_scalar, intrinsic_ne},
            {Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, GreaterThan, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_gt_scalar, intrinsic_lt_scalar, scalar_gt_scalar, intrinsic_gt},
            {Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, GreaterThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_ge_scalar, intrinsic_le_scalar, scalar_ge_scalar, intrinsic_ge},
            {Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, LessThan, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_lt_scalar, intrinsic_gt_scalar, scalar_lt_scalar, intrinsic_lt},
            {Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, LessThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_le_scalar, intrinsic_ge_scalar, scalar_le_scalar, intrinsic_le},
            {Timestamp(TimeUnit::Nanosecond), Timestamp(TimeUnit::Nanosecond), Equal, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_eq_scalar, intrinsic_eq_scalar, scalar_eq_scalar, intrinsic_eq},
            {Timestamp(TimeUnit::Nanosecond), Timestamp(TimeUnit::Nanosecond), NotEqual, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_ne_scalar, intrinsic_ne_scalar, scalar_ne_scalar, intrinsic_ne},
            {Timestamp(TimeUnit::Nanosecond), Timestamp(TimeUnit::Nanosecond), GreaterThan, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_gt_scalar, intrinsic_lt_scalar, scalar_gt_scalar, intrinsic_gt},
            {Timestamp(TimeUnit::Nanosecond), Timestamp(TimeUnit::Nanosecond), GreaterThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_ge_scalar, intrinsic_le_scalar, scalar_ge_scalar, intrinsic_ge},
            {Timestamp(TimeUnit::Nanosecond), Timestamp(TimeUnit::Nanosecond), LessThan, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_lt_scalar, intrinsic_gt_scalar, scalar_lt_scalar, intrinsic_lt},
            {Timestamp(TimeUnit::Nanosecond), Timestamp(TimeUnit::Nanosecond), LessThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_le_scalar, intrinsic_ge_scalar, scalar_le_scalar, intrinsic_le},

            // // Timestamp: low precision lhs with high precision rhs
            {Timestamp(TimeUnit::Second), Timestamp(TimeUnit::Millisecond), Equal, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_eq_scalar::<1_000, 1>, timestamp_eq_scalar::<1_000, 1>, timestamp_scalar_eq_scalar::<1_000, 1>, timestamp_eq::<1_000, 1>},
            {Timestamp(TimeUnit::Second), Timestamp(TimeUnit::Millisecond), NotEqual, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_ne_scalar::<1_000, 1>, timestamp_ne_scalar::<1_000, 1>, timestamp_scalar_ne_scalar::<1_000, 1>, timestamp_ne::<1_000, 1>},
            {Timestamp(TimeUnit::Second), Timestamp(TimeUnit::Millisecond), GreaterThan, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_gt_scalar::<1_000, 1>, timestamp_lt_scalar::<1_000, 1>, timestamp_scalar_gt_scalar::<1_000, 1>, timestamp_gt::<1_000, 1>},
            {Timestamp(TimeUnit::Second), Timestamp(TimeUnit::Millisecond), GreaterThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_ge_scalar::<1_000, 1>, timestamp_le_scalar::<1_000, 1>, timestamp_scalar_ge_scalar::<1_000, 1>, timestamp_ge::<1_000, 1>},
            {Timestamp(TimeUnit::Second), Timestamp(TimeUnit::Millisecond), LessThan, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_lt_scalar::<1_000, 1>, timestamp_gt_scalar::<1_000, 1>, timestamp_scalar_lt_scalar::<1_000, 1>, timestamp_lt::<1_000, 1>},
            {Timestamp(TimeUnit::Second), Timestamp(TimeUnit::Millisecond), LessThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_le_scalar::<1_000, 1>, timestamp_ge_scalar::<1_000, 1>, timestamp_scalar_le_scalar::<1_000, 1>, timestamp_le::<1_000, 1>},
            {Timestamp(TimeUnit::Second), Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Equal, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_eq_scalar::<1_000_000, 1>, timestamp_eq_scalar::<1_000_000, 1>, timestamp_scalar_eq_scalar::<1_000_000, 1>, timestamp_eq::<1_000_000, 1>},
            {Timestamp(TimeUnit::Second), Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, NotEqual, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_ne_scalar::<1_000_000, 1>, timestamp_ne_scalar::<1_000_000, 1>, timestamp_scalar_ne_scalar::<1_000_000, 1>, timestamp_ne::<1_000_000, 1>},
            {Timestamp(TimeUnit::Second), Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, GreaterThan, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_gt_scalar::<1_000_000, 1>, timestamp_lt_scalar::<1_000_000, 1>, timestamp_scalar_gt_scalar::<1_000_000, 1>, timestamp_gt::<1_000_000, 1>},
            {Timestamp(TimeUnit::Second), Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, GreaterThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_ge_scalar::<1_000_000, 1>, timestamp_le_scalar::<1_000_000, 1>, timestamp_scalar_ge_scalar::<1_000_000, 1>, timestamp_ge::<1_000_000, 1>},
            {Timestamp(TimeUnit::Second), Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, LessThan, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_lt_scalar::<1_000_000, 1>, timestamp_gt_scalar::<1_000_000, 1>, timestamp_scalar_lt_scalar::<1_000_000, 1>, timestamp_lt::<1_000_000, 1>},
            {Timestamp(TimeUnit::Second), Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, LessThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_le_scalar::<1_000_000, 1>, timestamp_ge_scalar::<1_000_000, 1>, timestamp_scalar_ge_scalar::<1_000_000, 1>, timestamp_le::<1_000_000, 1>},
            {Timestamp(TimeUnit::Second), Timestamp(TimeUnit::Nanosecond), Equal, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_eq_scalar::<1_000_000_000, 1>, timestamp_eq_scalar::<1_000_000_000, 1>, timestamp_scalar_eq_scalar::<1_000_000_000, 1>, timestamp_eq::<1_000_000_000, 1>},
            {Timestamp(TimeUnit::Second), Timestamp(TimeUnit::Nanosecond), NotEqual, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_ne_scalar::<1_000_000_000, 1>, timestamp_ne_scalar::<1_000_000_000, 1>, timestamp_scalar_ne_scalar::<1_000_000_000, 1>, timestamp_ne::<1_000_000_000, 1>},
            {Timestamp(TimeUnit::Second), Timestamp(TimeUnit::Nanosecond), GreaterThan, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_gt_scalar::<1_000_000_000, 1>, timestamp_lt_scalar::<1_000_000_000, 1>, timestamp_scalar_gt_scalar::<1_000_000_000, 1>, timestamp_gt::<1_000_000_000, 1>},
            {Timestamp(TimeUnit::Second), Timestamp(TimeUnit::Nanosecond), GreaterThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_ge_scalar::<1_000_000_000, 1>, timestamp_le_scalar::<1_000_000_000, 1>, timestamp_scalar_ge_scalar::<1_000_000_000, 1>, timestamp_ge::<1_000_000_000, 1>},
            {Timestamp(TimeUnit::Second), Timestamp(TimeUnit::Nanosecond), LessThan, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_lt_scalar::<1_000_000_000, 1>, timestamp_gt_scalar::<1_000_000_000, 1>, timestamp_scalar_lt_scalar::<1_000_000_000, 1>, timestamp_lt::<1_000_000_000, 1>},
            {Timestamp(TimeUnit::Second), Timestamp(TimeUnit::Nanosecond), LessThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_le_scalar::<1_000_000_000, 1>, timestamp_ge_scalar::<1_000_000_000, 1>, timestamp_scalar_le_scalar::<1_000_000_000, 1>, timestamp_le::<1_000_000_000, 1>},
            {Timestamp(TimeUnit::Millisecond), Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Equal, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_eq_scalar::<1_000, 1>, timestamp_eq_scalar::<1_000, 1>, timestamp_scalar_eq_scalar::<1_000, 1>, timestamp_eq::<1_000, 1>},
            {Timestamp(TimeUnit::Millisecond), Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, NotEqual, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_ne_scalar::<1_000, 1>, timestamp_ne_scalar::<1_000, 1>, timestamp_scalar_ne_scalar::<1_000, 1>, timestamp_ne::<1_000, 1>},
            {Timestamp(TimeUnit::Millisecond), Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, GreaterThan, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_gt_scalar::<1_000, 1>, timestamp_lt_scalar::<1_000, 1>, timestamp_scalar_gt_scalar::<1_000, 1>, timestamp_gt::<1_000, 1>},
            {Timestamp(TimeUnit::Millisecond), Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, GreaterThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_ge_scalar::<1_000, 1>, timestamp_le_scalar::<1_000, 1>, timestamp_scalar_ge_scalar::<1_000, 1>, timestamp_ge::<1_000, 1>},
            {Timestamp(TimeUnit::Millisecond), Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, LessThan, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_lt_scalar::<1_000, 1>, timestamp_gt_scalar::<1_000, 1>, timestamp_scalar_lt_scalar::<1_000, 1>, timestamp_lt::<1_000, 1>},
            {Timestamp(TimeUnit::Millisecond), Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, LessThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_le_scalar::<1_000, 1>, timestamp_ge_scalar::<1_000, 1>, timestamp_scalar_le_scalar::<1_000, 1>, timestamp_le::<1_000, 1>},
            {Timestamp(TimeUnit::Millisecond), Timestamp(TimeUnit::Nanosecond), Equal, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_eq_scalar::<1_000_000, 1>, timestamp_eq_scalar::<1_000_000, 1>, timestamp_scalar_eq_scalar::<1_000_000, 1>, timestamp_eq::<1_000_000, 1>},
            {Timestamp(TimeUnit::Millisecond), Timestamp(TimeUnit::Nanosecond), NotEqual, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_ne_scalar::<1_000_000, 1>, timestamp_ne_scalar::<1_000_000, 1>, timestamp_scalar_ne_scalar::<1_000_000, 1>, timestamp_ne::<1_000_000, 1>},
            {Timestamp(TimeUnit::Millisecond), Timestamp(TimeUnit::Nanosecond), GreaterThan, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_gt_scalar::<1_000_000, 1>, timestamp_lt_scalar::<1_000_000, 1>, timestamp_scalar_gt_scalar::<1_000_000, 1>, timestamp_gt::<1_000_000, 1>},
            {Timestamp(TimeUnit::Millisecond), Timestamp(TimeUnit::Nanosecond), GreaterThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_ge_scalar::<1_000_000, 1>, timestamp_le_scalar::<1_000_000, 1>, timestamp_scalar_ge_scalar::<1_000_000, 1>, timestamp_ge::<1_000_000, 1>},
            {Timestamp(TimeUnit::Millisecond), Timestamp(TimeUnit::Nanosecond), LessThan, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_lt_scalar::<1_000_000, 1>, timestamp_gt_scalar::<1_000_000, 1>, timestamp_scalar_lt_scalar::<1_000_000, 1>, timestamp_lt::<1_000_000, 1>},
            {Timestamp(TimeUnit::Millisecond), Timestamp(TimeUnit::Nanosecond), LessThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_le_scalar::<1_000_000, 1>, timestamp_ge_scalar::<1_000_000, 1>, timestamp_scalar_le_scalar::<1_000_000, 1>, timestamp_le::<1_000_000, 1>},
            {Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Timestamp(TimeUnit::Nanosecond), Equal, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_eq_scalar::<1_000, 1>, timestamp_eq_scalar::<1_000, 1>, timestamp_scalar_eq_scalar::<1_000, 1>, timestamp_eq::<1_000, 1>},
            {Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Timestamp(TimeUnit::Nanosecond), NotEqual, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_ne_scalar::<1_000, 1>, timestamp_ne_scalar::<1_000, 1>, timestamp_scalar_ne_scalar::<1_000, 1>, timestamp_ne::<1_000, 1>},
            {Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Timestamp(TimeUnit::Nanosecond), GreaterThan, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_gt_scalar::<1_000, 1>, timestamp_lt_scalar::<1_000, 1>, timestamp_scalar_gt_scalar::<1_000, 1>, timestamp_gt::<1_000, 1>},
            {Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Timestamp(TimeUnit::Nanosecond), GreaterThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_ge_scalar::<1_000, 1>, timestamp_le_scalar::<1_000, 1>, timestamp_scalar_ge_scalar::<1_000, 1>, timestamp_ge::<1_000, 1>},
            {Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Timestamp(TimeUnit::Nanosecond), LessThan, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_lt_scalar::<1_000, 1>, timestamp_gt_scalar::<1_000, 1>, timestamp_scalar_lt_scalar::<1_000, 1>, timestamp_lt::<1_000, 1>},
            {Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Timestamp(TimeUnit::Nanosecond), LessThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_le_scalar::<1_000, 1>, timestamp_ge_scalar::<1_000, 1>, timestamp_scalar_le_scalar::<1_000, 1>, timestamp_le::<1_000, 1>},

            // Timestamp: high precision lhs with low precision rhs
            {Timestamp(TimeUnit::Millisecond), Timestamp(TimeUnit::Second), Equal, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_eq_scalar::<1, 1_000>, timestamp_eq_scalar::<1, 1_000>, timestamp_scalar_eq_scalar::<1, 1_000>, timestamp_eq::<1, 1_000>},
            {Timestamp(TimeUnit::Millisecond), Timestamp(TimeUnit::Second), NotEqual, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_ne_scalar::<1, 1_000>, timestamp_ne_scalar::<1, 1_000>, timestamp_scalar_ne_scalar::<1, 1_000>, timestamp_ne::<1, 1_000>},
            {Timestamp(TimeUnit::Millisecond), Timestamp(TimeUnit::Second), GreaterThan, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_gt_scalar::<1, 1_000>, timestamp_lt_scalar::<1, 1_000>, timestamp_scalar_gt_scalar::<1, 1_000>, timestamp_gt::<1, 1_000>},
            {Timestamp(TimeUnit::Millisecond), Timestamp(TimeUnit::Second), GreaterThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_ge_scalar::<1, 1_000>, timestamp_le_scalar::<1, 1_000>, timestamp_scalar_ge_scalar::<1, 1_000>, timestamp_ge::<1, 1_000>},
            {Timestamp(TimeUnit::Millisecond), Timestamp(TimeUnit::Second), LessThan, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_lt_scalar::<1, 1_000>, timestamp_gt_scalar::<1, 1_000>, timestamp_scalar_lt_scalar::<1, 1_000>, timestamp_lt::<1, 1_000>},
            {Timestamp(TimeUnit::Millisecond), Timestamp(TimeUnit::Second), LessThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>, timestamp_le_scalar::<1, 1_000>, timestamp_ge_scalar::<1, 1_000>, timestamp_scalar_le_scalar::<1, 1_000>, timestamp_le::<1, 1_000>},
            {Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Timestamp(TimeUnit::Second), Equal, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_eq_scalar::<1, 1_000_000>, timestamp_eq_scalar::<1, 1_000_000>, timestamp_scalar_eq_scalar::<1, 1_000_000>, timestamp_eq::<1, 1_000_000>},
            {Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Timestamp(TimeUnit::Second), NotEqual, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_ne_scalar::<1, 1_000_000>, timestamp_ne_scalar::<1, 1_000_000>, timestamp_scalar_ne_scalar::<1, 1_000_000>, timestamp_ne::<1, 1_000_000>},
            {Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Timestamp(TimeUnit::Second), GreaterThan, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_gt_scalar::<1, 1_000_000>, timestamp_lt_scalar::<1, 1_000_000>, timestamp_scalar_gt_scalar::<1, 1_000_000>, timestamp_gt::<1, 1_000_000>},
            {Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Timestamp(TimeUnit::Second), GreaterThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_ge_scalar::<1, 1_000_000>, timestamp_le_scalar::<1, 1_000_000>, timestamp_scalar_ge_scalar::<1, 1_000_000>, timestamp_ge::<1, 1_000_000>},
            {Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Timestamp(TimeUnit::Second), LessThan, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_lt_scalar::<1, 1_000_000>, timestamp_gt_scalar::<1, 1_000_000>, timestamp_scalar_lt_scalar::<1, 1_000_000>, timestamp_lt::<1, 1_000_000>},
            {Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Timestamp(TimeUnit::Second), LessThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_le_scalar::<1, 1_000_000>, timestamp_ge_scalar::<1, 1_000_000>, timestamp_scalar_le_scalar::<1, 1_000_000>, timestamp_le::<1, 1_000_000>},
            {Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Timestamp(TimeUnit::Millisecond), Equal, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_eq_scalar::<1, 1_000>, timestamp_eq_scalar::<1, 1_000>, timestamp_scalar_eq_scalar::<1, 1_000>, timestamp_eq::<1, 1_000>},
            {Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Timestamp(TimeUnit::Millisecond), NotEqual, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_ne_scalar::<1, 1_000>, timestamp_ne_scalar::<1, 1_000>, timestamp_scalar_ne_scalar::<1, 1_000>, timestamp_ne::<1, 1_000>},
            {Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Timestamp(TimeUnit::Millisecond), GreaterThan, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_gt_scalar::<1, 1_000>, timestamp_lt_scalar::<1, 1_000>, timestamp_scalar_gt_scalar::<1, 1_000>, timestamp_gt::<1, 1_000>},
            {Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Timestamp(TimeUnit::Millisecond), GreaterThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_ge_scalar::<1, 1_000>, timestamp_le_scalar::<1, 1_000>, timestamp_scalar_ge_scalar::<1, 1_000>, timestamp_ge::<1, 1_000>},
            {Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Timestamp(TimeUnit::Millisecond), LessThan, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_lt_scalar::<1, 1_000>, timestamp_gt_scalar::<1, 1_000>, timestamp_scalar_lt_scalar::<1, 1_000>, timestamp_lt::<1, 1_000>},
            {Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Timestamp(TimeUnit::Millisecond), LessThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_le_scalar::<1, 1_000>, timestamp_ge_scalar::<1, 1_000>, timestamp_scalar_le_scalar::<1, 1_000>, timestamp_le::<1, 1_000>},
            {Timestamp(TimeUnit::Nanosecond), Timestamp(TimeUnit::Second), Equal, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_eq_scalar::<1, 1_000_000_000>, timestamp_eq_scalar::<1, 1_000_000_000>, timestamp_scalar_eq_scalar::<1, 1_000_000_000>, timestamp_eq::<1, 1_000_000_000>},
            {Timestamp(TimeUnit::Nanosecond), Timestamp(TimeUnit::Second), NotEqual, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_ne_scalar::<1, 1_000_000_000>, timestamp_ne_scalar::<1, 1_000_000_000>, timestamp_scalar_ne_scalar::<1, 1_000_000_000>, timestamp_ne::<1, 1_000_000_000>},
            {Timestamp(TimeUnit::Nanosecond), Timestamp(TimeUnit::Second), GreaterThan, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_gt_scalar::<1, 1_000_000_000>, timestamp_lt_scalar::<1, 1_000_000_000>, timestamp_scalar_gt_scalar::<1, 1_000_000_000>, timestamp_gt::<1, 1_000_000_000>},
            {Timestamp(TimeUnit::Nanosecond), Timestamp(TimeUnit::Second), GreaterThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_ge_scalar::<1, 1_000_000_000>, timestamp_le_scalar::<1, 1_000_000_000>, timestamp_scalar_ge_scalar::<1, 1_000_000_000>, timestamp_ge::<1, 1_000_000_000>},
            {Timestamp(TimeUnit::Nanosecond), Timestamp(TimeUnit::Second), LessThan, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_lt_scalar::<1, 1_000_000_000>, timestamp_gt_scalar::<1, 1_000_000_000>, timestamp_scalar_lt_scalar::<1, 1_000_000_000>, timestamp_lt::<1, 1_000_000_000>},
            {Timestamp(TimeUnit::Nanosecond), Timestamp(TimeUnit::Second), LessThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_le_scalar::<1, 1_000_000_000>, timestamp_ge_scalar::<1, 1_000_000_000>, timestamp_scalar_le_scalar::<1, 1_000_000_000>, timestamp_le::<1, 1_000_000_000>},
            {Timestamp(TimeUnit::Nanosecond), Timestamp(TimeUnit::Millisecond), Equal, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_eq_scalar::<1, 1_000_000>, timestamp_eq_scalar::<1, 1_000_000>, timestamp_scalar_eq_scalar::<1, 1_000_000>, timestamp_eq::<1, 1_000_000>},
            {Timestamp(TimeUnit::Nanosecond), Timestamp(TimeUnit::Millisecond), NotEqual, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_ne_scalar::<1, 1_000_000>, timestamp_ne_scalar::<1, 1_000_000>, timestamp_scalar_ne_scalar::<1, 1_000_000>, timestamp_ne::<1, 1_000_000>},
            {Timestamp(TimeUnit::Nanosecond), Timestamp(TimeUnit::Millisecond), GreaterThan, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_gt_scalar::<1, 1_000_000>, timestamp_lt_scalar::<1, 1_000_000>, timestamp_scalar_gt_scalar::<1, 1_000_000>, timestamp_gt::<1, 1_000_000>},
            {Timestamp(TimeUnit::Nanosecond), Timestamp(TimeUnit::Millisecond), GreaterThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_ge_scalar::<1, 1_000_000>, timestamp_le_scalar::<1, 1_000_000>, timestamp_scalar_ge_scalar::<1, 1_000_000>, timestamp_ge::<1, 1_000_000>},
            {Timestamp(TimeUnit::Nanosecond), Timestamp(TimeUnit::Millisecond), LessThan, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_lt_scalar::<1, 1_000_000>, timestamp_gt_scalar::<1, 1_000_000>, timestamp_scalar_lt_scalar::<1, 1_000_000>, timestamp_lt::<1, 1_000_000>},
            {Timestamp(TimeUnit::Nanosecond), Timestamp(TimeUnit::Millisecond), LessThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_le_scalar::<1, 1_000_000>, timestamp_ge_scalar::<1, 1_000_000>, timestamp_scalar_le_scalar::<1, 1_000_000>, timestamp_le::<1, 1_000_000>},
            {Timestamp(TimeUnit::Nanosecond), Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, Equal, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_eq_scalar::<1, 1_000>, timestamp_eq_scalar::<1, 1_000>, timestamp_scalar_eq_scalar::<1, 1_000>, timestamp_eq::<1, 1_000>},
            {Timestamp(TimeUnit::Nanosecond), Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, NotEqual, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_ne_scalar::<1, 1_000>, timestamp_ne_scalar::<1, 1_000>, timestamp_scalar_ne_scalar::<1, 1_000>, timestamp_ne::<1, 1_000>},
            {Timestamp(TimeUnit::Nanosecond), Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, GreaterThan, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_gt_scalar::<1, 1_000>, timestamp_lt_scalar::<1, 1_000>, timestamp_scalar_gt_scalar::<1, 1_000>, timestamp_gt::<1, 1_000>},
            {Timestamp(TimeUnit::Nanosecond), Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, GreaterThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_ge_scalar::<1, 1_000>, timestamp_le_scalar::<1, 1_000>, timestamp_scalar_ge_scalar::<1, 1_000>, timestamp_ge::<1, 1_000>},
            {Timestamp(TimeUnit::Nanosecond), Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, LessThan, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_lt_scalar::<1, 1_000>, timestamp_gt_scalar::<1, 1_000>, timestamp_scalar_lt_scalar::<1, 1_000>, timestamp_lt::<1, 1_000>},
            {Timestamp(TimeUnit::Nanosecond), Timestamp(TimeUnit::Microsecond) | Timestamptz{..}, LessThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>,  timestamp_le_scalar::<1, 1_000>, timestamp_ge_scalar::<1, 1_000>, timestamp_scalar_le_scalar::<1, 1_000>, timestamp_le::<1, 1_000>},

            // Time
            {Time | Timetz{..}, Time | Timetz{..}, Equal, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_eq_scalar, intrinsic_eq_scalar, scalar_eq_scalar, intrinsic_eq},
            {Time | Timetz{..}, Time | Timetz{..}, NotEqual, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_ne_scalar, intrinsic_ne_scalar, scalar_ne_scalar, intrinsic_ne},
            {Time | Timetz{..}, Time | Timetz{..}, GreaterThan, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_gt_scalar, intrinsic_lt_scalar, scalar_gt_scalar, intrinsic_gt},
            {Time | Timetz{..}, Time | Timetz{..}, GreaterThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_ge_scalar, intrinsic_le_scalar, scalar_ge_scalar, intrinsic_ge},
            {Time | Timetz{..}, Time | Timetz{..}, LessThan, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_lt_scalar, intrinsic_gt_scalar, scalar_lt_scalar, intrinsic_lt},
            {Time | Timetz{..}, Time | Timetz{..}, LessThanOrEqualTo, PrimitiveArray<i64>, PrimitiveArray<i64>, intrinsic_le_scalar, intrinsic_ge_scalar, scalar_le_scalar, intrinsic_le},

            // Date
            {Date, Date, Equal, PrimitiveArray<i32>, PrimitiveArray<i32>, intrinsic_eq_scalar, intrinsic_eq_scalar, scalar_eq_scalar, intrinsic_eq},
            {Date, Date, NotEqual, PrimitiveArray<i32>, PrimitiveArray<i32>, intrinsic_ne_scalar, intrinsic_ne_scalar, scalar_ne_scalar, intrinsic_ne},
            {Date, Date, GreaterThan, PrimitiveArray<i32>, PrimitiveArray<i32>, intrinsic_gt_scalar, intrinsic_lt_scalar, scalar_gt_scalar, intrinsic_gt},
            {Date, Date, GreaterThanOrEqualTo, PrimitiveArray<i32>, PrimitiveArray<i32>, intrinsic_ge_scalar, intrinsic_le_scalar, scalar_ge_scalar, intrinsic_ge},
            {Date, Date, LessThan, PrimitiveArray<i32>, PrimitiveArray<i32>, intrinsic_lt_scalar, intrinsic_gt_scalar, scalar_lt_scalar, intrinsic_lt},
            {Date, Date, LessThanOrEqualTo, PrimitiveArray<i32>, PrimitiveArray<i32>, intrinsic_le_scalar, intrinsic_ge_scalar, scalar_le_scalar, intrinsic_le},

            // Uuid
            {Uuid, Uuid, Equal, PrimitiveArray<i128>, PrimitiveArray<i128>, primitive_eq_scalar, primitive_eq_scalar, scalar_eq_scalar, primitive_eq},
            {Uuid, Uuid, NotEqual, PrimitiveArray<i128>, PrimitiveArray<i128>, primitive_ne_scalar, primitive_ne_scalar, scalar_ne_scalar, primitive_ne},

            // IPv4
            {IPv4, IPv4, Equal, PrimitiveArray<i32>, PrimitiveArray<i32>, intrinsic_eq_scalar, intrinsic_eq_scalar, scalar_eq_scalar, intrinsic_eq},
            {IPv4, IPv4, NotEqual, PrimitiveArray<i32>, PrimitiveArray<i32>, intrinsic_ne_scalar, intrinsic_ne_scalar, scalar_ne_scalar, intrinsic_ne},

            // IPv6
            {IPv6, IPv6, Equal, PrimitiveArray<i128>, PrimitiveArray<i128>, primitive_eq_scalar, primitive_eq_scalar, scalar_eq_scalar, primitive_eq},
            {IPv6, IPv6, NotEqual, PrimitiveArray<i128>, PrimitiveArray<i128>, primitive_ne_scalar, primitive_ne_scalar, scalar_ne_scalar, primitive_ne},

            // Boolean
            {Boolean, Boolean, Equal, BooleanArray, BooleanArray, boolean_eq_scalar, boolean_eq_scalar, scalar_eq_scalar, boolean_eq},
            {Boolean, Boolean, NotEqual, BooleanArray, BooleanArray, boolean_ne_scalar, boolean_ne_scalar, scalar_ne_scalar, boolean_ne},
            {Boolean, Boolean, GreaterThan, BooleanArray, BooleanArray, boolean_gt_scalar, boolean_lt_scalar, scalar_gt_scalar, boolean_gt},
            {Boolean, Boolean, GreaterThanOrEqualTo, BooleanArray, BooleanArray, boolean_ge_scalar, boolean_le_scalar, scalar_ge_scalar, boolean_ge},
            {Boolean, Boolean, LessThan, BooleanArray, BooleanArray, boolean_lt_scalar, boolean_gt_scalar, scalar_lt_scalar, boolean_lt},
            {Boolean, Boolean, LessThanOrEqualTo, BooleanArray, BooleanArray, boolean_le_scalar, boolean_ge_scalar, scalar_ge_scalar, boolean_le},

            // VarChar
            {VarChar, VarChar, Equal, StringArray, StringArray, string_eq_scalar, string_eq_scalar, scalar_eq_scalar, string_eq},
            {VarChar, VarChar, NotEqual, StringArray, StringArray, string_ne_scalar, string_ne_scalar, scalar_ne_scalar, string_ne},
            {VarChar, VarChar, GreaterThan, StringArray, StringArray, string_gt_scalar, string_lt_scalar, scalar_gt_scalar, string_gt},
            {VarChar, VarChar, GreaterThanOrEqualTo, StringArray, StringArray, string_ge_scalar, string_le_scalar, scalar_ge_scalar, string_ge},
            {VarChar, VarChar, LessThan, StringArray, StringArray, string_lt_scalar, string_gt_scalar, scalar_lt_scalar, string_lt},
            {VarChar, VarChar, LessThanOrEqualTo, StringArray, StringArray, string_le_scalar, string_ge_scalar, scalar_ge_scalar, string_le},

            // VarBinary
            {VarBinary, VarBinary, Equal, BinaryArray, BinaryArray, eq_scalar_default, eq_scalar_default, scalar_eq_scalar, eq_default},
            {VarBinary, VarBinary, NotEqual, BinaryArray, BinaryArray, ne_scalar_default, ne_scalar_default, scalar_ne_scalar, ne_default},
            {VarBinary, VarBinary, GreaterThan, BinaryArray, BinaryArray, gt_scalar_default, lt_scalar_default, scalar_gt_scalar, gt_default},
            {VarBinary, VarBinary, GreaterThanOrEqualTo, BinaryArray, BinaryArray, ge_scalar_default, le_scalar_default, scalar_ge_scalar, ge_default},
            {VarBinary, VarBinary, LessThan, BinaryArray, BinaryArray, lt_scalar_default, gt_scalar_default, scalar_lt_scalar, lt_default},
            {VarBinary, VarBinary, LessThanOrEqualTo, BinaryArray, BinaryArray, le_scalar_default, ge_scalar_default, scalar_le_scalar, le_default}
        }
    };
}

/// Infer the comparison function set
///
/// Returns `Some(ComparisonFunctionSet)` if the comparison is supported, otherwise `None`
pub fn infer_comparison_func_set(
    left: &LogicalType,
    right: &LogicalType,
    op: CmpOperator,
) -> Option<ComparisonFunctionSet> {
    use CmpOperator::*;
    use LogicalType::*;
    macro_rules! impl_comparison {
        ($({$left:pat, $right:pat, $op:ident, $left_ty:ty, $right_ty:ty, $array_scalar_func:path, $scalar_array_func:path, $scalars_func:path, $arrays_func:path}),+) => {
            match (left, right, op) {
                $(
                    ($left, $right, $op) => Some(ComparisonFunctionSet{
                        scalar_cmp_scalar: cmp_scalars_func!($left, $left_ty, $right, $right_ty, $scalars_func),
                        array_cmp_scalar: cmp_array_scalar_func!($left, $left_ty, $right, $right_ty, $array_scalar_func),
                        scalar_cmp_array: cmp_array_scalar_func!($left, $left_ty, $right, $right_ty, $scalar_array_func),
                        array_cmp_array: cmp_arrays_func!($left, $left_ty, $right, $right_ty, $arrays_func),
                    }),
                )+

                _ => None,
            }
        };
    }

    for_all_comparison!(impl_comparison)
}
