//! Comparison between [`IntrinsicType`]
//!
//! Following implementation heavily depends on the reading intrinsic types from
//! uninitialized  memory optimization. However, it is undefined behavior in the Rust.
//! See following links for details. In our cases, the cpu is the main reason to avoid
//! reading uninitialized memory. Should we remove this optimization?
//
//! - <https://doc.rust-lang.org/nightly/reference/behavior-considered-undefined.html>
//! - <https://www.ralfj.de/blog/2019/07/14/uninit.html>
//! - <https://stackoverflow.com/questions/11962457/why-is-using-an-uninitialized-variable-undefined-behavior>
//! - <https://langdev.stackexchange.com/questions/2870/why-would-accessing-uninitialized-memory-necessarily-be-undefined-behavior>

use crate::array::{Array, BooleanArray, PrimitiveArray};
use crate::bitmap::Bitmap;
use crate::compute::logical::and_inplace;
use crate::types::IntrinsicType;

#[cfg(feature = "portable_simd")]
mod portable_simd;
#[cfg(feature = "portable_simd")]
pub use portable_simd::*;

#[cfg(not(feature = "portable_simd"))]
mod stable;
#[cfg(not(feature = "portable_simd"))]
pub use stable::*;

/// Extension for partial order
///
/// # Note
///
/// We only distinguish the threshold based on the cpu flag, we do not include operator
/// here. Because different operators have same latency and throughput
pub trait PartialOrdExt: PartialOrd + IntrinsicType {
    /// If the selectivity is smaller than this threshold, partial computation is used
    const PARTIAL_CMP_THRESHOLD: f64;
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    /// If the selectivity is smaller than this threshold, partial computation is used
    const NEON_PARTIAL_CMP_THRESHOLD: f64;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// If the selectivity is smaller than this threshold, partial computation is used
    const AVX2_PARTIAL_CMP_THRESHOLD: f64;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// If the selectivity is smaller than this threshold, partial computation is used
    const AVX512_PARTIAL_CMP_THRESHOLD: f64;

    /// Array equal scalar function
    ///
    /// # Safety
    ///
    /// - If the `selection` is not empty, `array` and `selection` should have same length.
    /// Otherwise, undefined behavior happens
    ///
    /// - `selection` should not be referenced by any array
    ///
    /// - `array`'s data and validity should not reference `temp`'s data and validity. In the computation
    /// graph, `lhs` must be the descendant of `temp`
    ///
    /// - No other arrays that reference the `temp`'s data and validity are accessed! In the
    /// computation graph, it will never happens
    unsafe fn eq_scalar(
        selection: &mut Bitmap,
        array: &PrimitiveArray<Self>,
        scalar: Self,
        temp: &mut BooleanArray,
    );

    /// Array not equal scalar function
    ///
    /// # Safety
    ///
    /// - If the `selection` is not empty, `array` and `selection` should have same length.
    /// Otherwise, undefined behavior happens
    ///
    /// - `selection` should not be referenced by any array
    ///
    /// - `array`'s data and validity should not reference `temp`'s data and validity. In the computation
    /// graph, `lhs` must be the descendant of `temp`
    ///
    /// - No other arrays that reference the `temp`'s data and validity are accessed! In the
    /// computation graph, it will never happens
    unsafe fn ne_scalar(
        selection: &mut Bitmap,
        array: &PrimitiveArray<Self>,
        scalar: Self,
        temp: &mut BooleanArray,
    );

    /// Array greater than scalar function
    ///
    /// # Safety
    ///
    /// - If the `selection` is not empty, `array` and `selection` should have same length.
    /// Otherwise, undefined behavior happens
    ///
    /// - `selection` should not be referenced by any array
    ///
    /// - `array`'s data and validity should not reference `temp`'s data and validity. In the computation
    /// graph, `lhs` must be the descendant of `temp`
    ///
    /// - No other arrays that reference the `temp`'s data and validity are accessed! In the
    /// computation graph, it will never happens
    unsafe fn gt_scalar(
        selection: &mut Bitmap,
        array: &PrimitiveArray<Self>,
        scalar: Self,
        temp: &mut BooleanArray,
    );

    /// Array greater than or equal to scalar function
    ///
    /// # Safety
    ///
    /// - If the `selection` is not empty, `array` and `selection` should have same length.
    /// Otherwise, undefined behavior happens
    ///
    /// - `selection` should not be referenced by any array
    ///
    /// - `array`'s data and validity should not reference `temp`'s data and validity. In the computation
    /// graph, `lhs` must be the descendant of `temp`
    ///
    /// - No other arrays that reference the `temp`'s data and validity are accessed! In the
    /// computation graph, it will never happens
    unsafe fn ge_scalar(
        selection: &mut Bitmap,
        array: &PrimitiveArray<Self>,
        scalar: Self,
        temp: &mut BooleanArray,
    );

    /// Array less than scalar function
    ///
    /// # Safety
    ///
    /// - If the `selection` is not empty, `array` and `selection` should have same length.
    /// Otherwise, undefined behavior happens
    ///
    /// - `selection` should not be referenced by any array
    ///
    /// - `array`'s data and validity should not reference `temp`'s data and validity. In the computation
    /// graph, `lhs` must be the descendant of `temp`
    ///
    /// - No other arrays that reference the `temp`'s data and validity are accessed! In the
    /// computation graph, it will never happens
    unsafe fn lt_scalar(
        selection: &mut Bitmap,
        array: &PrimitiveArray<Self>,
        scalar: Self,
        temp: &mut BooleanArray,
    );

    /// Array less than or equal to scalar function
    ///
    /// # Safety
    ///
    /// - If the `selection` is not empty, `array` and `selection` should have same length.
    /// Otherwise, undefined behavior happens
    ///
    /// - `selection` should not be referenced by any array
    ///
    /// - `array`'s data and validity should not reference `temp`'s data and validity. In the computation
    /// graph, `lhs` must be the descendant of `temp`
    ///
    /// - No other arrays that reference the `temp`'s data and validity are accessed! In the
    /// computation graph, it will never happens
    unsafe fn le_scalar(
        selection: &mut Bitmap,
        array: &PrimitiveArray<Self>,
        scalar: Self,
        temp: &mut BooleanArray,
    );
}

/// Perform `array == scalar` between `PrimitiveArray<T>` and `T`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
///
/// - `array`'s data and validity should not reference `temp`'s data and validity. In the computation
/// graph, `lhs` must be the descendant of `temp`
///
/// - No other arrays that reference the `temp`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn eq_scalar<T>(
    selection: &mut Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    temp: &mut BooleanArray,
) where
    T: PartialOrdExt,
    PrimitiveArray<T>: Array<Element = T>,
{
    T::eq_scalar(selection, array, scalar, temp)
}

/// Perform `array != scalar` between `PrimitiveArray<T>` and `T`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
///
/// - `array`'s data and validity should not reference `temp`'s data and validity. In the computation
/// graph, `lhs` must be the descendant of `temp`
///
/// - No other arrays that reference the `temp`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn ne_scalar<T>(
    selection: &mut Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    temp: &mut BooleanArray,
) where
    T: PartialOrdExt,
    PrimitiveArray<T>: Array<Element = T>,
{
    T::ne_scalar(selection, array, scalar, temp)
}

/// Perform `array > scalar` between `PrimitiveArray<T>` and `T`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
///
/// - `array`'s data and validity should not reference `temp`'s data and validity. In the computation
/// graph, `lhs` must be the descendant of `temp`
///
/// - No other arrays that reference the `temp`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn gt_scalar<T>(
    selection: &mut Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    temp: &mut BooleanArray,
) where
    T: PartialOrdExt,
    PrimitiveArray<T>: Array<Element = T>,
{
    T::gt_scalar(selection, array, scalar, temp)
}

/// Perform `array >= scalar` between `PrimitiveArray<T>` and `T`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
///
/// - `array`'s data and validity should not reference `temp`'s data and validity. In the computation
/// graph, `lhs` must be the descendant of `temp`
///
/// - No other arrays that reference the `temp`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn ge_scalar<T>(
    selection: &mut Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    temp: &mut BooleanArray,
) where
    T: PartialOrdExt,
    PrimitiveArray<T>: Array<Element = T>,
{
    T::ge_scalar(selection, array, scalar, temp)
}

/// Perform `array < scalar` between `PrimitiveArray<T>` and `T`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
///
/// - `array`'s data and validity should not reference `temp`'s data and validity. In the computation
/// graph, `lhs` must be the descendant of `temp`
///
/// - No other arrays that reference the `temp`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn lt_scalar<T>(
    selection: &mut Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    temp: &mut BooleanArray,
) where
    T: PartialOrdExt,
    PrimitiveArray<T>: Array<Element = T>,
{
    T::lt_scalar(selection, array, scalar, temp)
}

/// Perform `array <= scalar` between `PrimitiveArray<T>` and `T`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
///
/// - `array`'s data and validity should not reference `temp`'s data and validity. In the computation
/// graph, `lhs` must be the descendant of `temp`
///
/// - No other arrays that reference the `temp`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn le_scalar<T>(
    selection: &mut Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    temp: &mut BooleanArray,
) where
    T: PartialOrdExt,
    PrimitiveArray<T>: Array<Element = T>,
{
    T::le_scalar(selection, array, scalar, temp)
}

/// Default implementation
unsafe fn timestamp_cmp_scalar_default<const AM: i64, const SM: i64>(
    selection: &mut Bitmap,
    array: &PrimitiveArray<i64>,
    scalar: i64,
    temp: &mut BooleanArray,
    partial_cmp_threshold: f64,
    cmp_scalars_func: impl Fn(i64, i64) -> bool,
) {
    debug_assert_selection_is_valid!(selection, array);

    and_inplace(selection, array.validity());
    if selection.ones_ratio() < partial_cmp_threshold {
        selection
            .mutate()
            .mutate_ones(|index| cmp_scalars_func(array.get_value_unchecked(index), scalar))
    } else {
        // It may still faster 😊
        temp.data_mut().mutate().reset(
            array.len(),
            array
                .values_iter()
                .map(|element| cmp_scalars_func(element, scalar)),
        );
        and_inplace(selection, temp.data());
    }
}

#[cfg(debug_assertions)]
fn check_timestamp_array_and_multiplier<const MULTIPLIER: i64>(array: &PrimitiveArray<i64>) {
    use crate::types::{LogicalType, TimeUnit};
    match array.logical_type() {
        LogicalType::Timestamp(unit) => {
            match unit {
                TimeUnit::Second => {
                    if MULTIPLIER != 1_000 || MULTIPLIER != 1_000_000 || MULTIPLIER != 1_000_000_000{
                        panic!("Timestamp::Microsecond, the multiplier can only be `1_000`. However, the passed multiplier is `{}`", MULTIPLIER)
                    }
                }
                TimeUnit::Millisecond => {
                    if MULTIPLIER != 1_000 || MULTIPLIER != 1_000_000{
                        panic!("Timestamp::Microsecond, the multiplier can only be `1_000`/`1_000_000`. However, the passed multiplier is `{}`", MULTIPLIER)
                    }
                }
                TimeUnit::Microsecond => {
                    if MULTIPLIER != 1_000{
                        panic!("Timestamp::Microsecond, the multiplier can only be `1_000`. However, the passed multiplier is `{}`", MULTIPLIER)
                    }
                }
                TimeUnit::Nanosecond => {
                    panic!("timestamp_cmp_scalar is not suitable for Timestamp::Nanosecond, use cmp_scalar instead")
                }
            }
        }
        LogicalType::Timestamptz { .. } => {
            if MULTIPLIER != 1_000 {
                panic!("Timestamptz is stored in microsecond, the multiplier can only be `1000`. However, the passed multiplier is `{}`", MULTIPLIER)
            }
        }
        ty  => panic!("timestamp_cmp_scalar function should only be called on Int64Array that has logical type `Timestamp` or `Timestamptz`, called on array that has logical type: {:?}", ty),
    }
}
