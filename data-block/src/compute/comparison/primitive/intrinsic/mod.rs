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

    /// Equality between two arrays
    ///
    /// # Safety
    ///
    /// - If the `selection` is not empty, `lhs`/`rhs` and `selection` should have same length.
    /// Otherwise, undefined behavior happens
    ///
    /// - `selection` should not be referenced by any array
    ///
    /// - `lhs`/`rhs`'s data and validity should not reference `temp`'s data and validity. In the computation
    /// graph, `lhs`/`rhs` must be the descendant of `temp`
    ///
    /// - No other arrays that reference the `temp`'s data and validity are accessed! In the
    /// computation graph, it will never happens
    unsafe fn eq(
        selection: &mut Bitmap,
        lhs: &PrimitiveArray<Self>,
        rhs: &PrimitiveArray<Self>,
        temp: &mut BooleanArray,
    );

    /// Not equal between two arrays
    ///
    /// # Safety
    ///
    /// - If the `selection` is not empty, `lhs`/`rhs` and `selection` should have same length.
    /// Otherwise, undefined behavior happens
    ///
    /// - `selection` should not be referenced by any array
    ///
    /// - `lhs`/`rhs`'s data and validity should not reference `temp`'s data and validity. In the computation
    /// graph, `lhs`/`rhs` must be the descendant of `temp`
    ///
    /// - No other arrays that reference the `temp`'s data and validity are accessed! In the
    /// computation graph, it will never happens
    unsafe fn ne(
        selection: &mut Bitmap,
        lhs: &PrimitiveArray<Self>,
        rhs: &PrimitiveArray<Self>,
        temp: &mut BooleanArray,
    );

    /// Array greater than another array
    ///
    /// # Safety
    ///
    /// - If the `selection` is not empty, `lhs`/`rhs` and `selection` should have same length.
    /// Otherwise, undefined behavior happens
    ///
    /// - `selection` should not be referenced by any array
    ///
    /// - `lhs`/`rhs`'s data and validity should not reference `temp`'s data and validity. In the computation
    /// graph, `lhs`/`rhs` must be the descendant of `temp`
    ///
    /// - No other arrays that reference the `temp`'s data and validity are accessed! In the
    /// computation graph, it will never happens
    unsafe fn gt(
        selection: &mut Bitmap,
        lhs: &PrimitiveArray<Self>,
        rhs: &PrimitiveArray<Self>,
        temp: &mut BooleanArray,
    );

    /// Array greater than or equal to another array
    ///
    /// # Safety
    ///
    /// - If the `selection` is not empty, `lhs`/`rhs` and `selection` should have same length.
    /// Otherwise, undefined behavior happens
    ///
    /// - `selection` should not be referenced by any array
    ///
    /// - `lhs`/`rhs`'s data and validity should not reference `temp`'s data and validity. In the computation
    /// graph, `lhs`/`rhs` must be the descendant of `temp`
    ///
    /// - No other arrays that reference the `temp`'s data and validity are accessed! In the
    /// computation graph, it will never happens
    unsafe fn ge(
        selection: &mut Bitmap,
        lhs: &PrimitiveArray<Self>,
        rhs: &PrimitiveArray<Self>,
        temp: &mut BooleanArray,
    );

    /// Array less than another array
    ///
    /// # Safety
    ///
    /// - If the `selection` is not empty, `lhs`/`rhs` and `selection` should have same length.
    /// Otherwise, undefined behavior happens
    ///
    /// - `selection` should not be referenced by any array
    ///
    /// - `lhs`/`rhs`'s data and validity should not reference `temp`'s data and validity. In the computation
    /// graph, `lhs`/`rhs` must be the descendant of `temp`
    ///
    /// - No other arrays that reference the `temp`'s data and validity are accessed! In the
    /// computation graph, it will never happens
    unsafe fn lt(
        selection: &mut Bitmap,
        lhs: &PrimitiveArray<Self>,
        rhs: &PrimitiveArray<Self>,
        temp: &mut BooleanArray,
    );

    /// Array less than or equal to another array
    ///
    /// # Safety
    ///
    /// - If the `selection` is not empty, `lhs`/`rhs` and `selection` should have same length.
    /// Otherwise, undefined behavior happens
    ///
    /// - `selection` should not be referenced by any array
    ///
    /// - `lhs`/`rhs`'s data and validity should not reference `temp`'s data and validity. In the computation
    /// graph, `lhs`/`rhs` must be the descendant of `temp`
    ///
    /// - No other arrays that reference the `temp`'s data and validity are accessed! In the
    /// computation graph, it will never happens
    unsafe fn le(
        selection: &mut Bitmap,
        lhs: &PrimitiveArray<Self>,
        rhs: &PrimitiveArray<Self>,
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

/// Perform `PrimitiveArray<T> == PrimitiveArray<T>`
///
/// # Safety
///
/// - If the `selection` is not empty, `lhs`/`rhs` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
///
/// - `lhs`/`rhs`'s data and validity should not reference `temp`'s data and validity. In the computation
/// graph, `lhs`/`rhs` must be the descendant of `temp`
///
/// - No other arrays that reference the `temp`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn eq<T>(
    selection: &mut Bitmap,
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<T>,
    temp: &mut BooleanArray,
) where
    T: PartialOrdExt,
    PrimitiveArray<T>: Array<Element = T>,
{
    <T as PartialOrdExt>::eq(selection, lhs, rhs, temp)
}

/// Perform `PrimitiveArray<T> != PrimitiveArray<T>`
///
/// # Safety
///
/// - If the `selection` is not empty, `lhs`/`rhs` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
///
/// - `lhs`/`rhs`'s data and validity should not reference `temp`'s data and validity. In the computation
/// graph, `lhs`/`rhs` must be the descendant of `temp`
///
/// - No other arrays that reference the `temp`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn ne<T>(
    selection: &mut Bitmap,
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<T>,
    temp: &mut BooleanArray,
) where
    T: PartialOrdExt,
    PrimitiveArray<T>: Array<Element = T>,
{
    <T as PartialOrdExt>::ne(selection, lhs, rhs, temp)
}

/// Perform `PrimitiveArray<T> > PrimitiveArray<T>`
///
/// # Safety
///
/// - If the `selection` is not empty, `lhs`/`rhs` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
///
/// - `lhs`/`rhs`'s data and validity should not reference `temp`'s data and validity. In the computation
/// graph, `lhs`/`rhs` must be the descendant of `temp`
///
/// - No other arrays that reference the `temp`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn gt<T>(
    selection: &mut Bitmap,
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<T>,
    temp: &mut BooleanArray,
) where
    T: PartialOrdExt,
    PrimitiveArray<T>: Array<Element = T>,
{
    <T as PartialOrdExt>::gt(selection, lhs, rhs, temp)
}

/// Perform `PrimitiveArray<T> >= PrimitiveArray<T>`
///
/// # Safety
///
/// - If the `selection` is not empty, `lhs`/`rhs` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
///
/// - `lhs`/`rhs`'s data and validity should not reference `temp`'s data and validity. In the computation
/// graph, `lhs`/`rhs` must be the descendant of `temp`
///
/// - No other arrays that reference the `temp`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn ge<T>(
    selection: &mut Bitmap,
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<T>,
    temp: &mut BooleanArray,
) where
    T: PartialOrdExt,
    PrimitiveArray<T>: Array<Element = T>,
{
    <T as PartialOrdExt>::ge(selection, lhs, rhs, temp)
}

/// Perform `PrimitiveArray<T> < PrimitiveArray<T>`
///
/// # Safety
///
/// - If the `selection` is not empty, `lhs`/`rhs` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
///
/// - `lhs`/`rhs`'s data and validity should not reference `temp`'s data and validity. In the computation
/// graph, `lhs`/`rhs` must be the descendant of `temp`
///
/// - No other arrays that reference the `temp`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn lt<T>(
    selection: &mut Bitmap,
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<T>,
    temp: &mut BooleanArray,
) where
    T: PartialOrdExt,
    PrimitiveArray<T>: Array<Element = T>,
{
    <T as PartialOrdExt>::lt(selection, lhs, rhs, temp)
}

/// Perform `PrimitiveArray<T> <= PrimitiveArray<T>`
///
/// # Safety
///
/// - If the `selection` is not empty, `lhs`/`rhs` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
///
/// - `lhs`/`rhs`'s data and validity should not reference `temp`'s data and validity. In the computation
/// graph, `lhs`/`rhs` must be the descendant of `temp`
///
/// - No other arrays that reference the `temp`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn le<T>(
    selection: &mut Bitmap,
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<T>,
    temp: &mut BooleanArray,
) where
    T: PartialOrdExt,
    PrimitiveArray<T>: Array<Element = T>,
{
    <T as PartialOrdExt>::le(selection, lhs, rhs, temp)
}

#[cfg(debug_assertions)]
fn check_timestamp_array_and_multiplier<const MULTIPLIER: i64>(array: &PrimitiveArray<i64>) {
    use crate::types::{LogicalType, TimeUnit};
    match array.logical_type() {
        LogicalType::Timestamp(unit) => {
            match unit {
                TimeUnit::Second => {
                    if MULTIPLIER != 1 || MULTIPLIER != 1_000 || MULTIPLIER != 1_000_000 || MULTIPLIER != 1_000_000_000{
                        panic!("Timestamp::Second, the multiplier can only be `1`/`1_000`/`1_000_000`/`1_000_000_000`. However, the passed multiplier is `{}`", MULTIPLIER)
                    }
                }
                TimeUnit::Millisecond => {
                    if MULTIPLIER != 1 || MULTIPLIER != 1_000 || MULTIPLIER != 1_000_000{
                        panic!("Timestamp::Millisecond, the multiplier can only be `1`/`1_000`/`1_000_000`. However, the passed multiplier is `{}`", MULTIPLIER)
                    }
                }
                TimeUnit::Microsecond => {
                    if MULTIPLIER != 1 || MULTIPLIER != 1_000{
                        panic!("Timestamp::Microsecond, the multiplier can only be `1`/`1_000`. However, the passed multiplier is `{}`", MULTIPLIER)
                    }
                }
                TimeUnit::Nanosecond => {
                    if MULTIPLIER != 1{
                        panic!("Timestamp::Nanosecond, the multiplier can only be `1`. However, the passed multiplier is `{}`", MULTIPLIER)
                    }
                }
            }
        }
        LogicalType::Timestamptz { .. } => {
            if MULTIPLIER != 1 || MULTIPLIER != 1_000 {
                panic!("Timestamptz is stored in microsecond, the multiplier can only be `1`/`1000`. However, the passed multiplier is `{}`", MULTIPLIER)
            }
        }
        ty  => panic!("timestamp_cmp_scalar function should only be called on Int64Array that has logical type `Timestamp` or `Timestamptz`, called on array that has logical type: {:?}", ty),
    }
}
