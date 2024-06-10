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
#[cfg(not(feature = "portable_simd"))]
mod stable;

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
    /// - `array`'s data and validity should not reference `dst`'s data and validity. In the computation
    /// graph, `lhs` must be the descendant of `dst`
    ///
    /// - No other arrays that reference the `dst`'s data and validity are accessed! In the
    /// computation graph, it will never happens
    unsafe fn eq_scalar(
        selection: &mut Bitmap,
        array: &PrimitiveArray<Self>,
        scalar: Self,
        dst: &mut BooleanArray,
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
    /// - `array`'s data and validity should not reference `dst`'s data and validity. In the computation
    /// graph, `lhs` must be the descendant of `dst`
    ///
    /// - No other arrays that reference the `dst`'s data and validity are accessed! In the
    /// computation graph, it will never happens
    unsafe fn ne_scalar(
        selection: &mut Bitmap,
        array: &PrimitiveArray<Self>,
        scalar: Self,
        dst: &mut BooleanArray,
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
    /// - `array`'s data and validity should not reference `dst`'s data and validity. In the computation
    /// graph, `lhs` must be the descendant of `dst`
    ///
    /// - No other arrays that reference the `dst`'s data and validity are accessed! In the
    /// computation graph, it will never happens
    unsafe fn gt_scalar(
        selection: &mut Bitmap,
        array: &PrimitiveArray<Self>,
        scalar: Self,
        dst: &mut BooleanArray,
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
    /// - `array`'s data and validity should not reference `dst`'s data and validity. In the computation
    /// graph, `lhs` must be the descendant of `dst`
    ///
    /// - No other arrays that reference the `dst`'s data and validity are accessed! In the
    /// computation graph, it will never happens
    unsafe fn ge_scalar(
        selection: &mut Bitmap,
        array: &PrimitiveArray<Self>,
        scalar: Self,
        dst: &mut BooleanArray,
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
    /// - `array`'s data and validity should not reference `dst`'s data and validity. In the computation
    /// graph, `lhs` must be the descendant of `dst`
    ///
    /// - No other arrays that reference the `dst`'s data and validity are accessed! In the
    /// computation graph, it will never happens
    unsafe fn lt_scalar(
        selection: &mut Bitmap,
        array: &PrimitiveArray<Self>,
        scalar: Self,
        dst: &mut BooleanArray,
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
    /// - `array`'s data and validity should not reference `dst`'s data and validity. In the computation
    /// graph, `lhs` must be the descendant of `dst`
    ///
    /// - No other arrays that reference the `dst`'s data and validity are accessed! In the
    /// computation graph, it will never happens
    unsafe fn le_scalar(
        selection: &mut Bitmap,
        array: &PrimitiveArray<Self>,
        scalar: Self,
        dst: &mut BooleanArray,
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
/// - `array`'s data and validity should not reference `dst`'s data and validity. In the computation
/// graph, `lhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn eq_scalar<T>(
    selection: &mut Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    dst: &mut BooleanArray,
) where
    T: PartialOrdExt,
    PrimitiveArray<T>: Array<Element = T>,
{
    T::eq_scalar(selection, array, scalar, dst)
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
/// - `array`'s data and validity should not reference `dst`'s data and validity. In the computation
/// graph, `lhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn ne_scalar<T>(
    selection: &mut Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    dst: &mut BooleanArray,
) where
    T: PartialOrdExt,
    PrimitiveArray<T>: Array<Element = T>,
{
    T::ne_scalar(selection, array, scalar, dst)
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
/// - `array`'s data and validity should not reference `dst`'s data and validity. In the computation
/// graph, `lhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn gt_scalar<T>(
    selection: &mut Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    dst: &mut BooleanArray,
) where
    T: PartialOrdExt,
    PrimitiveArray<T>: Array<Element = T>,
{
    T::gt_scalar(selection, array, scalar, dst)
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
/// - `array`'s data and validity should not reference `dst`'s data and validity. In the computation
/// graph, `lhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn ge_scalar<T>(
    selection: &mut Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    dst: &mut BooleanArray,
) where
    T: PartialOrdExt,
    PrimitiveArray<T>: Array<Element = T>,
{
    T::ge_scalar(selection, array, scalar, dst)
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
/// - `array`'s data and validity should not reference `dst`'s data and validity. In the computation
/// graph, `lhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn lt_scalar<T>(
    selection: &mut Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    dst: &mut BooleanArray,
) where
    T: PartialOrdExt,
    PrimitiveArray<T>: Array<Element = T>,
{
    T::lt_scalar(selection, array, scalar, dst)
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
/// - `array`'s data and validity should not reference `dst`'s data and validity. In the computation
/// graph, `lhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn le_scalar<T>(
    selection: &mut Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    dst: &mut BooleanArray,
) where
    T: PartialOrdExt,
    PrimitiveArray<T>: Array<Element = T>,
{
    T::le_scalar(selection, array, scalar, dst)
}
