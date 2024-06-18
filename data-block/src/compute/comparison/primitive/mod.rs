//! Comparison between primitive arrays
//!
//! TODO: Separate intrinsic with others

use crate::array::primitive::PrimitiveType;
use crate::array::{Array, BooleanArray, PrimitiveArray};
use crate::bitmap::Bitmap;
use crate::compute::logical::and_inplace;

mod private {
    use crate::array::PrimitiveType;

    #[inline]
    pub(super) fn eq<T: PrimitiveType + PartialEq>(lhs: T, rhs: T) -> bool {
        lhs == rhs
    }

    #[inline]
    pub(super) fn ne<T: PrimitiveType + PartialEq>(lhs: T, rhs: T) -> bool {
        lhs != rhs
    }

    #[inline]
    pub(super) fn gt<T: PrimitiveType + PartialOrd>(lhs: T, rhs: T) -> bool {
        lhs > rhs
    }

    #[inline]
    pub(super) fn ge<T: PrimitiveType + PartialOrd>(lhs: T, rhs: T) -> bool {
        lhs >= rhs
    }

    #[inline]
    pub(super) fn lt<T: PrimitiveType + PartialOrd>(lhs: T, rhs: T) -> bool {
        lhs < rhs
    }

    #[inline]
    pub(super) fn le<T: PrimitiveType + PartialOrd>(lhs: T, rhs: T) -> bool {
        lhs <= rhs
    }
}

#[cfg(test)]
macro_rules! cmp_assert {
    ($lhs:expr, $rhs:expr, $cmp_func:path, $gt:expr) => {
        unsafe {
            let len = $lhs.len();
            let mut dst = Bitmap::with_capacity(len);
            {
                let mut guard = dst.mutate();
                let uninitialized = guard.clear_and_resize(len);
                $cmp_func($lhs, $rhs, uninitialized.as_mut_ptr());
            }

            assert_eq!(dst, Bitmap::from_slice_and_len($gt, len));
        }
    };
}

pub mod intrinsic;

/// Default implementation
unsafe fn cmp_scalar_default<T>(
    selection: &mut Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    temp: &mut BooleanArray,
    partial_cmp_threshold: f64,
    cmp_scalars_func: impl Fn(T, T) -> bool,
) where
    PrimitiveArray<T>: Array<Element = T>,
    T: PrimitiveType,
{
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

/// Default implementation
unsafe fn cmp_default<T>(
    selection: &mut Bitmap,
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<T>,
    temp: &mut BooleanArray,
    partial_cmp_threshold: f64,
    cmp_scalars_func: impl Fn(T, T) -> bool,
) where
    PrimitiveArray<T>: Array<Element = T>,
    T: PrimitiveType,
{
    debug_assert_selection_is_valid!(selection, lhs);
    debug_assert_eq!(lhs.len(), rhs.len());

    and_inplace(selection, lhs.validity());
    and_inplace(selection, rhs.validity());
    if selection.ones_ratio() < partial_cmp_threshold {
        selection.mutate().mutate_ones(|index| {
            cmp_scalars_func(
                lhs.get_value_unchecked(index),
                rhs.get_value_unchecked(index),
            )
        })
    } else {
        // It may still faster 😊
        temp.data_mut().mutate().reset(
            lhs.len(),
            lhs.values_iter()
                .zip(rhs.values_iter())
                .map(|(lhs, rhs)| cmp_scalars_func(lhs, rhs)),
        );
        and_inplace(selection, temp.data());
    }
}

/// Primitive type but not intrinsic type. Using this trait to avoid intrinsic type using
/// this function
pub trait NonIntrinsicPrimitiveType: PrimitiveType + PartialOrd {
    /// If the selectivity is smaller than this threshold, partial computation is used
    const PARTIAL_CMP_THRESHOLD: f64;
}

impl NonIntrinsicPrimitiveType for i128 {
    const PARTIAL_CMP_THRESHOLD: f64 = 0.0;
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
    PrimitiveArray<T>: Array<Element = T>,
    T: NonIntrinsicPrimitiveType,
{
    cmp_scalar_default(
        selection,
        array,
        scalar,
        dst,
        T::PARTIAL_CMP_THRESHOLD,
        private::eq,
    )
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
    PrimitiveArray<T>: Array<Element = T>,
    T: NonIntrinsicPrimitiveType,
{
    cmp_scalar_default(
        selection,
        array,
        scalar,
        dst,
        T::PARTIAL_CMP_THRESHOLD,
        private::ne,
    )
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
    PrimitiveArray<T>: Array<Element = T>,
    T: NonIntrinsicPrimitiveType,
{
    cmp_scalar_default(
        selection,
        array,
        scalar,
        dst,
        T::PARTIAL_CMP_THRESHOLD,
        private::gt,
    )
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
    PrimitiveArray<T>: Array<Element = T>,
    T: NonIntrinsicPrimitiveType,
{
    cmp_scalar_default(
        selection,
        array,
        scalar,
        dst,
        T::PARTIAL_CMP_THRESHOLD,
        private::ge,
    )
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
    PrimitiveArray<T>: Array<Element = T>,
    T: NonIntrinsicPrimitiveType,
{
    cmp_scalar_default(
        selection,
        array,
        scalar,
        dst,
        T::PARTIAL_CMP_THRESHOLD,
        private::lt,
    )
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
    PrimitiveArray<T>: Array<Element = T>,
    T: NonIntrinsicPrimitiveType,
{
    cmp_scalar_default(
        selection,
        array,
        scalar,
        dst,
        T::PARTIAL_CMP_THRESHOLD,
        private::le,
    )
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
    dst: &mut BooleanArray,
) where
    PrimitiveArray<T>: Array<Element = T>,
    T: NonIntrinsicPrimitiveType,
{
    cmp_default(
        selection,
        lhs,
        rhs,
        dst,
        T::PARTIAL_CMP_THRESHOLD,
        private::eq,
    )
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
    dst: &mut BooleanArray,
) where
    PrimitiveArray<T>: Array<Element = T>,
    T: NonIntrinsicPrimitiveType,
{
    cmp_default(
        selection,
        lhs,
        rhs,
        dst,
        T::PARTIAL_CMP_THRESHOLD,
        private::ne,
    )
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
    dst: &mut BooleanArray,
) where
    PrimitiveArray<T>: Array<Element = T>,
    T: NonIntrinsicPrimitiveType,
{
    cmp_default(
        selection,
        lhs,
        rhs,
        dst,
        T::PARTIAL_CMP_THRESHOLD,
        private::gt,
    )
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
    dst: &mut BooleanArray,
) where
    PrimitiveArray<T>: Array<Element = T>,
    T: NonIntrinsicPrimitiveType,
{
    cmp_default(
        selection,
        lhs,
        rhs,
        dst,
        T::PARTIAL_CMP_THRESHOLD,
        private::ge,
    )
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
    dst: &mut BooleanArray,
) where
    PrimitiveArray<T>: Array<Element = T>,
    T: NonIntrinsicPrimitiveType,
{
    cmp_default(
        selection,
        lhs,
        rhs,
        dst,
        T::PARTIAL_CMP_THRESHOLD,
        private::lt,
    )
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
    dst: &mut BooleanArray,
) where
    PrimitiveArray<T>: Array<Element = T>,
    T: NonIntrinsicPrimitiveType,
{
    cmp_default(
        selection,
        lhs,
        rhs,
        dst,
        T::PARTIAL_CMP_THRESHOLD,
        private::le,
    )
}
