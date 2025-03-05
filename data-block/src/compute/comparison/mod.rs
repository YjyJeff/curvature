//! Comparison on arrays
//!
//! As we all know, comparison is a boolean expression. Therefore, the comparison
//! result will be combined into the `selection` arguments

pub mod boolean;
pub mod primitive;
pub mod string;

use crate::array::Array;
use crate::bitmap::Bitmap;
use crate::compute::logical::and_inplace;
use crate::element::{Element, ElementRef};

/// A default `cmp(array, scalar)` function. It is not optimal in most of the cases
#[inline]
unsafe fn cmp_scalar<A: Array, F>(
    selection: &mut Bitmap,
    array: &A,
    scalar: <A::Element as Element>::ElementRef<'_>,
    cmp_func: F,
) where
    F: Fn(<A::Element as Element>::ElementRef<'_>, <A::Element as Element>::ElementRef<'_>) -> bool,
{
    unsafe {
        #[cfg(feature = "verify")]
        assert_selection_is_valid!(selection, array);

        let validity = array.validity();

        if validity.all_valid() && selection.all_valid() {
            selection.mutate().reset(
                array.len(),
                array.values_iter().map(|lhs| cmp_func(lhs, scalar)),
            );
        } else {
            and_inplace(selection, validity);
            selection
                .mutate()
                .mutate_ones(|index| cmp_func(array.get_value_unchecked(index), scalar));
        }
    }
}

/// Perform `array == scalar` between an Array and its ElementRef
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
///   Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn eq_scalar<A: Array>(
    selection: &mut Bitmap,
    array: &A,
    scalar: <A::Element as Element>::ElementRef<'_>,
) where
    for<'a, 'b> <A::Element as Element>::ElementRef<'a>:
        PartialEq<<A::Element as Element>::ElementRef<'b>>,
{
    unsafe { cmp_scalar(selection, array, scalar, |lhs, rhs| lhs == rhs) }
}

/// Perform `array != scalar` between an Array and its ElementRef
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
///   Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn ne_scalar<A: Array>(
    selection: &mut Bitmap,
    array: &A,
    scalar: <A::Element as Element>::ElementRef<'_>,
) where
    for<'a, 'b> <A::Element as Element>::ElementRef<'a>:
        PartialEq<<A::Element as Element>::ElementRef<'b>>,
{
    unsafe { cmp_scalar(selection, array, scalar, |lhs, rhs| lhs != rhs) }
}

/// Perform `array > scalar` between an Array and its ElementRef
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
///   Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn gt_scalar<A: Array>(
    selection: &mut Bitmap,
    array: &A,
    scalar: <A::Element as Element>::ElementRef<'_>,
) where
    for<'a, 'b> <A::Element as Element>::ElementRef<'a>:
        PartialOrd<<A::Element as Element>::ElementRef<'b>>,
{
    unsafe { cmp_scalar(selection, array, scalar, |lhs, rhs| lhs > rhs) }
}

/// Perform `array >= scalar` between an Array and its ElementRef
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
///   Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn ge_scalar<A: Array>(
    selection: &mut Bitmap,
    array: &A,
    scalar: <A::Element as Element>::ElementRef<'_>,
) where
    for<'a, 'b> <A::Element as Element>::ElementRef<'a>:
        PartialOrd<<A::Element as Element>::ElementRef<'b>>,
{
    unsafe { cmp_scalar(selection, array, scalar, |lhs, rhs| lhs >= rhs) }
}

/// Perform `array < scalar` between an Array and its ElementRef
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
///   Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn lt_scalar<A: Array>(
    selection: &mut Bitmap,
    array: &A,
    scalar: <A::Element as Element>::ElementRef<'_>,
) where
    for<'a, 'b> <A::Element as Element>::ElementRef<'a>:
        PartialOrd<<A::Element as Element>::ElementRef<'b>>,
{
    unsafe { cmp_scalar(selection, array, scalar, |lhs, rhs| lhs < rhs) }
}

/// Perform `array <= scalar` between an Array and its ElementRef
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
///   Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn le_scalar<A: Array>(
    selection: &mut Bitmap,
    array: &A,
    scalar: <A::Element as Element>::ElementRef<'_>,
) where
    for<'a, 'b> <A::Element as Element>::ElementRef<'a>:
        PartialOrd<<A::Element as Element>::ElementRef<'b>>,
{
    unsafe { cmp_scalar(selection, array, scalar, |lhs, rhs| lhs <= rhs) }
}

/// Returns true if two elements ref are equal
#[inline]
pub fn scalar_eq_scalar<'a, T>(lhs: T, rhs: T) -> bool
where
    T: ElementRef<'a> + PartialEq,
{
    lhs == rhs
}

/// Returns true if two elements ref are not equal
#[inline]
pub fn scalar_ne_scalar<'a, T>(lhs: T, rhs: T) -> bool
where
    T: ElementRef<'a> + PartialEq,
{
    lhs != rhs
}

/// Returns true if left element ref greater than right element ref
#[inline]
pub fn scalar_gt_scalar<'a, T>(lhs: T, rhs: T) -> bool
where
    T: ElementRef<'a> + PartialOrd,
{
    lhs > rhs
}

/// Returns true if left element ref greater than or equal to right element ref
#[inline]
pub fn scalar_ge_scalar<'a, T>(lhs: T, rhs: T) -> bool
where
    T: ElementRef<'a> + PartialOrd,
{
    lhs >= rhs
}

/// Returns true if left element ref less than right element ref
#[inline]
pub fn scalar_lt_scalar<'a, T>(lhs: T, rhs: T) -> bool
where
    T: ElementRef<'a> + PartialOrd,
{
    lhs < rhs
}

/// Returns true if left element ref less than or equal to right element ref
#[inline]
pub fn scalar_le_scalar<'a, T>(lhs: T, rhs: T) -> bool
where
    T: ElementRef<'a> + PartialOrd,
{
    lhs <= rhs
}

/// Returns true if left timestamp equal to right timestamp
#[inline]
pub fn timestamp_scalar_eq_scalar<const LM: i64, const RM: i64>(lhs: i64, rhs: i64) -> bool {
    lhs * LM == rhs * RM
}

/// Returns true if left timestamp not equal to right timestamp
#[inline]
pub fn timestamp_scalar_ne_scalar<const LM: i64, const RM: i64>(lhs: i64, rhs: i64) -> bool {
    lhs * LM != rhs * RM
}

/// Returns true if left timestamp greater than right timestamp
#[inline]
pub fn timestamp_scalar_gt_scalar<const LM: i64, const RM: i64>(lhs: i64, rhs: i64) -> bool {
    lhs * LM > rhs * RM
}

/// Returns true if left timestamp greater than or equal to right timestamp
#[inline]
pub fn timestamp_scalar_ge_scalar<const LM: i64, const RM: i64>(lhs: i64, rhs: i64) -> bool {
    lhs * LM >= rhs * RM
}

/// Returns true if left timestamp less than right timestamp
#[inline]
pub fn timestamp_scalar_lt_scalar<const LM: i64, const RM: i64>(lhs: i64, rhs: i64) -> bool {
    lhs * LM < rhs * RM
}

/// Returns true if left timestamp less than or equal to right timestamp
#[inline]
pub fn timestamp_scalar_le_scalar<const LM: i64, const RM: i64>(lhs: i64, rhs: i64) -> bool {
    lhs * LM <= rhs * RM
}

/// A default `cmp(array, array)` function. It is not optimal in most of the cases
#[inline]
unsafe fn cmp<A: Array, F>(selection: &mut Bitmap, lhs: &A, rhs: &A, cmp_func: F)
where
    F: Fn(<A::Element as Element>::ElementRef<'_>, <A::Element as Element>::ElementRef<'_>) -> bool,
{
    unsafe {
        #[cfg(feature = "verify")]
        {
            assert_selection_is_valid!(selection, lhs);
            assert_eq!(lhs.len(), rhs.len());
        }

        if selection.all_valid() && lhs.validity().all_valid() && rhs.validity().all_valid() {
            selection.mutate().reset(
                lhs.len(),
                lhs.values_iter()
                    .zip(rhs.values_iter())
                    .map(|(lhs, rhs)| cmp_func(lhs, rhs)),
            );
        } else {
            and_inplace(selection, lhs.validity());
            and_inplace(selection, rhs.validity());
            selection.mutate().mutate_ones(|index| {
                cmp_func(
                    lhs.get_value_unchecked(index),
                    rhs.get_value_unchecked(index),
                )
            });
        }
    }
}

/// Perform `array == array`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
///   Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn eq<A: Array>(selection: &mut Bitmap, lhs: &A, rhs: &A)
where
    for<'a, 'b> <A::Element as Element>::ElementRef<'a>:
        PartialEq<<A::Element as Element>::ElementRef<'b>>,
{
    unsafe { cmp(selection, lhs, rhs, |lhs, rhs| lhs == rhs) }
}

/// Perform `array != array`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
///   Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn ne<A: Array>(selection: &mut Bitmap, lhs: &A, rhs: &A)
where
    for<'a, 'b> <A::Element as Element>::ElementRef<'a>:
        PartialEq<<A::Element as Element>::ElementRef<'b>>,
{
    unsafe { cmp(selection, lhs, rhs, |lhs, rhs| lhs != rhs) }
}

/// Perform `array > array`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
///   Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn gt<A: Array>(selection: &mut Bitmap, lhs: &A, rhs: &A)
where
    for<'a, 'b> <A::Element as Element>::ElementRef<'a>:
        PartialOrd<<A::Element as Element>::ElementRef<'b>>,
{
    unsafe { cmp(selection, lhs, rhs, |lhs, rhs| lhs > rhs) }
}

/// Perform `array >= array`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
///   Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn ge<A: Array>(selection: &mut Bitmap, lhs: &A, rhs: &A)
where
    for<'a, 'b> <A::Element as Element>::ElementRef<'a>:
        PartialOrd<<A::Element as Element>::ElementRef<'b>>,
{
    unsafe { cmp(selection, lhs, rhs, |lhs, rhs| lhs >= rhs) }
}

/// Perform `array < array`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
///   Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn lt<A: Array>(selection: &mut Bitmap, lhs: &A, rhs: &A)
where
    for<'a, 'b> <A::Element as Element>::ElementRef<'a>:
        PartialOrd<<A::Element as Element>::ElementRef<'b>>,
{
    unsafe { cmp(selection, lhs, rhs, |lhs, rhs| lhs < rhs) }
}

/// Perform `array <= array`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
///   Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn le<A: Array>(selection: &mut Bitmap, lhs: &A, rhs: &A)
where
    for<'a, 'b> <A::Element as Element>::ElementRef<'a>:
        PartialOrd<<A::Element as Element>::ElementRef<'b>>,
{
    unsafe { cmp(selection, lhs, rhs, |lhs, rhs| lhs <= rhs) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::StringArray;
    use crate::element::string::StringView;

    #[test]
    fn test_eq_scalar() {
        let lhs = [
            "curvature",
            "curve",
            "curse",
            "tcp",
            "http",
            "auto-vectorization",
            "tcp",
            "three body is pretty awesome",
        ];
        let array = StringArray::from_values_iter(lhs);
        // All selected, all valid, inlined
        let mut selection = Bitmap::from_slice_and_len(&[u64::MAX], 8);
        unsafe {
            eq_scalar(&mut selection, &array, StringView::from_static_str("tcp"));
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [3, 6]);
        let mut selection = Bitmap::from_slice_and_len(&[u64::MAX], 8);
        unsafe {
            ne_scalar(&mut selection, &array, StringView::from_static_str("tcp"));
        }
        assert_eq!(
            selection.iter_ones().collect::<Vec<_>>(),
            [0, 1, 2, 4, 5, 7]
        );

        // All selected, all valid, not-inlined
        let mut selection = Bitmap::from_slice_and_len(&[u64::MAX], 8);
        unsafe {
            eq_scalar(
                &mut selection,
                &array,
                StringView::from_static_str("auto-vectorization"),
            );
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [5]);
        let mut selection = Bitmap::from_slice_and_len(&[u64::MAX], 8);
        unsafe {
            ne_scalar(
                &mut selection,
                &array,
                StringView::from_static_str("auto-vectorization"),
            );
        }
        assert_eq!(
            selection.iter_ones().collect::<Vec<_>>(),
            [0, 1, 2, 3, 4, 6, 7]
        );

        // Partial selected, all valid, inlined
        let mut selection = Bitmap::from_slice_and_len(&[0xf7], 8);
        unsafe {
            eq_scalar(&mut selection, &array, StringView::from_static_str("tcp"));
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [6]);
        let mut selection = Bitmap::from_slice_and_len(&[0xbe], 8);
        unsafe {
            ne_scalar(&mut selection, &array, StringView::from_static_str("tcp"));
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [1, 2, 4, 5, 7]);

        // Partial selected, all valid, not-inlined
        let mut selection = Bitmap::from_slice_and_len(&[0x31], 8);
        unsafe {
            eq_scalar(
                &mut selection,
                &array,
                StringView::from_static_str("auto-vectorization"),
            );
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [5]);
        let mut selection = Bitmap::from_slice_and_len(&[0x20], 8);
        unsafe {
            ne_scalar(
                &mut selection,
                &array,
                StringView::from_static_str("auto-vectorization"),
            );
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), []);

        let lhs = [
            Some("curvature"),
            None,
            Some("curse"),
            Some(""),
            Some("http"),
            Some("auto-vectorization"),
            Some(""),
            None,
        ];
        let array = StringArray::from_iter(lhs);

        // All selected, partial valid, inlined
        let mut selection = Bitmap::from_slice_and_len(&[u64::MAX], 8);
        unsafe {
            eq_scalar(&mut selection, &array, StringView::from_static_str(""));
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [3, 6]);
        let mut selection = Bitmap::from_slice_and_len(&[u64::MAX], 8);
        unsafe {
            ne_scalar(&mut selection, &array, StringView::from_static_str(""));
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [0, 2, 4, 5]);

        // All selected, partial valid, not-inlined
        let mut selection = Bitmap::from_slice_and_len(&[u64::MAX], 8);
        unsafe {
            eq_scalar(
                &mut selection,
                &array,
                StringView::from_static_str("auto-vectorization"),
            );
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [5]);

        let mut selection = Bitmap::from_slice_and_len(&[u64::MAX], 8);
        unsafe {
            ne_scalar(
                &mut selection,
                &array,
                StringView::from_static_str("auto-vectorization"),
            );
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [0, 2, 3, 4, 6,]);

        // Partial selected, partial valid, inlined
        let mut selection = Bitmap::from_slice_and_len(&[0xf7], 8);
        unsafe {
            eq_scalar(&mut selection, &array, StringView::from_static_str(""));
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [6]);
        let mut selection = Bitmap::from_slice_and_len(&[0xf7], 8);
        unsafe {
            ne_scalar(&mut selection, &array, StringView::from_static_str(""));
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [0, 2, 4, 5,]);

        // Partial selected, partial valid, not-inlined
        let mut selection = Bitmap::from_slice_and_len(&[0x31], 8);
        unsafe {
            eq_scalar(
                &mut selection,
                &array,
                StringView::from_static_str("auto-vectorization"),
            );
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [5]);
        let mut selection = Bitmap::from_slice_and_len(&[0x31], 8);
        unsafe {
            ne_scalar(
                &mut selection,
                &array,
                StringView::from_static_str("auto-vectorization"),
            );
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [0, 4]);
    }
}
