//! Compare string array

use crate::array::{Array, StringArray};
use crate::bitmap::Bitmap;
use crate::compute::logical::and_inplace;
use crate::element::string::StringView;

#[inline]
unsafe fn cmp_scalar<F>(
    selection: &mut Bitmap,
    array: &StringArray,
    scalar: StringView<'_>,
    cmp_func: F,
) where
    for<'a, 'b> F: Fn(StringView<'a>, StringView<'b>) -> bool,
{
    debug_assert_selection_is_valid!(selection, array);

    let validity = array.validity();

    // FIXME: Tune the hyper parameter and fast path condition
    if selection.all_valid() && validity.ones_ratio() >= 0.8 {
        // This branch is faster or not totally depends on data distribution.
        // If the order can be determined by the prefix, it is faster even if
        // we do some unnecessary computation
        selection.mutate().reset(
            array.len(),
            array.values_iter().map(|element| cmp_func(element, scalar)),
        );
        and_inplace(selection, validity);
    } else {
        and_inplace(selection, validity);
        selection
            .mutate()
            .mutate_ones(|index| cmp_func(array.get_value_unchecked(index), scalar));
    }
}

unsafe fn cmp<F>(selection: &mut Bitmap, lhs: &StringArray, rhs: &StringArray, cmp_func: F)
where
    for<'a, 'b> F: Fn(StringView<'a>, StringView<'b>) -> bool,
{
    debug_assert_selection_is_valid!(selection, lhs);
    debug_assert_eq!(lhs.len(), rhs.len());

    // TBD: Fast path?
    if selection.all_valid() && lhs.validity().all_valid() && rhs.validity.all_valid() {
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

/// Perform `array == scalar` between an [`StringArray`] and [`StringView`]
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn eq_scalar(selection: &mut Bitmap, array: &StringArray, scalar: StringView<'_>) {
    cmp_scalar(selection, array, scalar, |element, scalar| {
        element == scalar
    })
}

/// Perform `array != scalar` between an [`StringArray`] and [`StringView`]
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn ne_scalar(selection: &mut Bitmap, array: &StringArray, scalar: StringView<'_>) {
    cmp_scalar(selection, array, scalar, |element, scalar| {
        element != scalar
    })
}

/// Perform `array > scalar` between an [`StringArray`] and [`StringView`]
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn gt_scalar(selection: &mut Bitmap, array: &StringArray, scalar: StringView<'_>) {
    cmp_scalar(selection, array, scalar, |element, scalar| element > scalar)
}

/// Perform `array >= scalar` between an [`StringArray`] and [`StringView`]
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn ge_scalar(selection: &mut Bitmap, array: &StringArray, scalar: StringView<'_>) {
    cmp_scalar(selection, array, scalar, |element, scalar| {
        element >= scalar
    })
}

/// Perform `array < scalar` between an [`StringArray`] and [`StringView`]
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn lt_scalar(selection: &mut Bitmap, array: &StringArray, scalar: StringView<'_>) {
    cmp_scalar(selection, array, scalar, |element, scalar| element < scalar)
}

/// Perform `array <= scalar` between an [`StringArray`] and [`StringView`]
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn le_scalar(selection: &mut Bitmap, array: &StringArray, scalar: StringView<'_>) {
    cmp_scalar(selection, array, scalar, |element, scalar| {
        element <= scalar
    })
}

/// Perform `StringArray == StringArray`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn eq(selection: &mut Bitmap, lhs: &StringArray, rhs: &StringArray) {
    cmp(selection, lhs, rhs, |element, scalar| element == scalar)
}

/// Perform `StringArray != StringArray`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn ne(selection: &mut Bitmap, lhs: &StringArray, rhs: &StringArray) {
    cmp(selection, lhs, rhs, |element, scalar| element != scalar)
}

/// Perform `StringArray > StringArray`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn gt(selection: &mut Bitmap, lhs: &StringArray, rhs: &StringArray) {
    cmp(selection, lhs, rhs, |element, scalar| element > scalar)
}

/// Perform `StringArray >= StringArray`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn ge(selection: &mut Bitmap, lhs: &StringArray, rhs: &StringArray) {
    cmp(selection, lhs, rhs, |element, scalar| element >= scalar)
}

/// Perform `StringArray < StringArray`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn lt(selection: &mut Bitmap, lhs: &StringArray, rhs: &StringArray) {
    cmp(selection, lhs, rhs, |element, scalar| element < scalar)
}

/// Perform `StringArray <= StringArray`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn le(selection: &mut Bitmap, lhs: &StringArray, rhs: &StringArray) {
    cmp(selection, lhs, rhs, |element, scalar| element <= scalar)
}
