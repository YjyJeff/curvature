//! Comparison between boolean arrays

use crate::array::{Array, BooleanArray};
use crate::bitmap::Bitmap;
use crate::compute::logical::{and_inplace, and_not_inplace};

/// Perform `array == scalar` between [`BooleanArray`] and `bool`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn eq_scalar(selection: &mut Bitmap, array: &BooleanArray, scalar: bool) {
    and_inplace(selection, array.validity());
    if scalar {
        and_inplace(selection, array.data());
    } else {
        and_not_inplace(selection, array.data(), array.len())
    }
}

/// Perform `array != scalar` between [`BooleanArray`] and `bool`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn ne_scalar(selection: &mut Bitmap, array: &BooleanArray, scalar: bool) {
    and_inplace(selection, array.validity());

    if !scalar {
        and_inplace(selection, array.data());
    } else {
        and_not_inplace(selection, array.data(), array.len())
    }
}

/// Perform `array > scalar` between [`BooleanArray`] and `bool`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn gt_scalar(selection: &mut Bitmap, array: &BooleanArray, scalar: bool) {
    and_inplace(selection, array.validity());

    if !scalar {
        and_inplace(selection, array.data());
    } else {
        // Compiler will optimize it to memset
        selection
            .mutate()
            .clear_and_resize(array.len())
            .iter_mut()
            .for_each(|v| *v = 0);
    }
}

/// Perform `array > scalar` between [`BooleanArray`] and `bool`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn ge_scalar(selection: &mut Bitmap, array: &BooleanArray, scalar: bool) {
    and_inplace(selection, array.validity());
    if scalar {
        and_inplace(selection, array.data())
    }
}

/// Perform `array < scalar` between [`BooleanArray`] and `bool`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn lt_scalar(selection: &mut Bitmap, array: &BooleanArray, scalar: bool) {
    and_inplace(selection, array.validity());

    if scalar {
        and_not_inplace(selection, array.data(), array.len());
    } else {
        // Compiler will optimize it to memset
        selection
            .mutate()
            .clear_and_resize(array.len())
            .iter_mut()
            .for_each(|v| *v = 0);
    }
}

/// Perform `array < scalar` between [`BooleanArray`] and `bool`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn le_scalar(selection: &mut Bitmap, array: &BooleanArray, scalar: bool) {
    and_inplace(selection, array.validity());

    if !scalar {
        and_not_inplace(selection, array.data(), array.len());
    }
}

#[cfg(test)]
mod tests {

    use crate::bitmap::Bitmap;

    use super::*;

    fn test_reverse(dst: &Bitmap, lhs: &Bitmap) {
        assert_eq!(
            dst.iter().map(|v| !v).collect::<Vec<_>>(),
            lhs.iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_equal() {
        let lhs = BooleanArray::new_with_data(Bitmap::from_slice_and_len(&[0xf371], 16));
        let mut selection = Bitmap::new();

        unsafe { eq_scalar(&mut selection, &lhs, true) };
        assert_eq!(&selection, lhs.data());

        let lhs = BooleanArray::new_with_data(Bitmap::from_slice_and_len(&[0xf371], 40));

        selection.mutate().clear();
        unsafe { eq_scalar(&mut selection, &lhs, false) };
        test_reverse(&selection, lhs.data());

        selection.mutate().clear();
        unsafe { ne_scalar(&mut selection, &lhs, true) };
        test_reverse(&selection, lhs.data());

        selection.mutate().clear();
        unsafe { ne_scalar(&mut selection, &lhs, false) };
        assert_eq!(&selection, lhs.data());
    }

    #[test]
    fn test_order() {
        let len = 100;
        let lhs = BooleanArray::new_with_data(Bitmap::from_slice_and_len(&[0x52e3, 0xd2c8], len));
        let mut selection = Bitmap::new();

        unsafe { gt_scalar(&mut selection, &lhs, false) };
        assert_eq!(&selection, lhs.data());

        selection.mutate().clear();
        unsafe { gt_scalar(&mut selection, &lhs, true) };
        assert_eq!(selection.count_zeros(), len);

        selection.mutate().clear();
        unsafe { ge_scalar(&mut selection, &lhs, false) };
        assert!(selection.all_valid());

        selection.mutate().clear();
        unsafe { ge_scalar(&mut selection, &lhs, true) };
        assert_eq!(&selection, lhs.data());

        selection.mutate().clear();
        unsafe { lt_scalar(&mut selection, &lhs, false) };
        assert_eq!(selection.count_zeros(), len);

        selection.mutate().clear();
        unsafe { lt_scalar(&mut selection, &lhs, true) };
        test_reverse(&selection, lhs.data());

        selection.mutate().clear();
        unsafe { le_scalar(&mut selection, &lhs, false) };
        test_reverse(&selection, &lhs.data);

        selection.mutate().clear();
        unsafe { le_scalar(&mut selection, &lhs, true) };
        assert!(selection.all_valid());
    }
}
