//! filter out elements that are not selected

use crate::array::Array;
use crate::bitmap::Bitmap;

/// Filter out elements that are not selected
///
/// # Safety
///
/// - `array` and `selection` should have same length. Otherwise, undefined behavior happens
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
///   computation graph, it will never happens
pub unsafe fn filter<A: Array>(selection: &Bitmap, array: &A, dst: &mut A) {
    #[cfg(feature = "verify")]
    assert_eq!(selection.len(), array.len());

    let validity = array.validity();

    // Array and selection has same length, selection is not empty
    let count_ones = selection.count_ones_unchecked();

    if validity.all_valid() {
        dst.validity_mut().mutate().clear();
        dst.replace_with_trusted_len_values_ref_iterator(
            count_ones,
            selection
                .iter_ones()
                .map(|index| array.get_value_unchecked(index)),
        );
    } else {
        dst.replace_with_trusted_len_ref_iterator(
            count_ones,
            selection.iter_ones().map(|index| {
                if validity.get_unchecked(index) {
                    Some(array.get_value_unchecked(index))
                } else {
                    None
                }
            }),
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::array::Int32Array;

    use super::*;

    #[test]
    fn test_filter() {
        let array = Int32Array::from_iter([Some(10), None, Some(-10), Some(-1), None]);
        let mut dst = Int32Array::default();
        let selection = Bitmap::from_slice_and_len(&[0b01011], 5);

        unsafe { filter(&selection, &array, &mut dst) };
        assert_eq!(dst.iter().collect::<Vec<_>>(), [Some(10), None, Some(-1)]);

        let array = Int32Array::from_values_iter([-1, -2, -3, 1, 2, 3]);
        let selection = Bitmap::from_slice_and_len(&[0b101011], 6);
        unsafe { filter(&selection, &array, &mut dst) };
        assert_eq!(
            dst.iter().collect::<Vec<_>>(),
            [Some(-1), Some(-2), Some(1), Some(3)]
        );
    }
}
