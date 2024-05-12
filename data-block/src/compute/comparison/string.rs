//! Compare string array

use crate::array::{Array, BooleanArray, StringArray};
use crate::bitmap::Bitmap;
use crate::element::string::StringView;

macro_rules! impl_ord_scalar {
    ($func:ident, $op:tt) => {
        #[doc = concat!(" Perform `lhs ", stringify!($op), " rhs` between a StringArray and a StringView")]
        ///
        /// # Safety
        ///
        /// - `lhs`'s validity should not reference `dst`'s validity. In the computation graph,
        /// `lhs` must be the descendant of `dst`
        ///
        /// - No other arrays that reference the `dst`'s data and validity are accessed! In the
        /// computation graph, it will never happens
        pub unsafe fn $func(lhs: &StringArray, rhs: StringView<'_>, dst: &mut BooleanArray) {
            dst.validity.reference(&lhs.validity);

            let mut bitmap = dst.data.as_mut().mutate();
            bitmap.reset(lhs.len(), lhs.values_iter().map(|lhs| lhs $op rhs));
        }
    };
}

impl_ord_scalar!(eq_scalar, ==);
impl_ord_scalar!(ne_scalar, !=);
impl_ord_scalar!(gt_scalar, >);
impl_ord_scalar!(ge_scalar, >=);
impl_ord_scalar!(lt_scalar, <);
impl_ord_scalar!(le_scalar, <=);

macro_rules! impl_selected_ord_scalar {
    ($func:ident, $op:tt) => {
        #[doc = concat!(" Perform `lhs ", stringify!($op), " rhs` between a StringArray with given selection array and a StringView")]
        /// # Safety
        ///
        /// - `array` and `selection` should have same length. Otherwise, undefined behavior happens
        ///
        /// - `selection` should not be referenced by any array
        pub unsafe fn $func(selection: &mut Bitmap, array: &StringArray, scalar: StringView<'_>) {
            debug_assert_eq!(selection.len(), array.len());

            let selection_all_valid = selection.all_valid();
            let mut guard = selection.mutate();
            let validity = array.validity();

            if selection_all_valid {
                if validity.all_valid() {
                    // All selected, all valid
                    guard.reset(array.len(), array.values_iter().map(|lhs| lhs $op scalar));
                } else {
                    // All selected, partial valid
                    guard.reset(
                        array.len(),
                        array
                            .values_iter()
                            .zip(validity.iter())
                            .map(|(view, valid)| valid && view $op scalar),
                    )
                }
            } else if validity.all_valid() {
                // Partial selected, all valid
                guard.mutate_ones(|index| array.get_value_unchecked(index) $op scalar)
            } else {
                // Partial selected, partial valid
                guard.mutate_ones(|index| {
                    validity.get_unchecked(index) && array.get_value_unchecked(index) $op scalar
                })
            }
        }
    };
}

impl_selected_ord_scalar!(selected_eq_scalar, ==);
impl_selected_ord_scalar!(selected_ne_scalar, !=);
impl_selected_ord_scalar!(selected_gt_scalar, >);
impl_selected_ord_scalar!(selected_ge_scalar, >=);
impl_selected_ord_scalar!(selected_lt_scalar, <);
impl_selected_ord_scalar!(selected_le_scalar, <=);

#[cfg(test)]
mod tests {
    use crate::array::{BooleanArray, StringArray};
    use crate::types::LogicalType;

    use super::*;

    macro_rules! test_cmp_scalar {
        ($func:ident, $lhs:expr, $rhs:expr, $ground_truth:expr) => {
            let lhs_array = StringArray::from_values_iter($lhs);
            let mut dst = BooleanArray::new(LogicalType::Boolean).unwrap();
            unsafe { $func(&lhs_array, $rhs, &mut dst) };
            assert_eq!(dst.data.iter().collect::<Vec<_>>(), $ground_truth);
        };
    }

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
        // inline
        test_cmp_scalar!(
            eq_scalar,
            lhs,
            StringView::from_static_str("tcp"),
            [false, false, false, true, false, false, true, false]
        );

        // not inlined
        test_cmp_scalar!(
            eq_scalar,
            lhs,
            StringView::from_static_str("auto-vectorization"),
            [false, false, false, false, false, true, false, false]
        );

        // inline
        test_cmp_scalar!(
            ne_scalar,
            lhs,
            StringView::from_static_str("tcp"),
            [true, true, true, false, true, true, false, true]
        );

        // not inlined
        test_cmp_scalar!(
            ne_scalar,
            lhs,
            StringView::from_static_str("auto-vectorization"),
            [true, true, true, true, true, false, true, true]
        );
    }

    #[test]
    fn test_gt_scalar() {
        test_cmp_scalar!(
            gt_scalar,
            ["curvature", "curve", "curse"],
            StringView::from_static_str("curvature"),
            [false, true, false]
        );
    }

    #[test]
    fn test_ge_scalar() {
        test_cmp_scalar!(
            ge_scalar,
            ["curvature", "curve", "curse"],
            StringView::from_static_str("curvature"),
            [true, true, false]
        );
    }

    #[test]
    fn test_lt_scalar() {
        test_cmp_scalar!(
            lt_scalar,
            ["curvature", "curve", "curse"],
            StringView::from_static_str("curvature"),
            [false, false, true]
        );
    }

    #[test]
    fn test_le_scalar() {
        test_cmp_scalar!(
            le_scalar,
            ["curvature", "curve", "curse"],
            StringView::from_static_str("curvature"),
            [true, false, true]
        );
    }

    #[test]
    fn test_selected_eq_scalar() {
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
            selected_eq_scalar(&mut selection, &array, StringView::from_static_str("tcp"));
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [3, 6]);
        let mut selection = Bitmap::from_slice_and_len(&[u64::MAX], 8);
        unsafe {
            selected_ne_scalar(&mut selection, &array, StringView::from_static_str("tcp"));
        }
        assert_eq!(
            selection.iter_ones().collect::<Vec<_>>(),
            [0, 1, 2, 4, 5, 7]
        );

        // All selected, all valid, not-inlined
        let mut selection = Bitmap::from_slice_and_len(&[u64::MAX], 8);
        unsafe {
            selected_eq_scalar(
                &mut selection,
                &array,
                StringView::from_static_str("auto-vectorization"),
            );
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [5]);
        let mut selection = Bitmap::from_slice_and_len(&[u64::MAX], 8);
        unsafe {
            selected_ne_scalar(
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
            selected_eq_scalar(&mut selection, &array, StringView::from_static_str("tcp"));
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [6]);
        let mut selection = Bitmap::from_slice_and_len(&[0xbe], 8);
        unsafe {
            selected_ne_scalar(&mut selection, &array, StringView::from_static_str("tcp"));
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [1, 2, 4, 5, 7]);

        // Partial selected, all valid, not-inlined
        let mut selection = Bitmap::from_slice_and_len(&[0x31], 8);
        unsafe {
            selected_eq_scalar(
                &mut selection,
                &array,
                StringView::from_static_str("auto-vectorization"),
            );
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [5]);
        let mut selection = Bitmap::from_slice_and_len(&[0x20], 8);
        unsafe {
            selected_ne_scalar(
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
            selected_eq_scalar(&mut selection, &array, StringView::from_static_str(""));
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [3, 6]);
        let mut selection = Bitmap::from_slice_and_len(&[u64::MAX], 8);
        unsafe {
            selected_ne_scalar(&mut selection, &array, StringView::from_static_str(""));
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [0, 2, 4, 5]);

        // All selected, partial valid, not-inlined
        let mut selection = Bitmap::from_slice_and_len(&[u64::MAX], 8);
        unsafe {
            selected_eq_scalar(
                &mut selection,
                &array,
                StringView::from_static_str("auto-vectorization"),
            );
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [5]);

        let mut selection = Bitmap::from_slice_and_len(&[u64::MAX], 8);
        unsafe {
            selected_ne_scalar(
                &mut selection,
                &array,
                StringView::from_static_str("auto-vectorization"),
            );
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [0, 2, 3, 4, 6,]);

        // Partial selected, partial valid, inlined
        let mut selection = Bitmap::from_slice_and_len(&[0xf7], 8);
        unsafe {
            selected_eq_scalar(&mut selection, &array, StringView::from_static_str(""));
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [6]);
        let mut selection = Bitmap::from_slice_and_len(&[0xf7], 8);
        unsafe {
            selected_ne_scalar(&mut selection, &array, StringView::from_static_str(""));
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [0, 2, 4, 5,]);

        // Partial selected, partial valid, not-inlined
        let mut selection = Bitmap::from_slice_and_len(&[0x31], 8);
        unsafe {
            selected_eq_scalar(
                &mut selection,
                &array,
                StringView::from_static_str("auto-vectorization"),
            );
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [5]);
        let mut selection = Bitmap::from_slice_and_len(&[0x31], 8);
        unsafe {
            selected_ne_scalar(
                &mut selection,
                &array,
                StringView::from_static_str("auto-vectorization"),
            );
        }
        assert_eq!(selection.iter_ones().collect::<Vec<_>>(), [0, 4]);
    }
}
