//! Comparison between bytes array

use crate::array::{Array, BinaryArray, BooleanArray};
use crate::bitmap::Bitmap;

macro_rules! cmp_scalar {
    ($func_name:ident, $cmp:tt) => {
        #[doc = concat!("Perform `left ", stringify!($cmp), " right` operation on a [`BinaryArray`] and a Element")]
        ///
        /// # Safety
        ///
        /// - `lhs`'s validity should not reference `dst`'s validity. In the computation graph,
        /// `array` must be the descendant of `dst`
        ///
        /// - No other arrays that reference the `dst`'s data and validity are accessed! In the
        /// computation graph, it will never happens
        pub unsafe fn $func_name(
            lhs: &BinaryArray,
            rhs: &[u8],
            dst: &mut BooleanArray,
        ) {
            dst.validity.reference(&lhs.validity);

            dst.data
                .as_mut()
                .mutate()
                .reset(lhs.len(), lhs.values_iter().map(|lhs| lhs $cmp rhs));
        }
    };
}

cmp_scalar!(eq_scalar, ==);
cmp_scalar!(ne_scalar, !=);
cmp_scalar!(gt_scalar, >);
cmp_scalar!(ge_scalar, >=);
cmp_scalar!(lt_scalar, <);
cmp_scalar!(le_scalar, <=);

macro_rules! selected_cmp_scalar {
    ($func_name:ident, $cmp:tt) => {
        #[doc = concat!("Perform `left ", stringify!($cmp), " right` operation on a [`BinaryArray`] and a Element")]
        ///
        /// # Safety
        ///
        /// - `array` and `selection` should have same length. Otherwise, undefined behavior happens
        ///
        /// - `selection` should not be referenced by any array
        pub unsafe fn $func_name(selection: &mut Bitmap, array: &BinaryArray, scalar: &[u8]) {
            debug_assert_eq!(selection.len(), array.len());

            let selection_all_valid = selection.all_valid();
            let mut guard = selection.mutate();
            let validity = array.validity();
            if selection_all_valid {
                if validity.all_valid() {
                    guard.reset(
                        array.len(),
                        array.values_iter().map(|bytes| bytes $cmp scalar),
                    );
                } else {
                    guard.reset(
                        array.len(),
                        array
                            .values_iter()
                            .zip(validity.iter())
                            .map(|(bytes, valid)| valid && bytes $cmp scalar),
                    )
                }
            } else if validity.all_valid() {
                guard.mutate_ones(|index| array.get_value_unchecked(index) $cmp scalar);
            } else {
                guard.mutate_ones(|index| {
                    validity.get_unchecked(index) && array.get_value_unchecked(index) $cmp scalar
                })
            }
        }

    };
}

selected_cmp_scalar!(selected_eq_scalar, ==);
selected_cmp_scalar!(selected_ne_scalar, !=);
selected_cmp_scalar!(selected_gt_scalar, >);
selected_cmp_scalar!(selected_ge_scalar, >=);
selected_cmp_scalar!(selected_lt_scalar, <);
selected_cmp_scalar!(selected_le_scalar, <=);
