//! Compare string array

use libc::memcmp;

use crate::array::{Array, BooleanArray, StringArray};
use crate::element::string::{StringView, PREFIX_LEN};

macro_rules! impl_eq_scalar {
    ($func:ident, $op:tt, $conjunct:tt) => {
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

            let bitmap = dst.data.as_mut();
            if rhs.is_inlined() {
                bitmap.reset(
                    lhs.len(),
                    lhs.values_iter().map(|lhs| {
                        // Compare the whole string view. Inlined bytes is padded with 0
                        lhs.as_u128() $op rhs.as_u128()
                    }),
                );
            } else {
                let cmp_len = rhs.length as usize - PREFIX_LEN;
                bitmap.reset(
                    lhs.len(),
                    lhs.values_iter().map(|lhs| {
                        // Compare the whole string view. Inlined bytes is padded with 0
                        lhs.size_and_prefix_as_u64() $op rhs.size_and_prefix_as_u64()
                            $conjunct memcmp(
                                lhs.indirect_ptr().add(PREFIX_LEN) as _,
                                rhs.indirect_ptr().add(PREFIX_LEN) as _,
                                cmp_len,
                            ) $op 0
                    }),
                );
            }
        }
    };
}

impl_eq_scalar!(eq_scalar, ==, &&);
impl_eq_scalar!(ne_scalar, !=, ||);

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

            let bitmap = dst.data.as_mut();
            if rhs.is_inlined_in_prefix() {
                // We only need to compare the prefix
                bitmap.reset(
                    lhs.len(),
                    lhs.values_iter().map(|lhs| {
                        let cmp =
                            memcmp(lhs.inlined_ptr() as _, rhs.inlined_ptr() as _, PREFIX_LEN);
                        if cmp != 0 {
                            cmp $op 0
                        } else {
                            lhs.length $op rhs.length
                        }
                    }),
                );
            } else {
                bitmap.reset(lhs.len(), lhs.values_iter().map(|lhs| lhs $op rhs));
            }
        }
    };
}

impl_ord_scalar!(gt_scalar, >);
impl_ord_scalar!(ge_scalar, >=);
impl_ord_scalar!(lt_scalar, <);
impl_ord_scalar!(le_scalar, <=);

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
}
