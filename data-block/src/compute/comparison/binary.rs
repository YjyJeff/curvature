//! Comparison between bytes array

use crate::array::{Array, BinaryArray, BooleanArray};
use crate::mutate_array_func;

#[inline]
unsafe fn cmp_scalar<OP>(lhs: &BinaryArray, rhs: &[u8], dst: &mut BooleanArray, op: OP)
where
    OP: Fn(&[u8], &[u8]) -> bool,
{
    dst.validity.reference(&lhs.validity);

    dst.data
        .exactly_once_mut()
        .reset(lhs.len(), lhs.values_iter().map(|lhs| op(lhs, rhs)));
}

macro_rules! cmp_scalar {
    ($func_name:ident, $cmp:tt, $trait_bound:ident) => {
        mutate_array_func!(
            #[doc = concat!("Perform `left ", stringify!($cmp), " right` operation on a [`BinaryArray`] and a Element")]
            pub unsafe fn $func_name(
                lhs: &BinaryArray,
                rhs: &[u8],
                dst: &mut BooleanArray,
            ) {
                cmp_scalar(lhs, rhs, dst, |lhs, rhs| lhs $cmp rhs)
            }
        );
    };
}

cmp_scalar!(eq_scalar, ==, PartialEq);
cmp_scalar!(ne_scalar, !=, PartialEq);
cmp_scalar!(gt_scalar, >, PartialOrd);
cmp_scalar!(ge_scalar, >=, PartialOrd);
cmp_scalar!(lt_scalar, <, PartialOrd);
cmp_scalar!(le_scalar, <=, PartialOrd);
