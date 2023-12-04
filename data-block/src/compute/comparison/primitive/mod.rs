//! Comparison between primitive arrays

#[cfg(test)]
macro_rules! cmp_assert {
    ($lhs:expr, $rhs:ident, $cmp_func:ident, $gt:expr) => {
        unsafe {
            let len = $lhs.len();
            let mut dst = Bitmap::with_capacity(len);
            let uninitialized = dst.clear_and_resize(len);
            $cmp_func($lhs, $rhs, uninitialized.as_mut_ptr());

            assert_eq!(dst, Bitmap::from_slice_and_len($gt, len));
        }
    };
}

/// TODO: Support compare `Int128` and `Decimal`
#[cfg(not(feature = "portable_simd"))]
mod stable;
#[cfg(not(feature = "portable_simd"))]
pub use stable::*;

#[cfg(feature = "portable_simd")]
mod portable_simd;
#[cfg(feature = "portable_simd")]
pub use portable_simd::*;

use crate::aligned_vec::AlignedVec;
use crate::array::primitive::PrimitiveType;
use crate::bitmap::BitStore;

macro_rules! impl_cmp_scalar_default {
    ($func_name:ident, $op:tt) => {
        #[inline]
        unsafe fn $func_name<T: PrimitiveType + PartialOrd>(
            lhs: &AlignedVec<T>,
            rhs: T,
            dst: *mut BitStore,
        ) {
            crate::bitmap::reset_bitmap_raw(
                dst,
                lhs.len(),
                lhs.as_slice().iter().map(|&lhs| lhs $op rhs),
            );
        }
    };
}

impl_cmp_scalar_default!(eq_scalar_default_, ==);
impl_cmp_scalar_default!(ne_scalar_default_, !=);
impl_cmp_scalar_default!(gt_scalar_default_, >);
impl_cmp_scalar_default!(ge_scalar_default_, >=);
impl_cmp_scalar_default!(lt_scalar_default_, <);
impl_cmp_scalar_default!(le_scalar_default_, <=);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::Bitmap;

    #[test]
    fn test_order_scalar_default() {
        let lhs = AlignedVec::<i64>::from_slice(&[
            7, 9, 100, 234, 45, 100, -10, -99, -123, -56, 100, 345, -34554,
        ]);

        let rhs = 100;
        cmp_assert!(&lhs, rhs, eq_scalar_default_, &[0x0424]);
        cmp_assert!(&lhs, rhs, ne_scalar_default_, &[!0x0424]);
        cmp_assert!(&lhs, rhs, gt_scalar_default_, &[0x0808]);
        cmp_assert!(&lhs, rhs, ge_scalar_default_, &[0x0c2c]);
        cmp_assert!(&lhs, rhs, lt_scalar_default_, &[!0x0c2c]);
        cmp_assert!(&lhs, rhs, le_scalar_default_, &[!0x0808]);
    }
}
