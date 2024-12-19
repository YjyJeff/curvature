//! Computation of arrays

#[cfg(feature = "verify")]
macro_rules! assert_selection_is_valid {
    ($selection:ident, $array:ident) => {
        assert!($selection.is_empty() || $selection.len() == $array.len());
    };
}

pub mod arith;
pub mod comparison;
pub mod filter;
pub mod logical;
pub mod null;
pub mod regex_match;
pub mod sequence;

use self::logical::and_bitmaps_dynamic;
use crate::array::swar::SwarPtr;
use crate::bitmap::Bitmap;
use crate::types::PrimitiveType;

/// Types that supported by cpu
pub trait IntrinsicType: PrimitiveType {
    #[cfg(feature = "portable_simd")]
    /// Simd type of this intrinsic type
    type SimdType: IntrinsicSimdType<IntrinsicType = Self>;
}

#[cfg(not(feature = "portable_simd"))]
macro_rules! impl_intrinsic {
    ($({$ty:ty, $_simd_ty:ty, $_lanes:expr, $_bitmask:ty}),+) => {
        $(
            impl IntrinsicType for $ty{}
        )+
    };
}

#[cfg(not(feature = "portable_simd"))]
crate::macros::for_all_intrinsic!(impl_intrinsic);

/// Safety: Mutate condition
unsafe fn combine_validities(
    lhs: &SwarPtr<Bitmap>,
    rhs: &SwarPtr<Bitmap>,
    dst: &mut SwarPtr<Bitmap>,
) {
    match (lhs.all_valid(), rhs.all_valid()) {
        (true, true) => {
            // Both of them are all valid
            dst.as_mut().mutate().clear();
        }
        (true, false) => {
            // left is all valid, reference right
            dst.reference(rhs);
        }
        (false, true) => {
            // right is all valid, reference left
            dst.reference(lhs)
        }
        (false, false) => {
            // Both of them are not empty
            and_bitmaps_dynamic(
                lhs.as_raw_slice(),
                rhs.as_raw_slice(),
                dst.as_mut().mutate().clear_and_resize(lhs.len()),
            )
        }
    }
}

#[cfg(feature = "portable_simd")]
pub use portable_simd::*;

#[cfg(feature = "portable_simd")]
mod portable_simd {
    use super::IntrinsicType;
    use crate::aligned_vec::AlignedVec;
    use crate::utils::roundup_loops;
    use std::simd::{f32x16, f64x8, i16x32, i32x16, i64x8, i8x64, u16x32, u32x16, u64x8, u8x64};

    /// Corresponding Simd type of the intrinsic type
    pub trait IntrinsicSimdType: Copy + 'static {
        /// Number of lanes
        const LANES: usize;
        /// Corresponding alloc type
        type IntrinsicType: IntrinsicType<SimdType = Self>;

        /// Construct a simd type based on a single intrinsic type
        fn splat(v: Self::IntrinsicType) -> Self;
    }

    macro_rules! impl_intrinsic {
        ($({$ty:ty, $simd_ty:ty, $lanes:expr, $bitmask:ty}),+) => {
            $(
                impl IntrinsicType for $ty {
                    type SimdType = $simd_ty;
                }

                impl IntrinsicSimdType for $simd_ty {
                    const LANES: usize = $lanes;
                    type IntrinsicType = $ty;

                    #[inline]
                    fn splat(v: $ty) -> Self {
                        <$simd_ty>::splat(v)
                    }
                }
            )+
        };
    }

    crate::macros::for_all_intrinsic!(impl_intrinsic);

    impl<T: IntrinsicType> AlignedVec<T> {
        /// Transmute the vec to a slice of the SIMD type
        ///
        /// It is guaranteed to be safe, `AlignedVec`'s cap is multiple of 64bytes
        #[inline]
        #[must_use]
        pub fn as_intrinsic_simd(&self) -> &[T::SimdType] {
            unsafe {
                std::slice::from_raw_parts(
                    self.ptr.as_ptr() as *const _,
                    roundup_loops(self.len, <T::SimdType as IntrinsicSimdType>::LANES),
                )
            }
        }

        /// Transmute the vec to a slice of the SIMD type
        ///
        /// It is guaranteed to be safe, `AlignedVec`'s cap is multiple of 64bytes
        #[inline]
        #[must_use]
        pub fn as_intrinsic_simd_mut(&mut self) -> &mut [T::SimdType] {
            unsafe {
                std::slice::from_raw_parts_mut(
                    self.ptr.as_ptr() as *mut _,
                    roundup_loops(self.len, <T::SimdType as IntrinsicSimdType>::LANES),
                )
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_as_intrinsic_simd() {
            let val = AlignedVec::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
            let slice = val.as_intrinsic_simd();
            assert_eq!(slice.len(), 2);
        }
    }
}
