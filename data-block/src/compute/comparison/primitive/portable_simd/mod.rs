use crate::aligned_vec::AlignedVec;
use crate::array::{Array, BooleanArray, PrimitiveArray};
use crate::compute::{for_all_intrinsic, IntrinsicSimdType, IntrinsicType};
use crate::utils::roundup_loops;
use std::simd::{f32x16, f64x8, i16x32, i32x16, i64x8, i8x64, u16x32, u32x16, u64x8, u8x64};
use std::simd::{Mask, SimdElement, SimdPartialEq, SimdPartialOrd, ToBitMask};

/// Compare intrinsic simd type
pub trait IntrinsicSimdOrd: IntrinsicSimdType + SimdPartialOrd<Mask = Self::MaskType> {
    /// Type of the bitmask
    type BitMaskType: Default + Clone;
    /// Type of the mask after comparison
    type MaskType: ToBitMask<BitMask = Self::BitMaskType>;
}

macro_rules! impl_simd_ord {
    ($({$ty:ty, $simd_ty:ty, $lanes:expr, $bitmask:ty}),+) => {
        $(
            impl IntrinsicSimdOrd for $simd_ty {
                type BitMaskType = $bitmask;
                type MaskType = Mask<<$ty as SimdElement>::Mask, $lanes>;
            }
        )+
    };
}

for_all_intrinsic!(impl_simd_ord);

#[inline(always)]
fn cmp_scalar<T, F>(
    lhs: &AlignedVec<T>,
    rhs: T,
    dst: &mut [<T::SimdType as IntrinsicSimdOrd>::BitMaskType],
    cmp: F,
) where
    T: IntrinsicType,
    T::SimdType: IntrinsicSimdOrd,
    F: Fn(T::SimdType, T::SimdType) -> <T::SimdType as IntrinsicSimdOrd>::BitMaskType,
{
    let rhs = T::SimdType::splat(rhs);
    lhs.as_intrinsic_simd()
        .iter()
        .zip(dst)
        .for_each(|(lhs, dst)| {
            *dst = cmp(*lhs, rhs);
        });
}

macro_rules! impl_equal {
    ($lhs:ident, $rhs:ident, $dst:ident) => {
        cmp_scalar($lhs, $rhs, $dst, |lhs, rhs| lhs.simd_eq(rhs).to_bitmask())
    };
}

crate::dynamic_func!(
    eq_scalar_avx512,
    eq_scalar_avx2,
    eq_scalar_neon,
    eq_scalar_default,
    eq_scalar_dynamic,
    impl_equal,
    <T>,
    (lhs: &AlignedVec<T>, rhs: T, dst: &mut [<T::SimdType as IntrinsicSimdOrd>::BitMaskType]),
    where
        T: IntrinsicType,
        T::SimdType: IntrinsicSimdOrd
);

macro_rules! impl_not_equal {
    ($lhs:ident, $rhs:ident, $dst:ident) => {
        cmp_scalar($lhs, $rhs, $dst, |lhs, rhs| lhs.simd_ne(rhs).to_bitmask())
    };
}

crate::dynamic_func!(
    ne_scalar_avx512,
    ne_scalar_avx2,
    ne_scalar_neon,
    ne_scalar_default,
    ne_scalar_dynamic,
    impl_not_equal,
    <T>,
    (lhs: &AlignedVec<T>, rhs: T, dst: &mut [<T::SimdType as IntrinsicSimdOrd>::BitMaskType]),
    where
        T: IntrinsicType,
        T::SimdType: IntrinsicSimdOrd
);

macro_rules! impl_greater_than {
    ($lhs:ident, $rhs:ident, $dst:ident) => {
        cmp_scalar($lhs, $rhs, $dst, |lhs, rhs| lhs.simd_gt(rhs).to_bitmask())
    };
}

crate::dynamic_func!(
    gt_scalar_avx512,
    gt_scalar_avx2,
    gt_scalar_neon,
    gt_scalar_default,
    gt_scalar_dynamic,
    impl_greater_than,
    <T>,
    (lhs: &AlignedVec<T>, rhs: T, dst: &mut [<T::SimdType as IntrinsicSimdOrd>::BitMaskType]),
    where
        T: IntrinsicType,
        T::SimdType: IntrinsicSimdOrd
);

macro_rules! impl_greater_than_or_equal {
    ($lhs:ident, $rhs:ident, $dst:ident) => {
        cmp_scalar($lhs, $rhs, $dst, |lhs, rhs| lhs.simd_ge(rhs).to_bitmask())
    };
}

crate::dynamic_func!(
    ge_scalar_avx512,
    ge_scalar_avx2,
    ge_scalar_neon,
    ge_scalar_default,
    ge_scalar_dynamic,
    impl_greater_than_or_equal,
    <T>,
    (lhs: &AlignedVec<T>, rhs: T, dst: &mut [<T::SimdType as IntrinsicSimdOrd>::BitMaskType]),
    where
        T: IntrinsicType,
        T::SimdType: IntrinsicSimdOrd
);

macro_rules! impl_less_than {
    ($lhs:ident, $rhs:ident, $dst:ident) => {
        cmp_scalar($lhs, $rhs, $dst, |lhs, rhs| lhs.simd_lt(rhs).to_bitmask())
    };
}

crate::dynamic_func!(
    lt_scalar_avx512,
    lt_scalar_avx2,
    lt_scalar_neon,
    lt_scalar_default,
    lt_scalar_dynamic,
    impl_less_than,
    <T>,
    (lhs: &AlignedVec<T>, rhs: T, dst: &mut [<T::SimdType as IntrinsicSimdOrd>::BitMaskType]),
    where
        T: IntrinsicType,
        T::SimdType: IntrinsicSimdOrd
);

macro_rules! impl_less_than_or_equal {
    ($lhs:ident, $rhs:ident, $dst:ident) => {
        cmp_scalar($lhs, $rhs, $dst, |lhs, rhs| lhs.simd_le(rhs).to_bitmask())
    };
}

crate::dynamic_func!(
    le_scalar_avx512,
    le_scalar_avx2,
    le_scalar_neon,
    le_scalar_default,
    le_scalar_dynamic,
    impl_less_than_or_equal,
    <T>,
    (lhs: &AlignedVec<T>, rhs: T, dst: &mut [<T::SimdType as IntrinsicSimdOrd>::BitMaskType]),
    where
        T: IntrinsicType,
        T::SimdType: IntrinsicSimdOrd
);

macro_rules! cmp_scalar {
    ($func_name:ident, $cmp_func:ident, $op:tt) => {
        #[doc = concat!("Perform `lhs ", stringify!($op), " rhs` between `PrimitiveArray<T>` and `T`")]
        pub fn $func_name<T>(lhs: &PrimitiveArray<T>, rhs: T, dst: &mut BooleanArray)
        where
            T: IntrinsicType,
            T::SimdType: IntrinsicSimdOrd,
            PrimitiveArray<T>: Array,
        {
            // Reference dst's validity to lhs'validity
            dst.validity.reference(&lhs.validity);

            // SAFETY:
            // - dst is the unique owner of the data, and no-one will write to dst anymore.
            //   During the function call, `exactly_once_mut` is called exactly once
            // - we call avx2 function iff the feature is detected and sse2 is enabled by default
            // - BooleanArray's capacity is also multiple of 64, which means its bitmap can be mapped
            //   one-to-one to lhs.data
            unsafe {
                let uninitialized = dst.data.exactly_once_mut().clear_and_resize(lhs.len());

                $cmp_func(
                    &lhs.data,
                    rhs,
                    std::slice::from_raw_parts_mut(
                        uninitialized.as_mut_ptr() as _,
                        roundup_loops(lhs.len(), <T::SimdType as IntrinsicSimdType>::LANES),
                    ),
                )
            }
        }
    };
}

cmp_scalar!(eq_scalar, eq_scalar_dynamic, ==);
cmp_scalar!(ne_scalar, ne_scalar_dynamic, !=);
cmp_scalar!(gt_scalar, gt_scalar_dynamic, >);
cmp_scalar!(ge_scalar, ge_scalar_dynamic, >=);
cmp_scalar!(lt_scalar, lt_scalar_dynamic, <);
cmp_scalar!(le_scalar, le_scalar_dynamic, <=);
