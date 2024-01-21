//! Using `sse2`/`avx2`/`sse4.1` to implement SIMD, stable code can use SIMD via this module
//!

use super::{CmpFunc, PrimitiveCmpElement};
use crate::compute::comparison::primitive::{
    eq_scalar_default_, ge_scalar_default_, gt_scalar_default_, le_scalar_default_,
    lt_scalar_default_, ne_scalar_default_,
};

macro_rules! x86_target_use {
    ($($name:ident),+) => {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::{$($name),+};
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{$($name),+};
    };
}

/// Compare integer scalar with specified target_feature
macro_rules! cmp_int_scalar {
    // Signed integer match this pattern
    ($target_feature:expr, $prefix:ident, $int_ty:ty, $suffix:ident) => {
        paste::paste! {
            #[target_feature(enable = $target_feature)]
            pub(super) unsafe fn [<eq_scalar_ $int_ty _ $target_feature _>] (lhs: &AlignedVec<$int_ty>, rhs: $int_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $target_feature>]::<false, false, _>(lhs, rhs, dst as _, |a, b| [<$prefix _cmpeq_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(super) unsafe fn [<ne_scalar_ $int_ty _ $target_feature _>] (lhs: &AlignedVec<$int_ty>, rhs: $int_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $target_feature>]::<true, false, _>(lhs, rhs, dst as _, |a, b| [<$prefix _cmpeq_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(super) unsafe fn [<gt_scalar_ $int_ty _ $target_feature _>] (lhs: &AlignedVec<$int_ty>, rhs: $int_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $target_feature>]::<false, false, _>(lhs, rhs, dst as _, |a, b| [<$prefix _cmpgt_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(super) unsafe fn [<ge_scalar_ $int_ty _ $target_feature _>] (lhs: &AlignedVec<$int_ty>, rhs: $int_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $target_feature>]::<true, false, _>(lhs, rhs, dst as _, |a, b| [<$prefix _cmplt_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(super) unsafe fn [<lt_scalar_ $int_ty _ $target_feature _>] (lhs: &AlignedVec<$int_ty>, rhs: $int_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $target_feature>]::<false, false, _>(lhs, rhs, dst as _, |a, b| [<$prefix _cmplt_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(super) unsafe fn [<le_scalar_ $int_ty _ $target_feature _>] (lhs: &AlignedVec<$int_ty>, rhs: $int_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $target_feature>]::<true, false, _>(lhs, rhs, dst as _, |a, b| [<$prefix _cmpgt_ $suffix>](a, b))
            }
        }
    };
    // Unsigned integer match this pattern, because sse2 and avx2 does not support unsigned integer now
    ($target_feature:expr, $prefix:ident, $uint_ty:ty, $int_ty:ty, $suffix:ident) => {
        paste::paste! {
            #[target_feature(enable = $target_feature)]
            pub(super) unsafe fn [<eq_scalar_ $uint_ty _ $target_feature _>] (lhs: &AlignedVec<$uint_ty>, rhs: $uint_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $target_feature>]::<false, false, _>(transmute(lhs), rhs as _, dst as _, |a, b| [<$prefix _cmpeq_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(super) unsafe fn [<ne_scalar_ $uint_ty _ $target_feature _>] (lhs: &AlignedVec<$uint_ty>, rhs: $uint_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $target_feature>]::<true, false, _>(transmute(lhs), rhs as _, dst as _, |a, b| [<$prefix _cmpeq_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(super) unsafe fn [<gt_scalar_ $uint_ty _ $target_feature _>] (lhs: &AlignedVec<$uint_ty>, rhs: $uint_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $target_feature>]::<false, true, _>(transmute(lhs), rhs as _, dst as _, |a, b| [<$prefix _cmpgt_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(super) unsafe fn [<ge_scalar_ $uint_ty _ $target_feature _>] (lhs: &AlignedVec<$uint_ty>, rhs: $uint_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $target_feature>]::<true, true, _>(transmute(lhs), rhs as _, dst as _, |a, b| [<$prefix _cmplt_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(super) unsafe fn [<lt_scalar_ $uint_ty _ $target_feature _>] (lhs: &AlignedVec<$uint_ty>, rhs: $uint_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $target_feature>]::<false, true, _>(transmute(lhs), rhs as _, dst as _, |a, b| [<$prefix _cmplt_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(super) unsafe fn [<le_scalar_ $uint_ty _ $target_feature _>] (lhs: &AlignedVec<$uint_ty>, rhs: $uint_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $target_feature>]::<true, true, _>(transmute(lhs), rhs as _, dst as _, |a, b| [<$prefix _cmpgt_ $suffix>](a, b))
            }
        }
    };
}

/// Compare float scalar with specified target_feature
macro_rules! cmp_float_scalar {
    ($target_feature:expr, $prefix:ident, $ty:ty, $suffix:ident) => {
        paste::paste! {
            #[target_feature(enable = $target_feature)]
            pub(super) unsafe fn [<eq_scalar_ $ty _ $target_feature _>] (lhs: &AlignedVec<$ty>, rhs: $ty, dst: *mut BitStore) {
                [<cmp_scalar_ $ty _ $target_feature>](lhs, rhs, dst as _, |a, b| [<$prefix _cmpeq_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(super) unsafe fn [<ne_scalar_ $ty _ $target_feature _>] (lhs: &AlignedVec<$ty>, rhs: $ty, dst: *mut BitStore) {
                [<cmp_scalar_ $ty _ $target_feature>](lhs, rhs, dst as _, |a, b| [<$prefix _cmpneq_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(super) unsafe fn [<gt_scalar_ $ty _ $target_feature _>] (lhs: &AlignedVec<$ty>, rhs: $ty, dst: *mut BitStore) {
                [<cmp_scalar_ $ty _ $target_feature>](lhs, rhs, dst as _, |a, b| [<$prefix _cmpgt_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(super) unsafe fn [<ge_scalar_ $ty _ $target_feature _>] (lhs: &AlignedVec<$ty>, rhs: $ty, dst: *mut BitStore) {
                [<cmp_scalar_ $ty _ $target_feature>](lhs, rhs, dst as _, |a, b| [<$prefix _cmpge_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(super) unsafe fn [<lt_scalar_ $ty _ $target_feature _>] (lhs: &AlignedVec<$ty>, rhs: $ty, dst: *mut BitStore) {
                [<cmp_scalar_ $ty _ $target_feature>](lhs, rhs, dst as _, |a, b| [<$prefix _cmplt_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(super) unsafe fn [<le_scalar_ $ty _ $target_feature _>] (lhs: &AlignedVec<$ty>, rhs: $ty, dst: *mut BitStore) {
                [<cmp_scalar_ $ty _ $target_feature>](lhs, rhs, dst as _, |a, b| [<$prefix _cmple_ $suffix>](a, b))
            }
        }
    };
}

/// Does not include i64 and u64, because sse2 does not support i64
macro_rules! intrinsic_cmp_scalar {
    ($ty:ty) => {
        paste::paste! {
            impl PrimitiveCmpElement for $ty {
                const EQ_FUNC_AVX2: CmpFunc<Self> = [<eq_scalar_ $ty _avx2_>];
                const EQ_FUNC_SSE2: CmpFunc<Self> = [<eq_scalar_ $ty _sse2_>];
                const NE_FUNC_AVX2: CmpFunc<Self> = [<ne_scalar_ $ty _avx2_>];
                const NE_FUNC_SSE2: CmpFunc<Self> = [<ne_scalar_ $ty _sse2_>];
                const GT_FUNC_AVX2: CmpFunc<Self> = [<gt_scalar_ $ty _avx2_>];
                const GT_FUNC_SSE2: CmpFunc<Self> = [<gt_scalar_ $ty _sse2_>];
                const GE_FUNC_AVX2: CmpFunc<Self> = [<ge_scalar_ $ty _avx2_>];
                const GE_FUNC_SSE2: CmpFunc<Self> = [<ge_scalar_ $ty _sse2_>];
                const LT_FUNC_AVX2: CmpFunc<Self> = [<lt_scalar_ $ty _avx2_>];
                const LT_FUNC_SSE2: CmpFunc<Self> = [<lt_scalar_ $ty _sse2_>];
                const LE_FUNC_AVX2: CmpFunc<Self> = [<le_scalar_ $ty _avx2_>];
                const LE_FUNC_SSE2: CmpFunc<Self> = [<le_scalar_ $ty _sse2_>];
            }
        }
    };
}

intrinsic_cmp_scalar!(i8);
intrinsic_cmp_scalar!(u8);
intrinsic_cmp_scalar!(i16);
intrinsic_cmp_scalar!(u16);
intrinsic_cmp_scalar!(i32);
intrinsic_cmp_scalar!(u32);
intrinsic_cmp_scalar!(f32);
intrinsic_cmp_scalar!(f64);

macro_rules! impl_primitive_cmp_trait {
    (   $ty:ty,
        $eq_avx2:ident, $eq_sse2:ident,
        $ne_avx2:ident, $ne_sse2:ident,
        $gt_avx2:ident, $gt_sse2:ident,
        $ge_avx2:ident, $ge_sse2:ident,
        $lt_avx2:ident, $lt_sse2:ident,
        $le_avx2:ident, $le_sse2:ident
    ) => {
        impl PrimitiveCmpElement for $ty {
            const EQ_FUNC_AVX2: CmpFunc<Self> = $eq_avx2;
            const EQ_FUNC_SSE2: CmpFunc<Self> = $eq_sse2;
            const NE_FUNC_AVX2: CmpFunc<Self> = $ne_avx2;
            const NE_FUNC_SSE2: CmpFunc<Self> = $ne_sse2;
            const GT_FUNC_AVX2: CmpFunc<Self> = $gt_avx2;
            const GT_FUNC_SSE2: CmpFunc<Self> = $gt_sse2;
            const GE_FUNC_AVX2: CmpFunc<Self> = $ge_avx2;
            const GE_FUNC_SSE2: CmpFunc<Self> = $ge_sse2;
            const LT_FUNC_AVX2: CmpFunc<Self> = $lt_avx2;
            const LT_FUNC_SSE2: CmpFunc<Self> = $lt_sse2;
            const LE_FUNC_AVX2: CmpFunc<Self> = $le_avx2;
            const LE_FUNC_SSE2: CmpFunc<Self> = $le_sse2;
        }
    };
}

impl_primitive_cmp_trait!(
    i64,
    eq_scalar_i64_avx2_,
    eq_scalar_default_,
    ne_scalar_i64_avx2_,
    ne_scalar_default_,
    gt_scalar_i64_avx2_,
    gt_scalar_default_,
    ge_scalar_i64_avx2_,
    ge_scalar_default_,
    lt_scalar_i64_avx2_,
    lt_scalar_default_,
    le_scalar_i64_avx2_,
    le_scalar_default_
);

impl_primitive_cmp_trait!(
    u64,
    eq_scalar_u64_avx2_,
    eq_scalar_default_,
    ne_scalar_u64_avx2_,
    ne_scalar_default_,
    gt_scalar_u64_avx2_,
    gt_scalar_default_,
    ge_scalar_u64_avx2_,
    ge_scalar_default_,
    lt_scalar_u64_avx2_,
    lt_scalar_default_,
    le_scalar_u64_avx2_,
    le_scalar_default_
);

/// FIXME: Mixing macro and trait is pretty ugly
pub mod avx2;
use avx2::*;

/// FIXME: Mixing macro and trait is pretty ugly
pub mod sse2;
use sse2::*;
