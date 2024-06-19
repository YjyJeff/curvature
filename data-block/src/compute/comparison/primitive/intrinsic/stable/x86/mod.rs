//! Manually SIMD

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
    ($target_feature:expr, $func_suffix:ident, $prefix:ident, $int_ty:ty, $suffix:ident) => {
        paste::paste! {
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<eq_scalar_ $int_ty _ $func_suffix>] (lhs: &AlignedVec<$int_ty>, rhs: $int_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $func_suffix>]::<false, false, _>(lhs, rhs, dst as _, |a, b| [<$prefix _cmpeq_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<ne_scalar_ $int_ty _ $func_suffix>] (lhs: &AlignedVec<$int_ty>, rhs: $int_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $func_suffix>]::<true, false, _>(lhs, rhs, dst as _, |a, b| [<$prefix _cmpeq_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<gt_scalar_ $int_ty _ $func_suffix>] (lhs: &AlignedVec<$int_ty>, rhs: $int_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $func_suffix>]::<false, false, _>(lhs, rhs, dst as _, |a, b| [<$prefix _cmpgt_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<ge_scalar_ $int_ty _ $func_suffix>] (lhs: &AlignedVec<$int_ty>, rhs: $int_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $func_suffix>]::<true, false, _>(lhs, rhs, dst as _, |a, b| [<$prefix _cmplt_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<lt_scalar_ $int_ty _ $func_suffix>] (lhs: &AlignedVec<$int_ty>, rhs: $int_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $func_suffix>]::<false, false, _>(lhs, rhs, dst as _, |a, b| [<$prefix _cmplt_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<le_scalar_ $int_ty _ $func_suffix>] (lhs: &AlignedVec<$int_ty>, rhs: $int_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $func_suffix>]::<true, false, _>(lhs, rhs, dst as _, |a, b| [<$prefix _cmpgt_ $suffix>](a, b))
            }
        }
    };
    // Unsigned integer match this pattern, because sse2 and avx2 does not support unsigned integer
    ($target_feature:expr, $func_suffix:ident, $prefix:ident, $uint_ty:ty, $int_ty:ty, $suffix:ident) => {
        paste::paste! {
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<eq_scalar_ $uint_ty _ $func_suffix>] (lhs: &AlignedVec<$uint_ty>, rhs: $uint_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $func_suffix>]::<false, false, _>(transmute(lhs), rhs as _, dst as _, |a, b| [<$prefix _cmpeq_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<ne_scalar_ $uint_ty _ $func_suffix>] (lhs: &AlignedVec<$uint_ty>, rhs: $uint_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $func_suffix>]::<true, false, _>(transmute(lhs), rhs as _, dst as _, |a, b| [<$prefix _cmpeq_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<gt_scalar_ $uint_ty _ $func_suffix>] (lhs: &AlignedVec<$uint_ty>, rhs: $uint_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $func_suffix>]::<false, true, _>(transmute(lhs), rhs as _, dst as _, |a, b| [<$prefix _cmpgt_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<ge_scalar_ $uint_ty _ $func_suffix>] (lhs: &AlignedVec<$uint_ty>, rhs: $uint_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $func_suffix>]::<true, true, _>(transmute(lhs), rhs as _, dst as _, |a, b| [<$prefix _cmplt_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<lt_scalar_ $uint_ty _ $func_suffix>] (lhs: &AlignedVec<$uint_ty>, rhs: $uint_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $func_suffix>]::<false, true, _>(transmute(lhs), rhs as _, dst as _, |a, b| [<$prefix _cmplt_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<le_scalar_ $uint_ty _ $func_suffix>] (lhs: &AlignedVec<$uint_ty>, rhs: $uint_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $func_suffix>]::<true, true, _>(transmute(lhs), rhs as _, dst as _, |a, b| [<$prefix _cmpgt_ $suffix>](a, b))
            }
        }
    };
}

/// Compare float scalar with specified target_feature
macro_rules! cmp_float {
    ($target_feature:expr, $func_suffix:ident, $prefix:ident, $ty:ty, $suffix:ident) => {
        paste::paste! {
            // array cmp scalar
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<eq_scalar_ $ty _ $func_suffix>] (lhs: &AlignedVec<$ty>, rhs: $ty, dst: *mut BitStore) {
                [<cmp_scalar_ $ty _ $func_suffix>](lhs, rhs, dst as _, |a, b| [<$prefix _cmpeq_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<ne_scalar_ $ty _ $func_suffix>] (lhs: &AlignedVec<$ty>, rhs: $ty, dst: *mut BitStore) {
                [<cmp_scalar_ $ty _ $func_suffix>](lhs, rhs, dst as _, |a, b| [<$prefix _cmpneq_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<gt_scalar_ $ty _ $func_suffix>] (lhs: &AlignedVec<$ty>, rhs: $ty, dst: *mut BitStore) {
                [<cmp_scalar_ $ty _ $func_suffix>](lhs, rhs, dst as _, |a, b| [<$prefix _cmpgt_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<ge_scalar_ $ty _ $func_suffix>] (lhs: &AlignedVec<$ty>, rhs: $ty, dst: *mut BitStore) {
                [<cmp_scalar_ $ty _ $func_suffix>](lhs, rhs, dst as _, |a, b| [<$prefix _cmpge_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<lt_scalar_ $ty _ $func_suffix>] (lhs: &AlignedVec<$ty>, rhs: $ty, dst: *mut BitStore) {
                [<cmp_scalar_ $ty _ $func_suffix>](lhs, rhs, dst as _, |a, b| [<$prefix _cmplt_ $suffix>](a, b))
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<le_scalar_ $ty _ $func_suffix>] (lhs: &AlignedVec<$ty>, rhs: $ty, dst: *mut BitStore) {
                [<cmp_scalar_ $ty _ $func_suffix>](lhs, rhs, dst as _, |a, b| [<$prefix _cmple_ $suffix>](a, b))
            }

            // // array cmp array
            // #[target_feature(enable = $target_feature)]
            // pub(crate) unsafe fn [<eq_ $ty _ $func_suffix>] (lhs: &AlignedVec<$ty>, rhs: &AlignedVec<$ty>, dst: *mut BitStore) {
            //     [<cmp_ $ty _ $func_suffix>](lhs, rhs, dst as _, |a, b| [<$prefix _cmpeq_ $suffix>](a, b))
            // }
            // #[target_feature(enable = $target_feature)]
            // pub(crate) unsafe fn [<ne_ $ty _ $func_suffix>] (lhs: &AlignedVec<$ty>, rhs: &AlignedVec<$ty>, dst: *mut BitStore) {
            //     [<cmp_ $ty _ $func_suffix>](lhs, rhs, dst as _, |a, b| [<$prefix _cmpneq_ $suffix>](a, b))
            // }
            // #[target_feature(enable = $target_feature)]
            // pub(crate) unsafe fn [<gt_ $ty _ $func_suffix>] (lhs: &AlignedVec<$ty>, rhs: &AlignedVec<$ty>, dst: *mut BitStore) {
            //     [<cmp_ $ty _ $func_suffix>](lhs, rhs, dst as _, |a, b| [<$prefix _cmpgt_ $suffix>](a, b))
            // }
            // #[target_feature(enable = $target_feature)]
            // pub(crate) unsafe fn [<ge_ $ty _ $func_suffix>] (lhs: &AlignedVec<$ty>, rhs: &AlignedVec<$ty>, dst: *mut BitStore) {
            //     [<cmp_ $ty _ $func_suffix>](lhs, rhs, dst as _, |a, b| [<$prefix _cmpge_ $suffix>](a, b))
            // }
            // #[target_feature(enable = $target_feature)]
            // pub(crate) unsafe fn [<lt_ $ty _ $func_suffix>] (lhs: &AlignedVec<$ty>, rhs: &AlignedVec<$ty>, dst: *mut BitStore) {
            //     [<cmp_ $ty _ $func_suffix>](lhs, rhs, dst as _, |a, b| [<$prefix _cmplt_ $suffix>](a, b))
            // }
            // #[target_feature(enable = $target_feature)]
            // pub(crate) unsafe fn [<le_ $ty _ $func_suffix>] (lhs: &AlignedVec<$ty>, rhs: &AlignedVec<$ty>, dst: *mut BitStore) {
            //     [<cmp_ $ty _ $func_suffix>](lhs, rhs, dst as _, |a, b| [<$prefix _cmple_ $suffix>](a, b))
            // }
        }
    };
}

/// Avx2
mod avx2;
pub(crate) use avx2::*;
/// Sse2 and Sse4
mod v2;
pub(crate) use v2::*;
