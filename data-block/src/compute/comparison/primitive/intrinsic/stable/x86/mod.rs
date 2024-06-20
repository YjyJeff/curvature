//! Manually SIMD

macro_rules! x86_target_use {
    ($($name:ident),+) => {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::{$($name),+};
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{$($name),+};
    };
}

/// Compare integer with specified target_feature
macro_rules! cmp_int {
    // Signed integer match this pattern
    ($target_feature:expr, $func_suffix:ident, $prefix:ident, $int_ty:ty, $suffix:ident) => {
        paste::paste! {
            // array cmp scalar
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<eq_scalar_ $int_ty _ $func_suffix>] (lhs: &AlignedVec<$int_ty>, rhs: $int_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $func_suffix>]::<false, false>(lhs, rhs, dst as _, [<$prefix _cmpeq_ $suffix>])
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<ne_scalar_ $int_ty _ $func_suffix>] (lhs: &AlignedVec<$int_ty>, rhs: $int_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $func_suffix>]::<true, false>(lhs, rhs, dst as _, [<$prefix _cmpeq_ $suffix>])
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<gt_scalar_ $int_ty _ $func_suffix>] (lhs: &AlignedVec<$int_ty>, rhs: $int_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $func_suffix>]::<false, false>(lhs, rhs, dst as _, [<$prefix _cmpgt_ $suffix>])
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<ge_scalar_ $int_ty _ $func_suffix>] (lhs: &AlignedVec<$int_ty>, rhs: $int_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $func_suffix>]::<true, false>(lhs, rhs, dst as _, [<$prefix _cmplt_ $suffix>])
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<lt_scalar_ $int_ty _ $func_suffix>] (lhs: &AlignedVec<$int_ty>, rhs: $int_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $func_suffix>]::<false, false>(lhs, rhs, dst as _, [<$prefix _cmplt_ $suffix>])
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<le_scalar_ $int_ty _ $func_suffix>] (lhs: &AlignedVec<$int_ty>, rhs: $int_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $func_suffix>]::<true, false>(lhs, rhs, dst as _, [<$prefix _cmpgt_ $suffix>])
            }

            // array cmp array
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<eq_ $int_ty _ $func_suffix>] (lhs: &AlignedVec<$int_ty>, rhs: &AlignedVec<$int_ty>, dst: *mut BitStore) {
                [<cmp_ $int_ty _ $func_suffix>]::<false, false>(lhs, rhs, dst as _, [<$prefix _cmpeq_ $suffix>])
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<ne_ $int_ty _ $func_suffix>] (lhs: &AlignedVec<$int_ty>, rhs: &AlignedVec<$int_ty>, dst: *mut BitStore) {
                [<cmp_ $int_ty _ $func_suffix>]::<true, false>(lhs, rhs, dst as _, [<$prefix _cmpeq_ $suffix>])
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<gt_ $int_ty _ $func_suffix>] (lhs: &AlignedVec<$int_ty>, rhs: &AlignedVec<$int_ty>, dst: *mut BitStore) {
                [<cmp_ $int_ty _ $func_suffix>]::<false, false>(lhs, rhs, dst as _, [<$prefix _cmpgt_ $suffix>])
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<ge_ $int_ty _ $func_suffix>] (lhs: &AlignedVec<$int_ty>, rhs: &AlignedVec<$int_ty>, dst: *mut BitStore) {
                [<cmp_ $int_ty _ $func_suffix>]::<true, false>(lhs, rhs, dst as _, [<$prefix _cmplt_ $suffix>])
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<lt_ $int_ty _ $func_suffix>] (lhs: &AlignedVec<$int_ty>, rhs: &AlignedVec<$int_ty>, dst: *mut BitStore) {
                [<cmp_ $int_ty _ $func_suffix>]::<false, false>(lhs, rhs, dst as _, [<$prefix _cmplt_ $suffix>])
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<le_ $int_ty _ $func_suffix>] (lhs: &AlignedVec<$int_ty>, rhs: &AlignedVec<$int_ty>, dst: *mut BitStore) {
                [<cmp_ $int_ty _ $func_suffix>]::<true, false>(lhs, rhs, dst as _, [<$prefix _cmpgt_ $suffix>])
            }
        }
    };
    // Unsigned integer match this pattern, because sse2 and avx2 does not support unsigned integer
    ($target_feature:expr, $func_suffix:ident, $prefix:ident, $uint_ty:ty, $int_ty:ty, $suffix:ident) => {
        paste::paste! {
            // array cmp scalar
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<eq_scalar_ $uint_ty _ $func_suffix>] (lhs: &AlignedVec<$uint_ty>, rhs: $uint_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $func_suffix>]::<false, false>(
                    transmute::<&AlignedVec<$uint_ty>, &AlignedVec<$int_ty>>(lhs),
                    rhs as _,
                    dst as _,
                    [<$prefix _cmpeq_ $suffix>]
                )
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<ne_scalar_ $uint_ty _ $func_suffix>] (lhs: &AlignedVec<$uint_ty>, rhs: $uint_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $func_suffix>]::<true, false>(
                    transmute::<&AlignedVec<$uint_ty>, &AlignedVec<$int_ty>>(lhs),
                    rhs as _,
                    dst as _,
                    [<$prefix _cmpeq_ $suffix>]
                )
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<gt_scalar_ $uint_ty _ $func_suffix>] (lhs: &AlignedVec<$uint_ty>, rhs: $uint_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $func_suffix>]::<false, true>(
                    transmute::<&AlignedVec<$uint_ty>, &AlignedVec<$int_ty>>(lhs),
                    rhs as _,
                    dst as _,
                    [<$prefix _cmpgt_ $suffix>]
                )
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<ge_scalar_ $uint_ty _ $func_suffix>] (lhs: &AlignedVec<$uint_ty>, rhs: $uint_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $func_suffix>]::<true, true>(
                    transmute::<&AlignedVec<$uint_ty>, &AlignedVec<$int_ty>>(lhs),
                    rhs as _,
                    dst as _,
                    [<$prefix _cmplt_ $suffix>]
                )
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<lt_scalar_ $uint_ty _ $func_suffix>] (lhs: &AlignedVec<$uint_ty>, rhs: $uint_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $func_suffix>]::<false, true>(
                    transmute::<&AlignedVec<$uint_ty>, &AlignedVec<$int_ty>>(lhs),
                    rhs as _,
                    dst as _,
                    [<$prefix _cmplt_ $suffix>]
                )
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<le_scalar_ $uint_ty _ $func_suffix>] (lhs: &AlignedVec<$uint_ty>, rhs: $uint_ty, dst: *mut BitStore) {
                [<cmp_scalar_ $int_ty _ $func_suffix>]::<true, true>(
                    transmute::<&AlignedVec<$uint_ty>, &AlignedVec<$int_ty>>(lhs),
                    rhs as _,
                    dst as _,
                    [<$prefix _cmpgt_ $suffix>]
                )
            }

            // array cmp array
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<eq_ $uint_ty _ $func_suffix>] (lhs: &AlignedVec<$uint_ty>, rhs: &AlignedVec<$uint_ty>, dst: *mut BitStore) {
                [<cmp_ $int_ty _ $func_suffix>]::<false, false>(
                    transmute::<&AlignedVec<$uint_ty>, &AlignedVec<$int_ty>>(lhs),
                    transmute::<&AlignedVec<$uint_ty>, &AlignedVec<$int_ty>>(rhs),
                    dst as _,
                    [<$prefix _cmpeq_ $suffix>]
                )
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<ne_ $uint_ty _ $func_suffix>] (lhs: &AlignedVec<$uint_ty>, rhs: &AlignedVec<$uint_ty>, dst: *mut BitStore) {
                [<cmp_ $int_ty _ $func_suffix>]::<true, false>(
                    transmute::<&AlignedVec<$uint_ty>, &AlignedVec<$int_ty>>(lhs),
                    transmute::<&AlignedVec<$uint_ty>, &AlignedVec<$int_ty>>(rhs),
                    dst as _,
                    [<$prefix _cmpeq_ $suffix>]
                )
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<gt_ $uint_ty _ $func_suffix>] (lhs: &AlignedVec<$uint_ty>, rhs: &AlignedVec<$uint_ty>, dst: *mut BitStore) {
                [<cmp_ $int_ty _ $func_suffix>]::<false, true>(
                    transmute::<&AlignedVec<$uint_ty>, &AlignedVec<$int_ty>>(lhs),
                    transmute::<&AlignedVec<$uint_ty>, &AlignedVec<$int_ty>>(rhs),
                    dst as _,
                    [<$prefix _cmpgt_ $suffix>]
                )
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<ge_ $uint_ty _ $func_suffix>] (lhs: &AlignedVec<$uint_ty>, rhs: &AlignedVec<$uint_ty>, dst: *mut BitStore) {
                [<cmp_ $int_ty _ $func_suffix>]::<true, true>(
                    transmute::<&AlignedVec<$uint_ty>, &AlignedVec<$int_ty>>(lhs),
                    transmute::<&AlignedVec<$uint_ty>, &AlignedVec<$int_ty>>(rhs),
                    dst as _,
                    [<$prefix _cmplt_ $suffix>]
                )
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<lt_ $uint_ty _ $func_suffix>] (lhs: &AlignedVec<$uint_ty>, rhs: &AlignedVec<$uint_ty>, dst: *mut BitStore) {
                [<cmp_ $int_ty _ $func_suffix>]::<false, true>(
                    transmute::<&AlignedVec<$uint_ty>, &AlignedVec<$int_ty>>(lhs),
                    transmute::<&AlignedVec<$uint_ty>, &AlignedVec<$int_ty>>(rhs),
                    dst as _,
                    [<$prefix _cmplt_ $suffix>]
                )
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<le_ $uint_ty _ $func_suffix>] (lhs: &AlignedVec<$uint_ty>, rhs: &AlignedVec<$uint_ty>, dst: *mut BitStore) {
                [<cmp_ $int_ty _ $func_suffix>]::<true, true>(
                    transmute::<&AlignedVec<$uint_ty>, &AlignedVec<$int_ty>>(lhs),
                    transmute::<&AlignedVec<$uint_ty>, &AlignedVec<$int_ty>>(rhs),
                    dst as _,
                    [<$prefix _cmpgt_ $suffix>]
                )
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
                [<cmp_scalar_ $ty _ $func_suffix>](lhs, rhs, dst as _, [<$prefix _cmpeq_ $suffix>])
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<ne_scalar_ $ty _ $func_suffix>] (lhs: &AlignedVec<$ty>, rhs: $ty, dst: *mut BitStore) {
                [<cmp_scalar_ $ty _ $func_suffix>](lhs, rhs, dst as _, [<$prefix _cmpneq_ $suffix>])
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<gt_scalar_ $ty _ $func_suffix>] (lhs: &AlignedVec<$ty>, rhs: $ty, dst: *mut BitStore) {
                [<cmp_scalar_ $ty _ $func_suffix>](lhs, rhs, dst as _, [<$prefix _cmpgt_ $suffix>])
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<ge_scalar_ $ty _ $func_suffix>] (lhs: &AlignedVec<$ty>, rhs: $ty, dst: *mut BitStore) {
                [<cmp_scalar_ $ty _ $func_suffix>](lhs, rhs, dst as _, [<$prefix _cmpge_ $suffix>])
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<lt_scalar_ $ty _ $func_suffix>] (lhs: &AlignedVec<$ty>, rhs: $ty, dst: *mut BitStore) {
                [<cmp_scalar_ $ty _ $func_suffix>](lhs, rhs, dst as _, [<$prefix _cmplt_ $suffix>])
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<le_scalar_ $ty _ $func_suffix>] (lhs: &AlignedVec<$ty>, rhs: $ty, dst: *mut BitStore) {
                [<cmp_scalar_ $ty _ $func_suffix>](lhs, rhs, dst as _, [<$prefix _cmple_ $suffix>])
            }

            // array cmp array
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<eq_ $ty _ $func_suffix>] (lhs: &AlignedVec<$ty>, rhs: &AlignedVec<$ty>, dst: *mut BitStore) {
                [<cmp_ $ty _ $func_suffix>](lhs, rhs, dst as _, [<$prefix _cmpeq_ $suffix>])
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<ne_ $ty _ $func_suffix>] (lhs: &AlignedVec<$ty>, rhs: &AlignedVec<$ty>, dst: *mut BitStore) {
                [<cmp_ $ty _ $func_suffix>](lhs, rhs, dst as _, [<$prefix _cmpneq_ $suffix>])
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<gt_ $ty _ $func_suffix>] (lhs: &AlignedVec<$ty>, rhs: &AlignedVec<$ty>, dst: *mut BitStore) {
                [<cmp_ $ty _ $func_suffix>](lhs, rhs, dst as _, [<$prefix _cmpgt_ $suffix>])
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<ge_ $ty _ $func_suffix>] (lhs: &AlignedVec<$ty>, rhs: &AlignedVec<$ty>, dst: *mut BitStore) {
                [<cmp_ $ty _ $func_suffix>](lhs, rhs, dst as _, [<$prefix _cmpge_ $suffix>])
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<lt_ $ty _ $func_suffix>] (lhs: &AlignedVec<$ty>, rhs: &AlignedVec<$ty>, dst: *mut BitStore) {
                [<cmp_ $ty _ $func_suffix>](lhs, rhs, dst as _, [<$prefix _cmplt_ $suffix>])
            }
            #[target_feature(enable = $target_feature)]
            pub(crate) unsafe fn [<le_ $ty _ $func_suffix>] (lhs: &AlignedVec<$ty>, rhs: &AlignedVec<$ty>, dst: *mut BitStore) {
                [<cmp_ $ty _ $func_suffix>](lhs, rhs, dst as _, [<$prefix _cmple_ $suffix>])
            }
        }
    };
}

/// Avx2
mod avx2;
pub(crate) use avx2::*;
/// Sse2 and Sse4
mod v2;
pub(crate) use v2::*;
