//! Using SIMD to perform comparison with stable rust
//!
//! LLVM/Rust/C++ can not perform auto-vectorization for comparing primitive types and
//! write the result to bitmap. It sounds counterintuitive but it is the [truth] 😂.
//! We manually implement this case with SIMD instructions to accelerate the comparison.
//! Benchmarks shows we got a huge performance improvement 😊.
//!
//! After writing SIMD code manually, I know why LLVM generates strange SIMD code
//!
//! [truth]: https://stackoverflow.com/questions/77350870/why-llvm-can-not-auto-vectorize-comparing-two-arrays-and-write-result-to-vector?noredirect=1#comment136366602_77350870

use super::PartialOrdExt;
use crate::aligned_vec::AlignedVec;
use crate::array::{Array, BooleanArray, PrimitiveArray};
use crate::bitmap::{BitStore, Bitmap};
use crate::compute::logical::and_inplace;
use crate::types::IntrinsicType;

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod arm;

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
use self::arm::*;

/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
///
/// - `array`'s data and validity should not reference `dst`'s data and validity. In the computation
/// graph, `lhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
/// computation graph, it will never happens
#[inline]
unsafe fn cmp_scalar<T: PartialOrdExt>(
    selection: &mut Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    dst: &mut BooleanArray,
    partial_arith_threshold: f64,
    cmp_func: unsafe fn(&AlignedVec<T>, T, *mut BitStore),
    cmp_scalars_func: impl Fn(T, T) -> bool,
) where
    PrimitiveArray<T>: Array<Element = T>,
{
    debug_assert_selection_is_valid!(selection, array);

    and_inplace(selection, array.validity());
    if selection.ones_ratio() < partial_arith_threshold {
        selection
            .mutate()
            .mutate_ones(|index| cmp_scalars_func(array.get_value_unchecked(index), scalar))
    } else {
        cmp_func(
            &array.data,
            scalar,
            dst.data_mut()
                .mutate()
                .clear_and_resize(array.len())
                .as_mut_ptr(),
        );
        and_inplace(selection, dst.data());
    }
}

#[inline]
fn eq<T: IntrinsicType + PartialEq>(lhs: T, rhs: T) -> bool {
    lhs == rhs
}

#[inline]
fn ne<T: IntrinsicType + PartialEq>(lhs: T, rhs: T) -> bool {
    lhs != rhs
}

#[inline]
fn gt<T: IntrinsicType + PartialOrd>(lhs: T, rhs: T) -> bool {
    lhs > rhs
}

#[inline]
fn ge<T: IntrinsicType + PartialOrd>(lhs: T, rhs: T) -> bool {
    lhs >= rhs
}

#[inline]
fn lt<T: IntrinsicType + PartialOrd>(lhs: T, rhs: T) -> bool {
    lhs < rhs
}

#[inline]
fn le<T: IntrinsicType + PartialOrd>(lhs: T, rhs: T) -> bool {
    lhs <= rhs
}

macro_rules! impl_partial_ord_ext {
    ($ty:ty, $default_th:expr, $neon_th:expr, $avx2_th:expr, $avx512_th:expr) => {
        impl PartialOrdExt for $ty {
            /// If the selectivity is smaller than this threshold, partial computation is used
            const PARTIAL_CMP_THRESHOLD: f64 = $default_th;
            #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
            /// If the selectivity is smaller than this threshold, partial computation is used
            const NEON_PARTIAL_CMP_THRESHOLD: f64 = $neon_th;
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            /// If the selectivity is smaller than this threshold, partial computation is used
            const AVX2_PARTIAL_CMP_THRESHOLD: f64 = $avx2_th;
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            /// If the selectivity is smaller than this threshold, partial computation is used
            const AVX512_PARTIAL_CMP_THRESHOLD: f64 = $avx512_th;

            impl_partial_ord_ext!(eq, $ty);
            impl_partial_ord_ext!(ne, $ty);
            impl_partial_ord_ext!(gt, $ty);
            impl_partial_ord_ext!(ge, $ty);
            impl_partial_ord_ext!(lt, $ty);
            impl_partial_ord_ext!(le, $ty);
        }
    };
    ($cmp_func:ident, $ty:ty) => {
        paste::paste! {
            unsafe fn [<$cmp_func _scalar>](
                selection: &mut Bitmap,
                array: &PrimitiveArray<$ty>,
                scalar: $ty,
                dst: &mut BooleanArray,
            ){
                unsafe {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        if std::arch::is_x86_feature_detected!("avx2") {
                            return cmp_scalar(selection, array, scalar, dst, Self::AVX2_PARTIAL_CMP_THRESHOLD, [<$cmp_func _scalar_ $ty _avx2>], $cmp_func);
                        } else {
                            return cmp_scalar(selection, array, scalar, dst, Self::PARTIAL_CMP_THRESHOLD, [<$cmp_func _scalar_ $ty _v2>], $cmp_func);
                        }
                    }

                    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
                    {
                        if std::arch::is_aarch64_feature_detected!("neon") {
                            return cmp_scalar(selection, array, scalar, dst, Self::NEON_PARTIAL_CMP_THRESHOLD, [<$cmp_func _scalar_ $ty _neon>], $cmp_func);
                        }
                    }

                    cmp_scalar_default(selection, array, scalar, dst, $cmp_func);
                }
            }
        }
    }
}

unsafe fn cmp_scalar_default<T: PartialOrdExt>(
    selection: &mut Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    dst: &mut BooleanArray,
    cmp_scalars_func: impl Fn(T, T) -> bool,
) where
    PrimitiveArray<T>: Array<Element = T>,
{
    debug_assert_selection_is_valid!(selection, array);

    and_inplace(selection, array.validity());
    if selection.ones_ratio() < T::PARTIAL_CMP_THRESHOLD {
        selection
            .mutate()
            .mutate_ones(|index| cmp_scalars_func(array.get_value_unchecked(index), scalar))
    } else {
        dst.data_mut().mutate().reset(
            array.len(),
            array
                .values_iter()
                .map(|element| cmp_scalars_func(element, scalar)),
        );
        and_inplace(selection, dst.data());
    }
}

// FIXME: Tune the parameter
impl_partial_ord_ext!(i8, 0.0, 0.0, 0.0, 0.0);
impl_partial_ord_ext!(u8, 0.0, 0.0, 0.0, 0.0);
impl_partial_ord_ext!(i16, 0.0, 0.0, 0.0, 0.0);
impl_partial_ord_ext!(u16, 0.0, 0.0, 0.0, 0.0);
impl_partial_ord_ext!(i32, 0.0, 0.0, 0.0, 0.0);
impl_partial_ord_ext!(u32, 0.0, 0.0, 0.0, 0.0);
impl_partial_ord_ext!(i64, 0.0, 0.0, 0.0, 0.0);
impl_partial_ord_ext!(u64, 0.0, 0.0, 0.0, 0.0);
impl_partial_ord_ext!(f32, 0.0, 0.0, 0.0, 0.0);
impl_partial_ord_ext!(f64, 0.0, 0.0, 0.0, 0.0);
