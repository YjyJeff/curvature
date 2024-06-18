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
use crate::compute::comparison::primitive::private::{eq, ge, gt, le, lt, ne};
use crate::compute::comparison::primitive::{cmp_default, cmp_scalar_default};
use crate::compute::comparison::{
    timestamp_scalar_eq_scalar, timestamp_scalar_ge_scalar, timestamp_scalar_gt_scalar,
    timestamp_scalar_le_scalar, timestamp_scalar_lt_scalar, timestamp_scalar_ne_scalar,
};
use crate::compute::logical::and_inplace;

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod arm;
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
use self::arm::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use self::x86::*;

/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
///
/// - `array`'s data and validity should not reference `temp`'s data and validity. In the computation
/// graph, `lhs` must be the descendant of `temp`
///
/// - No other arrays that reference the `temp`'s data and validity are accessed! In the
/// computation graph, it will never happens
#[inline]
unsafe fn cmp_scalar<T: PartialOrdExt>(
    selection: &mut Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    temp: &mut BooleanArray,
    partial_arith_threshold: f64,
    cmp_func: unsafe fn(&AlignedVec<T>, T, *mut BitStore),
    cmp_scalars_func: impl Fn(T, T) -> bool,
) where
    PrimitiveArray<T>: Array<Element = T>,
{
    debug_assert_selection_is_valid!(selection, array);
    // Benchmark shows the two and_inplace function cost 50% of the time 😭

    and_inplace(selection, array.validity());
    if selection.ones_ratio() < partial_arith_threshold {
        selection
            .mutate()
            .mutate_ones(|index| cmp_scalars_func(array.get_value_unchecked(index), scalar))
    } else {
        // Benchmark shows clear_and_resize may cost lots of time
        cmp_func(
            &array.data,
            scalar,
            temp.data_mut()
                .mutate()
                .clear_and_resize(array.len())
                .as_mut_ptr(),
        );
        and_inplace(selection, temp.data());
    }
}

/// # Safety
///
/// - If the `selection` is not empty, `lhs`/`rhs` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
///
/// - `lhs`/`rhs`'s data and validity should not reference `temp`'s data and validity. In the computation
/// graph, `lhs`/`rhs` must be the descendant of `temp`
///
/// - No other arrays that reference the `temp`'s data and validity are accessed! In the
/// computation graph, it will never happens
#[inline]
unsafe fn cmp<T: PartialOrdExt>(
    selection: &mut Bitmap,
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<T>,
    temp: &mut BooleanArray,
    partial_arith_threshold: f64,
    cmp_func: unsafe fn(&AlignedVec<T>, &AlignedVec<T>, *mut BitStore),
    cmp_scalars_func: impl Fn(T, T) -> bool,
) where
    PrimitiveArray<T>: Array<Element = T>,
{
    debug_assert_selection_is_valid!(selection, lhs);
    debug_assert_eq!(lhs.len(), rhs.len());
    // Benchmark shows the two and_inplace function cost 50% of the time 😭

    and_inplace(selection, lhs.validity());
    and_inplace(selection, rhs.validity());
    if selection.ones_ratio() < partial_arith_threshold {
        selection.mutate().mutate_ones(|index| {
            cmp_scalars_func(
                lhs.get_value_unchecked(index),
                rhs.get_value_unchecked(index),
            )
        })
    } else {
        // Benchmark shows clear_and_resize may cost lots of time
        cmp_func(
            &lhs.data,
            &rhs.data,
            temp.data_mut()
                .mutate()
                .clear_and_resize(lhs.len())
                .as_mut_ptr(),
        );
        and_inplace(selection, temp.data());
    }
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
                temp: &mut BooleanArray,
            ){
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    if std::arch::is_x86_feature_detected!("avx2") {
                        return cmp_scalar(selection, array, scalar, temp, Self::AVX2_PARTIAL_CMP_THRESHOLD, [<$cmp_func _scalar_ $ty _avx2>], $cmp_func);
                    } else {
                        return cmp_scalar(selection, array, scalar, temp, Self::PARTIAL_CMP_THRESHOLD, [<$cmp_func _scalar_ $ty _v2>], $cmp_func);
                    }
                }

                #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
                {
                    if std::arch::is_aarch64_feature_detected!("neon") {
                        return cmp_scalar(selection, array, scalar, temp, Self::NEON_PARTIAL_CMP_THRESHOLD, [<$cmp_func _scalar_ $ty _neon>], $cmp_func);
                    }
                }

                #[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), allow(unreachable_code))]
                {
                    cmp_scalar_default(selection, array, scalar, temp, $ty::PARTIAL_CMP_THRESHOLD, $cmp_func);
                }
            }

            unsafe fn $cmp_func(
                selection: &mut Bitmap,
                lhs: &PrimitiveArray<$ty>,
                rhs: &PrimitiveArray<$ty>,
                temp: &mut BooleanArray,
            ){
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    if std::arch::is_x86_feature_detected!("avx2") {
                        return cmp(selection, lhs, rhs, temp, Self::AVX2_PARTIAL_CMP_THRESHOLD, [<$cmp_func _ $ty _avx2>], $cmp_func);
                    } else {
                        return cmp(selection, lhs, rhs, temp, Self::PARTIAL_CMP_THRESHOLD, [<$cmp_func _ $ty _v2>], $cmp_func);
                    }
                }

                #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
                {
                    if std::arch::is_aarch64_feature_detected!("neon") {
                        return cmp(selection, lhs, rhs, temp, Self::NEON_PARTIAL_CMP_THRESHOLD, [<$cmp_func _ $ty _neon>], $cmp_func);
                    }
                }

                #[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), allow(unreachable_code))]
                {
                    cmp_default(selection, lhs, rhs, temp, $ty::PARTIAL_CMP_THRESHOLD, $cmp_func);
                }
            }
        }
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

/// If the selectivity is smaller than this threshold, partial computation is used
const TIMESTAMP_PARTIAL_CMP_THRESHOLD: f64 = 0.0;
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
/// If the selectivity is smaller than this threshold, partial computation is used
const TIMESTAMP_NEON_PARTIAL_CMP_THRESHOLD: f64 = 0.0;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
/// If the selectivity is smaller than this threshold, partial computation is used
const TIMESTAMP_AVX2_PARTIAL_CMP_THRESHOLD: f64 = 0.0;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
/// If the selectivity is smaller than this threshold, partial computation is used
const TIMESTAMP_AVX512_PARTIAL_CMP_THRESHOLD: f64 = 0.0;

macro_rules! timestamp_cmp_scalar {
    ($cmp:ident, $cmp_symbol:tt) => {
        paste::paste! {
            #[doc = concat!("Perform `array", stringify!(cmp_symbol), "scalar` between `PrimitiveArray<i64>`, with logical type")]
            /// `Timestamp`/`Timestamptz`, and `i64`
            ///
            /// # Safety
            ///
            /// - If the `selection` is not empty, `array` and `selection` should have same length.
            /// Otherwise, undefined behavior happens
            ///
            /// - `selection` should not be referenced by any array
            ///
            /// - `array`'s data and validity should not reference `temp`'s data and validity. In the computation
            /// graph, `lhs` must be the descendant of `temp`
            ///
            /// - No other arrays that reference the `temp`'s data and validity are accessed! In the
            /// computation graph, it will never happens
            pub unsafe fn [<timestamp_ $cmp _scalar>]<const AM: i64, const SM: i64>(
                selection: &mut Bitmap,
                array: &PrimitiveArray<i64>,
                scalar: i64,
                temp: &mut BooleanArray,
            ) {
                #[cfg(debug_assertions)]
                super::check_timestamp_array_and_multiplier::<AM>(array);

                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    if std::arch::is_x86_feature_detected!("avx2") {
                        return cmp_scalar(
                            selection,
                            array,
                            scalar,
                            temp,
                            TIMESTAMP_AVX2_PARTIAL_CMP_THRESHOLD,
                            [<timestamp_ $cmp _scalar_avx2>]::<AM, SM>,
                            [<timestamp_scalar_ $cmp _scalar>]::<AM, SM>,
                        );
                    } else {
                        return cmp_scalar(
                            selection,
                            array,
                            scalar,
                            temp,
                            TIMESTAMP_PARTIAL_CMP_THRESHOLD,
                            [<timestamp_ $cmp _scalar_v2>]::<AM, SM>,
                            [<timestamp_scalar_ $cmp _scalar>]::<AM, SM>,
                        );
                    }
                }


                #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
                {
                    if std::arch::is_aarch64_feature_detected!("neon") {
                        return cmp_scalar(
                            selection,
                            array,
                            scalar,
                            temp,
                            TIMESTAMP_NEON_PARTIAL_CMP_THRESHOLD,
                            [<timestamp_ $cmp _scalar_neon>]::<AM, SM>,
                            [<timestamp_scalar_ $cmp _scalar>]::<AM, SM>,
                        );
                    }
                }

                #[cfg_attr(
                    any(target_arch = "x86", target_arch = "x86_64"),
                    allow(unreachable_code)
                )]
                {
                    cmp_scalar_default(
                        selection,
                        array,
                        scalar,
                        temp,
                        TIMESTAMP_PARTIAL_CMP_THRESHOLD,
                        [<timestamp_scalar_ $cmp _scalar>]::<AM, SM>,
                    );
                }
            }


            /// # Safety
            ///
            /// - If the `selection` is not empty, `lhs`/`rhs` and `selection` should have same length.
            /// Otherwise, undefined behavior happens
            ///
            /// - `selection` should not be referenced by any array
            ///
            /// - `lhs`/`rhs`'s data and validity should not reference `temp`'s data and validity. In the computation
            /// graph, `lhs`/`rhs` must be the descendant of `temp`
            ///
            /// - No other arrays that reference the `temp`'s data and validity are accessed! In the
            /// computation graph, it will never happens
            pub unsafe fn [<timestamp_ $cmp>]<const LM: i64, const RM: i64>(
                selection: &mut Bitmap,
                lhs: &PrimitiveArray<i64>,
                rhs: &PrimitiveArray<i64>,
                temp: &mut BooleanArray,
            ) {
                #[cfg(debug_assertions)]
                {
                    super::check_timestamp_array_and_multiplier::<LM>(lhs);
                    super::check_timestamp_array_and_multiplier::<RM>(rhs);
                }

                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    if std::arch::is_x86_feature_detected!("avx2") {
                        return cmp(
                            selection,
                            lhs,
                            rhs,
                            temp,
                            TIMESTAMP_AVX2_PARTIAL_CMP_THRESHOLD,
                            [<timestamp_ $cmp _avx2>]::<LM, RM>,
                            [<timestamp_scalar_ $cmp _scalar>]::<LM, RM>,
                        );
                    } else {
                        return cmp_scalar(
                            selection,
                            lhs,
                            rhs,
                            temp,
                            TIMESTAMP_PARTIAL_CMP_THRESHOLD,
                            [<timestamp_ $cmp _v2>]::<LM, RM>,
                            [<timestamp_scalar_ $cmp _scalar>]::<LM, RM>,
                        );
                    }
                }

                #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
                {
                    if std::arch::is_aarch64_feature_detected!("neon") {
                        return cmp(
                            selection,
                            lhs,
                            rhs,
                            temp,
                            TIMESTAMP_NEON_PARTIAL_CMP_THRESHOLD,
                            [<timestamp_ $cmp _neon>]::<LM, RM>,
                            [<timestamp_scalar_ $cmp _scalar>]::<LM, RM>,
                        );
                    }
                }

                #[cfg_attr(
                    any(target_arch = "x86", target_arch = "x86_64"),
                    allow(unreachable_code)
                )]
                {
                    cmp_default(
                        selection,
                        lhs,
                        rhs,
                        temp,
                        TIMESTAMP_PARTIAL_CMP_THRESHOLD,
                        [<timestamp_scalar_ $cmp _scalar>]::<LM, RM>,
                    );
                }
            }
        }
    };
}

timestamp_cmp_scalar!(eq, ==);
timestamp_cmp_scalar!(ne, !=);
timestamp_cmp_scalar!(gt, >);
timestamp_cmp_scalar!(ge, >=);
timestamp_cmp_scalar!(lt, <);
timestamp_cmp_scalar!(le, <=);
