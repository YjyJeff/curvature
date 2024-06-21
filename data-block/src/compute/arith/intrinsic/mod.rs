//! Arithmetic on IntrinsicType

mod add;
mod div;
mod mul;
mod rem;
mod sub;

use crate::array::{Array, PrimitiveArray};
use crate::bitmap::Bitmap;
use crate::compute::combine_validities;
use crate::types::IntrinsicType;
pub use add::{add, add_scalar, AddExt};
pub use div::{div, div_scalar, DivExt};
pub use mul::{mul, mul_scalar, MulExt};
pub use rem::{rem, rem_scalar, RemCast, RemExt};
pub use sub::{sub, sub_scalar, SubExt};

// Add, Sub, Mul operations are supported by cpu, benchmark shows that they usually
// have same threshold based on number of bits
const NATIVE_PARTIAL_ARITH_THRESHOLD_8_BIT: f64 = 0.013;
const NATIVE_PARTIAL_ARITH_THRESHOLD_16_BIT: f64 = 0.026;
const NATIVE_PARTIAL_ARITH_THRESHOLD_32_BIT: f64 = 0.05;
const NATIVE_PARTIAL_ARITH_THRESHOLD_64_BIT: f64 = 0.1;

/// Generic array arith scalar function
#[inline]
unsafe fn arith_scalar<T, U, V>(
    selection: &Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    dst: &mut PrimitiveArray<V>,
    partial_arith_threshold: f64,
    arith_func: impl Fn(T, U) -> V,
    transformer: impl Fn(T) -> U,
) where
    PrimitiveArray<T>: Array,
    T: IntrinsicType,
    V: IntrinsicType,
    U: Copy,
{
    debug_assert_selection_is_valid!(selection, array);
    dst.validity.reference(&array.validity);

    let dst = dst.data.as_mut();
    let uninitialized = dst.clear_and_resize(array.len());

    let array_data = array.data.as_slice();
    let scalar = transformer(scalar);

    if selection.ones_ratio() < partial_arith_threshold {
        selection.iter_ones().for_each(|index| {
            *uninitialized.get_unchecked_mut(index) =
                arith_func(*array_data.get_unchecked(index), scalar);
        });
    } else {
        // Following code will be optimized for different target-feature to perform
        // auto-vectorization
        array_data
            .iter()
            .zip(uninitialized)
            .for_each(|(&element, uninit)| *uninit = arith_func(element, scalar));
    }
}

/// Arithmetic between two arrays
///
/// Allow `into_iter_on_ref` here because rust analyzer has bug. If we use `iter` instead of
/// `into_iter` in the block, the analyzer will tell us an error here. However, the code can
/// compile!
#[inline]
#[allow(clippy::into_iter_on_ref)]
unsafe fn arith_arrays<T, U, V>(
    selection: &Bitmap,
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<U>,
    dst: &mut PrimitiveArray<V>,
    partial_arith_threshold: f64,
    arith_func: impl Fn(T, U) -> V,
) where
    PrimitiveArray<T>: Array,
    PrimitiveArray<U>: Array,
    PrimitiveArray<V>: Array,
    T: IntrinsicType,
    V: IntrinsicType,
    U: IntrinsicType,
{
    debug_assert_selection_is_valid!(selection, lhs);
    debug_assert_eq!(lhs.len(), rhs.len());

    combine_validities(&lhs.validity, &rhs.validity, &mut dst.validity);

    let uninitialized = dst.data.as_mut().clear_and_resize(lhs.len());

    let lhs = lhs.data.as_slice();
    let rhs = rhs.data.as_slice();

    if selection.ones_ratio() < partial_arith_threshold {
        selection.iter_ones().for_each(|index| {
            *uninitialized.get_unchecked_mut(index) =
                arith_func(*lhs.get_unchecked(index), *rhs.get_unchecked(index));
        });
    } else {
        // Following code will be optimized for different target-feature to perform
        // auto-vectorization
        lhs.into_iter()
            .zip(rhs)
            .zip(uninitialized)
            .for_each(|((&lhs, &rhs), uninit)| *uninit = arith_func(lhs, rhs));
    }
}

macro_rules! dynamic_arith_scalar_func {
    (#[$doc:meta], $func_name:ident, $ext_trait:ident, $arith_func:path, $transformer:path) => {
        paste::paste! {
            #[$doc]
            ///
            /// # Safety
            ///
            /// - If the `selection` is not empty, `array` and `selection` should have same length.
            /// Otherwise, undefined behavior happens
            ///
            /// - `lhs` data and validity should not reference `dst`'s data and validity. In the computation
            /// graph, `lhs` must be the descendant of `dst`
            ///
            /// - No other arrays that reference the `dst`'s data and validity are accessed! In the
            /// computation graph, it will never happens
            pub unsafe fn $func_name<T>(
                selection: &Bitmap,
                array: &PrimitiveArray<T>,
                scalar: T,
                dst: &mut PrimitiveArray<T>,
            ) where
                PrimitiveArray<T>: Array,
                T: $ext_trait,
            {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    // Note that this `unsafe` block is safe because we're testing
                    // that the `avx512` feature is indeed available on our CPU.
                    #[cfg(feature = "avx512")]
                    if std::arch::is_x86_feature_detected!("avx512f") {
                        return [<$func_name _avx512>](selection, array, scalar, dst);
                    }

                    // Note that this `unsafe` block is safe because we're testing
                    // that the `avx2` feature is indeed available on our CPU.
                    if std::arch::is_x86_feature_detected!("avx2") {
                        return [<$func_name _avx2>](selection, array, scalar, dst);
                    }
                }

                #[cfg(target_arch = "aarch64")]
                {
                    // Note that this `unsafe` block is safe because we're testing
                    // that the `neon` feature is indeed available on our CPU.
                    if std::arch::is_aarch64_feature_detected!("neon") {
                        return [<$func_name _neon>](selection, array, scalar, dst);
                    }
                }

                arith_scalar(
                    selection,
                    array,
                    scalar,
                    dst,
                    T::PARTIAL_ARITH_THRESHOLD,
                    $arith_func,
                    $transformer,
                )
            }
        }
        dynamic_arith_scalar_func!(#[cfg(target_arch = "aarch64")], "neon", neon, $func_name, $ext_trait, $arith_func, $transformer);
        dynamic_arith_scalar_func!(#[cfg(any(target_arch = "x86", target_arch = "x86_64"))], "avx2", avx2, $func_name, $ext_trait, $arith_func, $transformer);
        #[cfg(feature = "avx512")]
        dynamic_arith_scalar_func!(#[cfg(any(target_arch = "x86", target_arch = "x86_64"))], "avx512f", avx512, $func_name, $ext_trait, $arith_func, $transformer);
    };
    (#[$target_arch:meta], $target_feature:expr, $func_suffix:ident, $func_name:ident, $ext_trait:ident, $arith_func:path, $transformer:path) => {
        // Generate target specific function
        paste::paste! {
            #[$target_arch]
            #[target_feature(enable = $target_feature)]
            #[inline]
            unsafe fn [<$func_name _ $func_suffix>]<T>(
                selection: &Bitmap,
                array: &PrimitiveArray<T>,
                scalar: T,
                dst: &mut PrimitiveArray<T>,
            ) where
                PrimitiveArray<T>: Array,
                T: $ext_trait,
            {
                arith_scalar(
                    selection,
                    array,
                    scalar,
                    dst,
                    T::[<$func_suffix:upper _PARTIAL_ARITH_THRESHOLD>],
                    $arith_func,
                    $transformer,
                )
            }
        }
    };
}

macro_rules! dynamic_arith_arrays_func {
    (#[$doc:meta], $func_name:ident, $ext_trait:ident, $arith_func:path) => {
        paste::paste! {
            #[$doc]
            ///
            /// # Safety
            ///
            /// - If the `selection` is not empty, `array` and `selection` should have same length.
            /// Otherwise, undefined behavior happens
            ///
            /// - `lhs` data and validity should not reference `dst`'s data and validity. In the computation
            /// graph, `lhs` must be the descendant of `dst`
            ///
            /// - No other arrays that reference the `dst`'s data and validity are accessed! In the
            /// computation graph, it will never happens
            pub unsafe fn $func_name<T>(
                selection: &Bitmap,
                lhs: &PrimitiveArray<T>,
                rhs: &PrimitiveArray<T>,
                dst: &mut PrimitiveArray<T>,
            ) where
                PrimitiveArray<T>: Array,
                T: $ext_trait,
            {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    // Note that this `unsafe` block is safe because we're testing
                    // that the `avx512` feature is indeed available on our CPU.
                    #[cfg(feature = "avx512")]
                    if std::arch::is_x86_feature_detected!("avx512f") {
                        return [<$func_name _avx512>](selection, lhs, rhs, dst);
                    }

                    // Note that this `unsafe` block is safe because we're testing
                    // that the `avx2` feature is indeed available on our CPU.
                    if std::arch::is_x86_feature_detected!("avx2") {
                        return [<$func_name _avx2>](selection, lhs, rhs, dst);
                    }
                }

                #[cfg(target_arch = "aarch64")]
                {
                    // Note that this `unsafe` block is safe because we're testing
                    // that the `neon` feature is indeed available on our CPU.
                    if std::arch::is_aarch64_feature_detected!("neon") {
                        return [<$func_name _neon>](selection, lhs, rhs, dst);
                    }
                }

                arith_arrays(
                    selection,
                    lhs,
                    rhs,
                    dst,
                    T::PARTIAL_ARITH_THRESHOLD,
                    $arith_func,
                )
            }
        }
        dynamic_arith_arrays_func!(#[cfg(target_arch = "aarch64")], "neon", neon, $func_name, $ext_trait, $arith_func);
        dynamic_arith_arrays_func!(#[cfg(any(target_arch = "x86", target_arch = "x86_64"))], "avx2", avx2, $func_name, $ext_trait, $arith_func);
        #[cfg(feature = "avx512")]
        dynamic_arith_arrays_func!(#[cfg(any(target_arch = "x86", target_arch = "x86_64"))], "avx512f", avx512, $func_name, $ext_trait, $arith_func);
    };
    (#[$target_arch:meta], $target_feature:expr, $func_suffix:ident, $func_name:ident, $ext_trait:ident, $arith_func:path) => {
        // Generate target specific function
        paste::paste! {
            #[$target_arch]
            #[target_feature(enable = $target_feature)]
            #[inline]
            unsafe fn [<$func_name _ $func_suffix>]<T>(
                selection: &Bitmap,
                lhs: &PrimitiveArray<T>,
                rhs: &PrimitiveArray<T>,
                dst: &mut PrimitiveArray<T>,
            ) where
                PrimitiveArray<T>: Array,
                T: $ext_trait,
            {
                arith_arrays(
                    selection,
                    lhs,
                    rhs,
                    dst,
                    T::[<$func_suffix:upper _PARTIAL_ARITH_THRESHOLD>],
                    $arith_func,
                )
            }
        }
    };
}

use dynamic_arith_arrays_func;
use dynamic_arith_scalar_func;
