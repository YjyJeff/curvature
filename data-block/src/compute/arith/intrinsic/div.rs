use std::ops::Div;

use super::{arith_arrays, arith_scalar};
use crate::array::{Array, PrimitiveArray};
use crate::bitmap::Bitmap;
use crate::types::IntrinsicType;

use strength_reduce::{
    StrengthReducedU16, StrengthReducedU32, StrengthReducedU64, StrengthReducedU8,
};

/// Div extension
pub trait DivExt: IntrinsicType + Div<Self::Divisor, Output = Self> + Div<Output = Self> {
    /// If the selectivity is smaller than this threshold, partial computation is used
    const PARTIAL_ARITH_THRESHOLD: f64;
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    /// If the selectivity is smaller than this threshold, partial computation is used
    const NEON_PARTIAL_ARITH_THRESHOLD: f64;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// If the selectivity is smaller than this threshold, partial computation is used
    const AVX2_PARTIAL_ARITH_THRESHOLD: f64;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// If the selectivity is smaller than this threshold, partial computation is used
    const AVX512_PARTIAL_ARITH_THRESHOLD: f64;

    /// Divisor that can accelerate div
    type Divisor: Copy;

    /// Create a new divisor from self
    fn new_divisor(self) -> Self::Divisor;
}

macro_rules! impl_div_ext_for_float {
    ($ty:ty, $default_th:expr, $neon_th:expr, $avx2_th:expr, $avx512_th:expr) => {
        impl DivExt for $ty {
            /// If the selectivity is smaller than this threshold, partial computation is used
            const PARTIAL_ARITH_THRESHOLD: f64 = $default_th;
            #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
            /// If the selectivity is smaller than this threshold, partial computation is used
            const NEON_PARTIAL_ARITH_THRESHOLD: f64 = $neon_th;
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            /// If the selectivity is smaller than this threshold, partial computation is used
            const AVX2_PARTIAL_ARITH_THRESHOLD: f64 = $avx2_th;
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            /// If the selectivity is smaller than this threshold, partial computation is used
            const AVX512_PARTIAL_ARITH_THRESHOLD: f64 = $avx512_th;

            type Divisor = $ty;

            #[inline]
            fn new_divisor(self) -> Self::Divisor {
                self
            }
        }
    };
}

// FIXME: Tune the parameter
impl_div_ext_for_float!(f32, 0.08, 0.08, 0.0, 0.0);
impl_div_ext_for_float!(f64, 0.16, 0.016, 0.0, 0.0);

macro_rules! impl_div_ext_for_unsigned_int {
    ($ty:ty, $d_ty:ty, $default_th:expr, $neon_th:expr, $avx2_th:expr, $avx512_th:expr) => {
        impl DivExt for $ty {
            /// If the selectivity is smaller than this threshold, partial computation is used
            const PARTIAL_ARITH_THRESHOLD: f64 = $default_th;
            #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
            /// If the selectivity is smaller than this threshold, partial computation is used
            const NEON_PARTIAL_ARITH_THRESHOLD: f64 = $neon_th;
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            /// If the selectivity is smaller than this threshold, partial computation is used
            const AVX2_PARTIAL_ARITH_THRESHOLD: f64 = $avx2_th;
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            /// If the selectivity is smaller than this threshold, partial computation is used
            const AVX512_PARTIAL_ARITH_THRESHOLD: f64 = $avx512_th;

            type Divisor = $d_ty;

            #[inline]
            fn new_divisor(self) -> Self::Divisor {
                <$d_ty>::new(self)
            }
        }
    };
}

// FIXME: Tune the parameter
impl_div_ext_for_unsigned_int!(u8, StrengthReducedU8, 0.05, 0.05, 0.0, 0.0);
impl_div_ext_for_unsigned_int!(u16, StrengthReducedU16, 0.1, 0.1, 0.0, 0.0);
impl_div_ext_for_unsigned_int!(u32, StrengthReducedU32, 0.2, 0.2, 0.0, 0.0);
impl_div_ext_for_unsigned_int!(u64, StrengthReducedU64, 0.3, 0.3, 0.0, 0.0);

/// Divisor of signed integer
#[derive(Debug, Clone, Copy)]
pub struct SignedIntegerDivisor<T: Clone + Copy> {
    neg: bool,
    divisor: T,
}

macro_rules! impl_div_ext_for_signed_int {
    ($ty:ty, $uty:ty, $d_ty:ty, $default_th:expr, $neon_th:expr, $avx2_th:expr, $avx512_th:expr) => {
        impl DivExt for $ty {
            /// If the selectivity is smaller than this threshold, partial computation is used
            const PARTIAL_ARITH_THRESHOLD: f64 = $default_th;
            #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
            /// If the selectivity is smaller than this threshold, partial computation is used
            const NEON_PARTIAL_ARITH_THRESHOLD: f64 = $neon_th;
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            /// If the selectivity is smaller than this threshold, partial computation is used
            const AVX2_PARTIAL_ARITH_THRESHOLD: f64 = $avx2_th;
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            /// If the selectivity is smaller than this threshold, partial computation is used
            const AVX512_PARTIAL_ARITH_THRESHOLD: f64 = $avx512_th;

            type Divisor = SignedIntegerDivisor<$d_ty>;

            #[inline]
            fn new_divisor(self) -> Self::Divisor {
                if self > 0 {
                    SignedIntegerDivisor {
                        neg: false,
                        divisor: <$d_ty>::new(self as $uty),
                    }
                } else {
                    SignedIntegerDivisor {
                        neg: true,
                        divisor: <$d_ty>::new((-self) as $uty),
                    }
                }
            }
        }

        impl Div<SignedIntegerDivisor<$d_ty>> for $ty {
            type Output = $ty;

            #[inline]
            fn div(self, rhs: SignedIntegerDivisor<$d_ty>) -> Self::Output {
                let (result, neg) = if self > 0 {
                    (self as $uty / rhs.divisor, false)
                } else {
                    ((-self) as $uty / rhs.divisor, true)
                };

                if rhs.neg ^ neg {
                    -(result as $ty)
                } else {
                    result as $ty
                }
            }
        }
    };
}

// FIXME: Tune the parameter
impl_div_ext_for_signed_int!(i8, u8, StrengthReducedU8, 0.05, 0.05, 0.0, 0.0);
impl_div_ext_for_signed_int!(i16, u16, StrengthReducedU16, 0.1, 0.1, 0.0, 0.0);
impl_div_ext_for_signed_int!(i32, u32, StrengthReducedU32, 0.2, 0.2, 0.0, 0.0);
impl_div_ext_for_signed_int!(i64, u64, StrengthReducedU64, 0.3, 0.3, 0.0, 0.0);

/// Perform `array / scalar` for `PrimitiveArray<T>` with `T`
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
pub unsafe fn div_scalar<T>(
    selection: &Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    dst: &mut PrimitiveArray<T>,
) where
    PrimitiveArray<T>: Array,
    T: DivExt,
{
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // Note that this `unsafe` block is safe because we're testing
        // that the `avx512` feature is indeed available on our CPU.
        #[cfg(feature = "avx512")]
        if std::arch::is_x86_feature_detected!("avx512f") {
            return div_scalar_avx512(selection, array, scalar, dst);
        }

        // Note that this `unsafe` block is safe because we're testing
        // that the `avx2` feature is indeed available on our CPU.
        if std::arch::is_x86_feature_detected!("avx2") {
            return div_scalar_avx2(selection, array, scalar, dst);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Note that this `unsafe` block is safe because we're testing
        // that the `neon` feature is indeed available on our CPU.
        if std::arch::is_aarch64_feature_detected!("neon") {
            return div_scalar_neon(selection, array, scalar, dst);
        }
    }

    arith_scalar(
        selection,
        array,
        scalar,
        dst,
        T::PARTIAL_ARITH_THRESHOLD,
        Div::div,
        T::new_divisor,
    )
}

#[cfg(feature = "avx512")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn div_scalar_avx512<T>(
    selection: &Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    dst: &mut PrimitiveArray<T>,
) where
    PrimitiveArray<T>: Array,
    T: DivExt,
{
    arith_scalar(
        selection,
        array,
        scalar,
        dst,
        T::AVX512_PARTIAL_ARITH_THRESHOLD,
        Div::div,
        T::new_divisor,
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn div_scalar_avx2<T>(
    selection: &Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    dst: &mut PrimitiveArray<T>,
) where
    PrimitiveArray<T>: Array,
    T: DivExt,
{
    arith_scalar(
        selection,
        array,
        scalar,
        dst,
        T::AVX2_PARTIAL_ARITH_THRESHOLD,
        Div::div,
        T::new_divisor,
    )
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn div_scalar_neon<T>(
    selection: &Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    dst: &mut PrimitiveArray<T>,
) where
    PrimitiveArray<T>: Array,
    T: DivExt,
{
    arith_scalar(
        selection,
        array,
        scalar,
        dst,
        T::NEON_PARTIAL_ARITH_THRESHOLD,
        Div::div,
        T::new_divisor,
    )
}

/// Perform `/` between two `PrimitiveArray<T>`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `lhs` and `rhs` must have same length
///
/// - `lhs`/`rhs` data and validity should not reference `dst`'s data and validity. In the computation
/// graph, `lhs`/`rhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn div<T>(
    selection: &Bitmap,
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<T>,
    dst: &mut PrimitiveArray<T>,
) where
    PrimitiveArray<T>: Array,
    T: DivExt,
{
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // Note that this `unsafe` block is safe because we're testing
        // that the `avx512` feature is indeed available on our CPU.
        #[cfg(feature = "avx512")]
        if std::arch::is_x86_feature_detected!("avx512f") {
            return div_avx512(selection, lhs, rhs, dst);
        }

        // Note that this `unsafe` block is safe because we're testing
        // that the `avx2` feature is indeed available on our CPU.
        if std::arch::is_x86_feature_detected!("avx2") {
            return div_avx2(selection, lhs, rhs, dst);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Note that this `unsafe` block is safe because we're testing
        // that the `neon` feature is indeed available on our CPU.
        if std::arch::is_aarch64_feature_detected!("neon") {
            return div_neon(selection, lhs, rhs, dst);
        }
    }

    arith_arrays(
        selection,
        lhs,
        rhs,
        dst,
        T::PARTIAL_ARITH_THRESHOLD,
        Div::div,
    )
}

#[cfg(feature = "avx512")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn div_avx512<T>(
    selection: &Bitmap,
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<T>,
    dst: &mut PrimitiveArray<T>,
) where
    PrimitiveArray<T>: Array,
    T: DivExt,
{
    arith_arrays(
        selection,
        lhs,
        rhs,
        dst,
        T::AVX512_PARTIAL_ARITH_THRESHOLD,
        Div::div,
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn div_avx2<T>(
    selection: &Bitmap,
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<T>,
    dst: &mut PrimitiveArray<T>,
) where
    PrimitiveArray<T>: Array,
    T: DivExt,
{
    arith_arrays(
        selection,
        lhs,
        rhs,
        dst,
        T::AVX2_PARTIAL_ARITH_THRESHOLD,
        Div::div,
    )
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn div_neon<T>(
    selection: &Bitmap,
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<T>,
    dst: &mut PrimitiveArray<T>,
) where
    PrimitiveArray<T>: Array,
    T: DivExt,
{
    arith_arrays(
        selection,
        lhs,
        rhs,
        dst,
        T::NEON_PARTIAL_ARITH_THRESHOLD,
        Div::div,
    )
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::types::LogicalType;

    #[test]
    fn test_div_scalar() {
        let lhs = PrimitiveArray::from_values_iter([10, -7, -3, -9, -20, 87, 100, 34, 65]);
        let rhs = -2;
        let mut dst = PrimitiveArray::new(LogicalType::Integer).unwrap();
        unsafe { div_scalar(&Bitmap::new(), &lhs, rhs, &mut dst) };
        assert_eq!(dst.values(), &[-5, 3, 1, 4, 10, -43, -50, -17, -32]);
    }
}
