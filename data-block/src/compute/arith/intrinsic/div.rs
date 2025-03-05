use std::ops::Div;

use super::{arith_arrays, arith_scalar, dynamic_arith_arrays_func, dynamic_arith_scalar_func};
use crate::array::{Array, PrimitiveArray};
use crate::bitmap::Bitmap;
use crate::types::IntrinsicType;

use strength_reduce::{
    StrengthReducedU8, StrengthReducedU16, StrengthReducedU32, StrengthReducedU64,
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

dynamic_arith_scalar_func!(#[doc = " Perform `array / scalar` between `PrimitiveArray<T>` and T"], div_scalar, DivExt, Div::div, T::new_divisor);
dynamic_arith_arrays_func!(#[doc = " Perform `/` between two `PrimitiveArray<T>`"], div, DivExt, Div::div);

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
