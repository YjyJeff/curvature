use std::ops::Rem;

use super::{arith_arrays, arith_scalar};
use crate::array::{Array, PrimitiveArray};
use crate::bitmap::Bitmap;
use crate::types::IntrinsicType;

use strength_reduce::{
    StrengthReducedU16, StrengthReducedU32, StrengthReducedU64, StrengthReducedU8,
};

/// Trait for casting used in rem
pub trait RemCast<T: RemExt>: IntrinsicType {
    /// Cast self to T that can perform Rem
    fn cast(self) -> T;

    /// Cast back T to self
    fn cast_back(val: T) -> Self;
}

macro_rules! impl_rem_cast {
    ($ty:ty, {$($cast_ty:ty),+}) => {
        $(
            impl RemCast<$cast_ty> for $ty {
                #[inline]
                fn cast(self) -> $cast_ty {
                    self as $cast_ty
                }

                #[inline]
                fn cast_back(val: $cast_ty) -> Self{
                    val as $ty
                }
            }
        )+
    };
}

impl_rem_cast!(u8, {u8, u16, u32, u64});
impl_rem_cast!(u16, {u16, u32, u64});
impl_rem_cast!(u32, {u32, u64});
impl_rem_cast!(u64, { u64 });
impl_rem_cast!(i8, {i8, i16, i32, i64});
impl_rem_cast!(i16, {i16, i32, i64});
impl_rem_cast!(i32, {i32, i64});
impl_rem_cast!(i64, { i64 });
impl_rem_cast!(f32, {f32, f64});
impl_rem_cast!(f64, { f64 });

/// Extent remainder
pub trait RemExt: IntrinsicType + Rem<Self::Remainder, Output = Self> + Rem<Output = Self> {
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

    /// Remainder that can accelerate rem
    type Remainder: Copy;
    /// Create a new remainder from self
    fn new_remainder(self) -> Self::Remainder;
}

macro_rules! impl_rem_ext_for_float {
    ($ty:ty, $default_th:expr, $neon_th:expr, $avx2_th:expr, $avx512_th:expr) => {
        impl RemExt for $ty {
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

            type Remainder = $ty;

            #[inline]
            fn new_remainder(self) -> Self::Remainder {
                self
            }
        }
    };
}

// FIXME: Tune the parameter
impl_rem_ext_for_float!(f32, 0.08, 0.08, 0.0, 0.0);
impl_rem_ext_for_float!(f64, 0.16, 0.016, 0.0, 0.0);

macro_rules! impl_rem_ext_for_unsigned_int {
    ($ty:ty, $d_ty:ty, $default_th:expr, $neon_th:expr, $avx2_th:expr, $avx512_th:expr) => {
        impl RemExt for $ty {
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

            type Remainder = $d_ty;

            #[inline]
            fn new_remainder(self) -> Self::Remainder {
                <$d_ty>::new(self)
            }
        }
    };
}

// FIXME: Tune the parameter
impl_rem_ext_for_unsigned_int!(u8, StrengthReducedU8, 0.05, 0.05, 0.0, 0.0);
impl_rem_ext_for_unsigned_int!(u16, StrengthReducedU16, 0.1, 0.1, 0.0, 0.0);
impl_rem_ext_for_unsigned_int!(u32, StrengthReducedU32, 0.2, 0.2, 0.0, 0.0);
impl_rem_ext_for_unsigned_int!(u64, StrengthReducedU64, 0.3, 0.3, 0.0, 0.0);

/// Remainder of signed integer
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct SignedIntegerRemainder<T: Clone + Copy>(T);

macro_rules! impl_rem_ext_for_signed_int {
    ($ty:ty, $uty:ty, $d_ty:ty, $default_th:expr, $neon_th:expr, $avx2_th:expr, $avx512_th:expr) => {
        impl RemExt for $ty {
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

            type Remainder = SignedIntegerRemainder<$d_ty>;

            #[inline]
            fn new_remainder(self) -> Self::Remainder {
                if self > 0 {
                    SignedIntegerRemainder(<$d_ty>::new(self as $uty))
                } else {
                    SignedIntegerRemainder(<$d_ty>::new((-self) as $uty))
                }
            }
        }

        impl Rem<SignedIntegerRemainder<$d_ty>> for $ty {
            type Output = $ty;

            #[inline]
            fn rem(self, rhs: SignedIntegerRemainder<$d_ty>) -> Self::Output {
                if self > 0 {
                    (self as $uty % rhs.0) as $ty
                } else {
                    -(((-self) as $uty % rhs.0) as $ty)
                }
            }
        }
    };
}

// FIXME: Tune the parameter
impl_rem_ext_for_signed_int!(i8, u8, StrengthReducedU8, 0.05, 0.05, 0.0, 0.0);
impl_rem_ext_for_signed_int!(i16, u16, StrengthReducedU16, 0.1, 0.1, 0.0, 0.0);
impl_rem_ext_for_signed_int!(i32, u32, StrengthReducedU32, 0.2, 0.2, 0.0, 0.0);
impl_rem_ext_for_signed_int!(i64, u64, StrengthReducedU64, 0.3, 0.3, 0.0, 0.0);

#[inline]
fn rem_scalar_func<T: RemExt, U: RemCast<T>>(lhs: T, rhs: T::Remainder) -> U {
    U::cast_back(lhs % rhs)
}

/// Perform `array % scalar` for `PrimitiveArray<T>` with `T`
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
pub unsafe fn rem_scalar<T, U>(
    selection: &Bitmap,
    array: &PrimitiveArray<T>,
    scalar: U,
    dst: &mut PrimitiveArray<U>,
) where
    PrimitiveArray<T>: Array,
    PrimitiveArray<U>: Array,
    T: RemExt,
    U: RemCast<T>,
{
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // Note that this `unsafe` block is safe because we're testing
        // that the `avx512` feature is indeed available on our CPU.
        #[cfg(feature = "avx512")]
        if std::arch::is_x86_feature_detected!("avx512f") {
            return rem_scalar_avx512(selection, array, scalar, dst);
        }

        // Note that this `unsafe` block is safe because we're testing
        // that the `avx2` feature is indeed available on our CPU.
        if std::arch::is_x86_feature_detected!("avx2") {
            return rem_scalar_avx2(selection, array, scalar, dst);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Note that this `unsafe` block is safe because we're testing
        // that the `neon` feature is indeed available on our CPU.
        if std::arch::is_aarch64_feature_detected!("neon") {
            return rem_scalar_neon(selection, array, scalar, dst);
        }
    }

    let scalar = scalar.cast();
    arith_scalar(
        selection,
        array,
        scalar,
        dst,
        T::PARTIAL_ARITH_THRESHOLD,
        rem_scalar_func,
        T::new_remainder,
    )
}

#[cfg(feature = "avx512")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn rem_scalar_avx512<T, U>(
    selection: &Bitmap,
    array: &PrimitiveArray<T>,
    scalar: U,
    dst: &mut PrimitiveArray<U>,
) where
    PrimitiveArray<T>: Array,
    PrimitiveArray<U>: Array,
    T: RemExt,
    U: RemCast<T>,
{
    let scalar = scalar.cast();
    arith_scalar(
        selection,
        array,
        scalar,
        dst,
        T::AVX512_PARTIAL_ARITH_THRESHOLD,
        rem_scalar_func,
        T::new_remainder,
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn rem_scalar_avx2<T, U>(
    selection: &Bitmap,
    array: &PrimitiveArray<T>,
    scalar: U,
    dst: &mut PrimitiveArray<U>,
) where
    PrimitiveArray<T>: Array,
    PrimitiveArray<U>: Array,
    T: RemExt,
    U: RemCast<T>,
{
    let scalar = scalar.cast();
    arith_scalar(
        selection,
        array,
        scalar,
        dst,
        T::AVX2_PARTIAL_ARITH_THRESHOLD,
        rem_scalar_func,
        T::new_remainder,
    )
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn rem_scalar_neon<T, U>(
    selection: &Bitmap,
    array: &PrimitiveArray<T>,
    scalar: U,
    dst: &mut PrimitiveArray<U>,
) where
    PrimitiveArray<T>: Array,
    PrimitiveArray<U>: Array,
    T: RemExt,
    U: RemCast<T>,
{
    let scalar = scalar.cast();
    arith_scalar(
        selection,
        array,
        scalar,
        dst,
        T::NEON_PARTIAL_ARITH_THRESHOLD,
        rem_scalar_func,
        T::new_remainder,
    )
}

/// We do not construct divisor here, because the [`strength_reduce`] shows
/// that: **division for 1-2 divisions is not worth the setup cost**
///
/// [`strength_reduce`]: https://docs.rs/strength_reduce/latest/strength_reduce/
#[inline]
fn rem_func<T: RemExt, U: RemCast<T>>(lhs: T, rhs: U) -> U {
    U::cast_back(lhs % rhs.cast())
}

/// Perform `%` between `PrimitiveArray<T>` and `PrimitiveArray<U>`
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
pub unsafe fn rem<T, U>(
    selection: &Bitmap,
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<U>,
    dst: &mut PrimitiveArray<U>,
) where
    PrimitiveArray<T>: Array,
    PrimitiveArray<U>: Array,
    T: RemExt,
    U: RemCast<T>,
{
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // Note that this `unsafe` block is safe because we're testing
        // that the `avx512` feature is indeed available on our CPU.
        #[cfg(feature = "avx512")]
        if std::arch::is_x86_feature_detected!("avx512f") {
            return rem_avx512(selection, lhs, rhs, dst);
        }

        // Note that this `unsafe` block is safe because we're testing
        // that the `avx2` feature is indeed available on our CPU.
        if std::arch::is_x86_feature_detected!("avx2") {
            return rem_avx2(selection, lhs, rhs, dst);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Note that this `unsafe` block is safe because we're testing
        // that the `neon` feature is indeed available on our CPU.
        if std::arch::is_aarch64_feature_detected!("neon") {
            return rem_neon(selection, lhs, rhs, dst);
        }
    }

    arith_arrays(
        selection,
        lhs,
        rhs,
        dst,
        T::PARTIAL_ARITH_THRESHOLD,
        rem_func,
    )
}

#[cfg(feature = "avx512")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn rem_avx512<T, U>(
    selection: &Bitmap,
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<U>,
    dst: &mut PrimitiveArray<U>,
) where
    PrimitiveArray<T>: Array,
    PrimitiveArray<U>: Array,
    T: RemExt,
    U: RemCast<T>,
{
    arith_arrays(
        selection,
        lhs,
        rhs,
        dst,
        T::AVX512_PARTIAL_ARITH_THRESHOLD,
        rem_func,
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn rem_avx2<T, U>(
    selection: &Bitmap,
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<U>,
    dst: &mut PrimitiveArray<U>,
) where
    PrimitiveArray<T>: Array,
    PrimitiveArray<U>: Array,
    T: RemExt,
    U: RemCast<T>,
{
    arith_arrays(
        selection,
        lhs,
        rhs,
        dst,
        T::AVX2_PARTIAL_ARITH_THRESHOLD,
        rem_func,
    )
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn rem_neon<T, U>(
    selection: &Bitmap,
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<U>,
    dst: &mut PrimitiveArray<U>,
) where
    PrimitiveArray<T>: Array,
    PrimitiveArray<U>: Array,
    T: RemExt,
    U: RemCast<T>,
{
    arith_arrays(
        selection,
        lhs,
        rhs,
        dst,
        T::NEON_PARTIAL_ARITH_THRESHOLD,
        rem_func,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::LogicalType;

    #[test]
    fn test_rem_scalar() {
        let lhs = PrimitiveArray::from_values_iter([10, -7, -3, -9, -20, 87, 100, 34, 65]);
        let rhs = -3;
        let mut dst = PrimitiveArray::new(LogicalType::Integer).unwrap();
        unsafe { rem_scalar(&Bitmap::new(), &lhs, rhs, &mut dst) };
        assert_eq!(dst.values(), &[1, -1, 0, 0, -2, 0, 1, 1, 2]);
    }
}
