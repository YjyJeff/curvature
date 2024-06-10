use std::convert::identity;
use std::ops::Add;

use super::{
    arith_arrays, arith_scalar, NATIVE_PARTIAL_ARITH_THRESHOLD_16_BIT,
    NATIVE_PARTIAL_ARITH_THRESHOLD_32_BIT, NATIVE_PARTIAL_ARITH_THRESHOLD_64_BIT,
    NATIVE_PARTIAL_ARITH_THRESHOLD_8_BIT,
};
use crate::array::{Array, PrimitiveArray};
use crate::bitmap::Bitmap;
use crate::types::IntrinsicType;

/// Extension trait for add
pub trait AddExt: IntrinsicType + Add<Output = Self> {
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
}

macro_rules! impl_add_ext {
    ($ty:ty, $default_th:expr, $neon_th:expr, $avx2_th:expr, $avx512_th:expr) => {
        impl AddExt for $ty {
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
        }
    };
}

impl_add_ext!(
    i8,
    NATIVE_PARTIAL_ARITH_THRESHOLD_8_BIT,
    NATIVE_PARTIAL_ARITH_THRESHOLD_8_BIT,
    0.0,
    0.0
);
impl_add_ext!(
    u8,
    NATIVE_PARTIAL_ARITH_THRESHOLD_8_BIT,
    NATIVE_PARTIAL_ARITH_THRESHOLD_8_BIT,
    0.0,
    0.0
);
impl_add_ext!(
    i16,
    NATIVE_PARTIAL_ARITH_THRESHOLD_16_BIT,
    NATIVE_PARTIAL_ARITH_THRESHOLD_16_BIT,
    0.0,
    0.0
);
impl_add_ext!(
    u16,
    NATIVE_PARTIAL_ARITH_THRESHOLD_16_BIT,
    NATIVE_PARTIAL_ARITH_THRESHOLD_16_BIT,
    0.0,
    0.0
);
impl_add_ext!(
    i32,
    NATIVE_PARTIAL_ARITH_THRESHOLD_32_BIT,
    NATIVE_PARTIAL_ARITH_THRESHOLD_32_BIT,
    0.0,
    0.0
);
impl_add_ext!(
    u32,
    NATIVE_PARTIAL_ARITH_THRESHOLD_32_BIT,
    NATIVE_PARTIAL_ARITH_THRESHOLD_32_BIT,
    0.0,
    0.0
);
impl_add_ext!(
    i64,
    NATIVE_PARTIAL_ARITH_THRESHOLD_64_BIT,
    NATIVE_PARTIAL_ARITH_THRESHOLD_64_BIT,
    0.0,
    0.0
);
impl_add_ext!(
    u64,
    NATIVE_PARTIAL_ARITH_THRESHOLD_64_BIT,
    NATIVE_PARTIAL_ARITH_THRESHOLD_64_BIT,
    0.0,
    0.0
);
impl_add_ext!(
    f32,
    NATIVE_PARTIAL_ARITH_THRESHOLD_32_BIT,
    NATIVE_PARTIAL_ARITH_THRESHOLD_32_BIT,
    0.0,
    0.0
);
impl_add_ext!(
    f64,
    NATIVE_PARTIAL_ARITH_THRESHOLD_64_BIT,
    NATIVE_PARTIAL_ARITH_THRESHOLD_64_BIT,
    0.0,
    0.0
);

/// Perform `array + scalar` for `PrimitiveArray<T>` with `T`
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
pub unsafe fn add_scalar<T>(
    selection: &Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    dst: &mut PrimitiveArray<T>,
) where
    PrimitiveArray<T>: Array,
    T: AddExt,
{
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // Note that this `unsafe` block is safe because we're testing
        // that the `avx512` feature is indeed available on our CPU.
        if std::arch::is_x86_feature_detected!("avx512f") {
            return add_scalar_avx512(selection, array, scalar, dst);
        }

        // Note that this `unsafe` block is safe because we're testing
        // that the `avx2` feature is indeed available on our CPU.
        if std::arch::is_x86_feature_detected!("avx2") {
            return add_scalar_avx2(selection, array, scalar, dst);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Note that this `unsafe` block is safe because we're testing
        // that the `neon` feature is indeed available on our CPU.
        if std::arch::is_aarch64_feature_detected!("neon") {
            return add_scalar_neon(selection, array, scalar, dst);
        }
    }

    arith_scalar(
        selection,
        array,
        scalar,
        dst,
        T::PARTIAL_ARITH_THRESHOLD,
        Add::add,
        identity,
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn add_scalar_avx512<T>(
    selection: &Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    dst: &mut PrimitiveArray<T>,
) where
    PrimitiveArray<T>: Array,
    T: AddExt,
{
    arith_scalar(
        selection,
        array,
        scalar,
        dst,
        T::AVX512_PARTIAL_ARITH_THRESHOLD,
        Add::add,
        identity,
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn add_scalar_avx2<T>(
    selection: &Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    dst: &mut PrimitiveArray<T>,
) where
    PrimitiveArray<T>: Array,
    T: AddExt,
{
    arith_scalar(
        selection,
        array,
        scalar,
        dst,
        T::AVX2_PARTIAL_ARITH_THRESHOLD,
        Add::add,
        identity,
    )
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn add_scalar_neon<T>(
    selection: &Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    dst: &mut PrimitiveArray<T>,
) where
    PrimitiveArray<T>: Array,
    T: AddExt,
{
    arith_scalar(
        selection,
        array,
        scalar,
        dst,
        T::NEON_PARTIAL_ARITH_THRESHOLD,
        Add::add,
        identity,
    )
}

/// Perform `+` between two `PrimitiveArray<T>`
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
pub unsafe fn add<T>(
    selection: &Bitmap,
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<T>,
    dst: &mut PrimitiveArray<T>,
) where
    PrimitiveArray<T>: Array,
    T: AddExt,
{
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // Note that this `unsafe` block is safe because we're testing
        // that the `avx512` feature is indeed available on our CPU.
        if std::arch::is_x86_feature_detected!("avx512f") {
            return add_avx512(selection, lhs, rhs, dst);
        }

        // Note that this `unsafe` block is safe because we're testing
        // that the `avx2` feature is indeed available on our CPU.
        if std::arch::is_x86_feature_detected!("avx2") {
            return add_avx2(selection, lhs, rhs, dst);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Note that this `unsafe` block is safe because we're testing
        // that the `neon` feature is indeed available on our CPU.
        if std::arch::is_aarch64_feature_detected!("neon") {
            return add_neon(selection, lhs, rhs, dst);
        }
    }

    arith_arrays(
        selection,
        lhs,
        rhs,
        dst,
        T::PARTIAL_ARITH_THRESHOLD,
        Add::add,
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn add_avx512<T>(
    selection: &Bitmap,
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<T>,
    dst: &mut PrimitiveArray<T>,
) where
    PrimitiveArray<T>: Array,
    T: AddExt,
{
    arith_arrays(
        selection,
        lhs,
        rhs,
        dst,
        T::AVX512_PARTIAL_ARITH_THRESHOLD,
        Add::add,
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn add_avx2<T>(
    selection: &Bitmap,
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<T>,
    dst: &mut PrimitiveArray<T>,
) where
    PrimitiveArray<T>: Array,
    T: AddExt,
{
    arith_arrays(
        selection,
        lhs,
        rhs,
        dst,
        T::AVX2_PARTIAL_ARITH_THRESHOLD,
        Add::add,
    )
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn add_neon<T>(
    selection: &Bitmap,
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<T>,
    dst: &mut PrimitiveArray<T>,
) where
    PrimitiveArray<T>: Array,
    T: AddExt,
{
    arith_arrays(
        selection,
        lhs,
        rhs,
        dst,
        T::NEON_PARTIAL_ARITH_THRESHOLD,
        Add::add,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::LogicalType;

    #[test]
    fn test_add_scalar() {
        let lhs = PrimitiveArray::from_values_iter([10, -7, -3, -9, -20, 87, 100, 34, 65]);
        let rhs = 1;
        let mut dst = PrimitiveArray::new(LogicalType::Integer).unwrap();
        unsafe { add_scalar(&Bitmap::new(), &lhs, rhs, &mut dst) };
        assert_eq!(dst.values(), &[11, -6, -2, -8, -19, 88, 101, 35, 66]);
    }
}
