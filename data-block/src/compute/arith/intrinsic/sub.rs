use std::convert::identity;
use std::ops::Sub;

use super::{
    arith_arrays, arith_scalar, dynamic_arith_arrays_func, dynamic_arith_scalar_func,
    NATIVE_PARTIAL_ARITH_THRESHOLD_16_BIT, NATIVE_PARTIAL_ARITH_THRESHOLD_32_BIT,
    NATIVE_PARTIAL_ARITH_THRESHOLD_64_BIT, NATIVE_PARTIAL_ARITH_THRESHOLD_8_BIT,
};
use crate::array::{Array, PrimitiveArray};
use crate::bitmap::Bitmap;
use crate::types::IntrinsicType;

/// Extension trait for sub
pub trait SubExt: IntrinsicType + Sub<Output = Self> {
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

macro_rules! impl_sub_ext {
    ($ty:ty, $default_th:expr, $neon_th:expr, $avx2_th:expr, $avx512_th:expr) => {
        impl SubExt for $ty {
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

impl_sub_ext!(
    i8,
    NATIVE_PARTIAL_ARITH_THRESHOLD_8_BIT,
    NATIVE_PARTIAL_ARITH_THRESHOLD_8_BIT,
    0.0,
    0.0
);
impl_sub_ext!(
    u8,
    NATIVE_PARTIAL_ARITH_THRESHOLD_8_BIT,
    NATIVE_PARTIAL_ARITH_THRESHOLD_8_BIT,
    0.0,
    0.0
);
impl_sub_ext!(
    i16,
    NATIVE_PARTIAL_ARITH_THRESHOLD_16_BIT,
    NATIVE_PARTIAL_ARITH_THRESHOLD_16_BIT,
    0.0,
    0.0
);
impl_sub_ext!(
    u16,
    NATIVE_PARTIAL_ARITH_THRESHOLD_16_BIT,
    NATIVE_PARTIAL_ARITH_THRESHOLD_16_BIT,
    0.0,
    0.0
);
impl_sub_ext!(
    i32,
    NATIVE_PARTIAL_ARITH_THRESHOLD_32_BIT,
    NATIVE_PARTIAL_ARITH_THRESHOLD_32_BIT,
    0.0,
    0.0
);
impl_sub_ext!(
    u32,
    NATIVE_PARTIAL_ARITH_THRESHOLD_32_BIT,
    NATIVE_PARTIAL_ARITH_THRESHOLD_32_BIT,
    0.0,
    0.0
);
impl_sub_ext!(
    i64,
    NATIVE_PARTIAL_ARITH_THRESHOLD_64_BIT,
    NATIVE_PARTIAL_ARITH_THRESHOLD_64_BIT,
    0.0,
    0.0
);
impl_sub_ext!(
    u64,
    NATIVE_PARTIAL_ARITH_THRESHOLD_64_BIT,
    NATIVE_PARTIAL_ARITH_THRESHOLD_64_BIT,
    0.0,
    0.0
);
impl_sub_ext!(
    f32,
    NATIVE_PARTIAL_ARITH_THRESHOLD_32_BIT,
    NATIVE_PARTIAL_ARITH_THRESHOLD_32_BIT,
    0.0,
    0.0
);
impl_sub_ext!(
    f64,
    NATIVE_PARTIAL_ARITH_THRESHOLD_64_BIT,
    NATIVE_PARTIAL_ARITH_THRESHOLD_64_BIT,
    0.0,
    0.0
);

dynamic_arith_scalar_func!(#[doc = " Perform `array - scalar` between `PrimitiveArray<T>` and T"], sub_scalar, SubExt, Sub::sub, identity);
dynamic_arith_arrays_func!(#[doc = " Perform `-` between two `PrimitiveArray<T>`"], sub, SubExt, Sub::sub);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::LogicalType;

    #[test]
    fn test_sub_scalar() {
        let lhs = PrimitiveArray::from_values_iter([10, -7, -3, -9, -20, 87, 100, 34, 65]);
        let rhs = 1;
        let mut dst = PrimitiveArray::new(LogicalType::Integer).unwrap();
        unsafe { sub_scalar(&Bitmap::new(), &lhs, rhs, &mut dst) };
        assert_eq!(dst.values(), &[9, -8, -4, -10, -21, 86, 99, 33, 64]);
    }
}
