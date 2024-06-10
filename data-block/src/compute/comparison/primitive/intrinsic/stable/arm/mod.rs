//! Using `neon` to implement SIMD, stable code can use SIMD via this module
//!
//! I can not avoid using macro, therefore, I do not use macro mix trait...

use crate::aligned_vec::AlignedVec;
use crate::bitmap::BitStore;
use crate::utils::roundup_loops;

macro_rules! arm_target_use {
    ($($name:ident),+) => {
        #[cfg(target_arch = "arm")]
        use std::arch::arm::{$($name),+};
        #[cfg(target_arch = "aarch64")]
        use std::arch::aarch64::{$($name),+};
    };
}

arm_target_use!(
    vld1q_u8,
    vld1q_s8,
    vandq_u8,
    vpaddq_u8,
    vgetq_lane_u64,
    vreinterpretq_u64_u8,
    vceqq_u8,
    vcgtq_u8,
    vcgeq_u8,
    vcltq_u8,
    vcleq_u8,
    vceqq_s8,
    vcgtq_s8,
    vcgeq_s8,
    vcltq_s8,
    vcleq_s8,
    vdupq_n_u8,
    vdupq_n_s8,
    vdupq_n_s16,
    vdupq_n_u16,
    vdupq_n_s32,
    vdupq_n_u32,
    vdupq_n_f32,
    vdupq_n_s64,
    vdupq_n_u64,
    vdupq_n_f64,
    vld1q_s16,
    vld1q_u16,
    vceqq_s16,
    vcgtq_s16,
    vcgeq_s16,
    vcltq_s16,
    vcleq_s16,
    vceqq_u16,
    vcgtq_u16,
    vcgeq_u16,
    vcltq_u16,
    vcleq_u16,
    vaddvq_u16,
    vandq_u16,
    vaddq_u16,
    vld1q_s32,
    vld1q_u32,
    vceqq_s32,
    vcgtq_s32,
    vcgeq_s32,
    vcltq_s32,
    vcleq_s32,
    vceqq_u32,
    vcgtq_u32,
    vcgeq_u32,
    vcltq_u32,
    vcleq_u32,
    vld1q_f32,
    vceqq_f32,
    vcgtq_f32,
    vcgeq_f32,
    vcltq_f32,
    vcleq_f32,
    vuzp1q_u16,
    vreinterpretq_u16_u32,
    vld1q_s64,
    vld1q_u64,
    vceqq_s64,
    vcgtq_s64,
    vcgeq_s64,
    vcltq_s64,
    vcleq_s64,
    vceqq_u64,
    vcgtq_u64,
    vcgeq_u64,
    vcltq_u64,
    vcleq_u64,
    vld1q_f64,
    vceqq_f64,
    vcgtq_f64,
    vcgeq_f64,
    vcltq_f64,
    vcleq_f64,
    vuzp1q_u32,
    vreinterpretq_u32_u64
);

macro_rules! cmp_scalar {
    ($ty:ty, $suffix:ident, $inner_macro:ident) => {
        paste::paste! {
            $inner_macro!([<eq_scalar_ $ty _neon>], $ty, [<vdupq_n_ $suffix>], [<vld1q_ $suffix>], [<vceqq_ $suffix>], );
            $inner_macro!([<ne_scalar_ $ty _neon>], $ty, [<vdupq_n_ $suffix>], [<vld1q_ $suffix>], [<vceqq_ $suffix>], true);
            $inner_macro!([<gt_scalar_ $ty _neon>], $ty, [<vdupq_n_ $suffix>], [<vld1q_ $suffix>], [<vcgtq_ $suffix>], );
            $inner_macro!([<ge_scalar_ $ty _neon>], $ty, [<vdupq_n_ $suffix>], [<vld1q_ $suffix>], [<vcgeq_ $suffix>], );
            $inner_macro!([<lt_scalar_ $ty _neon>], $ty, [<vdupq_n_ $suffix>], [<vld1q_ $suffix>], [<vcltq_ $suffix>], );
            $inner_macro!([<le_scalar_ $ty _neon>], $ty, [<vdupq_n_ $suffix>], [<vld1q_ $suffix>], [<vcleq_ $suffix>], );
        }
    };
}

const MASK_8: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

macro_rules! cmp_scalar_byte {
    ($func_name:ident,
        $ty:ty,
        $broadcast_fn:ident,
        $load_fn:ident,
        $cmp_func:ident,
        $($not:expr)?
    ) => {
        /// It is slower than auto-vectorization that write to boolean array 😭
        ///
        /// We implement pairwise here, benchmarks on M1 shows that pairwise is faster than
        /// interleaved, see [here] for details
        ///
        /// [here]: https://community.arm.com/arm-community-blogs/b/infrastructure-solutions-blog/posts/porting-x86-vector-bitmask-optimizations-to-arm-neon
        #[target_feature(enable = "neon")]
        #[inline]
        pub(crate) unsafe fn $func_name(
            lhs: &AlignedVec<$ty>,
            rhs: $ty,
            dst: *mut BitStore,
        ) {
            if lhs.len == 0 {
                return;
            }

            let rhs = $broadcast_fn(rhs);
            let mask_8 = vld1q_u8(MASK_8.as_ptr());
            let num_loops = roundup_loops(lhs.len, 64);
            let mut lhs_ptr = lhs.ptr.as_ptr();

            for simd_index in 0..num_loops{
                let lhs_0 = $load_fn(lhs_ptr);
                let lhs_1 = $load_fn(lhs_ptr.add(16));
                let lhs_2 = $load_fn(lhs_ptr.add(32));
                let lhs_3 = $load_fn(lhs_ptr.add(48));
                let cmp_0 = vandq_u8($cmp_func(lhs_0, rhs), mask_8);
                let cmp_1 = vandq_u8($cmp_func(lhs_1, rhs), mask_8);
                let cmp_2 = vandq_u8($cmp_func(lhs_2, rhs), mask_8);
                let cmp_3 = vandq_u8($cmp_func(lhs_3, rhs), mask_8);
                let sum_0 = vpaddq_u8(cmp_0, cmp_1);
                let sum_1 = vpaddq_u8(cmp_2, cmp_3);
                let sum_0 = vpaddq_u8(sum_0, sum_1);
                let sum_0 = vpaddq_u8(sum_0, sum_0);
                let mask = vgetq_lane_u64(vreinterpretq_u64_u8(sum_0), 0);
                $(
                    let mask = if $not{
                        mask ^ u64::MAX
                    }else{
                        mask
                    };
                )?
                *dst.add(simd_index) = mask;
                lhs_ptr = lhs_ptr.add(64);
            }
        }
    };
}

const MASK_16_LOW: [u16; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
const MASK_16_HIGH: [u16; 8] = [
    0x0100, 0x0200, 0x0400, 0x0800, 0x1000, 0x2000, 0x4000, 0x8000,
];

macro_rules! cmp_scalar_half_word {
    ($func_name:ident,
        $ty:ty,
        $broadcast_fn:ident,
        $load_fn:ident,
        $cmp_func:ident,
        $($not:expr)?
    ) => {
        #[target_feature(enable = "neon")]
        #[inline]
        pub(crate) unsafe fn $func_name(lhs: &AlignedVec<$ty>, rhs: $ty, dst: *mut BitStore) {
            if lhs.len == 0 {
                return;
            }

            let rhs = $broadcast_fn(rhs);
            let mask_low = vld1q_u16(MASK_16_LOW.as_ptr());
            let mask_high = vld1q_u16(MASK_16_HIGH.as_ptr());
            let num_loops = roundup_loops(lhs.len, 16);
            let mut lhs_ptr = lhs.ptr.as_ptr();
            let dst = dst as *mut u16;

            for simd_index in 0..num_loops {
                let lhs_0 = $load_fn(lhs_ptr);
                let lhs_1 = $load_fn(lhs_ptr.add(8));
                let cmp_0 = vandq_u16($cmp_func(lhs_0, rhs), mask_low);
                let cmp_1 = vandq_u16($cmp_func(lhs_1, rhs), mask_high);
                let mask = vaddvq_u16(vaddq_u16(cmp_0, cmp_1));
                $(
                    let mask = if $not{
                        mask ^ u16::MAX
                    }else{
                        mask
                    };
                )?
                *dst.add(simd_index) = mask as _;
                lhs_ptr = lhs_ptr.add(16);
            }
        }
    };
}

macro_rules! cmp_scalar_word {
    ($func_name:ident,
        $ty:ty,
        $broadcast_fn:ident,
        $load_fn:ident,
        $cmp_func:ident,
        $($not:expr)?
    ) => {
        #[target_feature(enable = "neon")]
        #[inline]
        pub(crate) unsafe fn $func_name(lhs: &AlignedVec<$ty>, rhs: $ty, dst: *mut BitStore) {
            if lhs.len == 0 {
                return;
            }

            let rhs = $broadcast_fn(rhs);
            let mask_low = vld1q_u16(MASK_16_LOW.as_ptr());
            let mask_high = vld1q_u16(MASK_16_HIGH.as_ptr());
            let num_loops = roundup_loops(lhs.len, 16);
            let mut lhs_ptr = lhs.ptr.as_ptr();
            let dst = dst as *mut u16;

            for simd_index in 0..num_loops {
                let lhs_0 = $load_fn(lhs_ptr);
                let lhs_1 = $load_fn(lhs_ptr.add(4));
                let lhs_2 = $load_fn(lhs_ptr.add(8));
                let lhs_3 = $load_fn(lhs_ptr.add(12));

                let cmp_0 = $cmp_func(lhs_0, rhs);
                let cmp_1 = $cmp_func(lhs_1, rhs);
                let cmp_2 = $cmp_func(lhs_2, rhs);
                let cmp_3 = $cmp_func(lhs_3, rhs);
                let uzp_0 = vuzp1q_u16(vreinterpretq_u16_u32(cmp_0), vreinterpretq_u16_u32(cmp_1));
                let uzp_1 = vuzp1q_u16(vreinterpretq_u16_u32(cmp_2), vreinterpretq_u16_u32(cmp_3));
                let uzp_0 = vandq_u16(uzp_0, mask_low);
                let uzp_1 = vandq_u16(uzp_1, mask_high);
                let mask = vaddvq_u16(vaddq_u16(uzp_0, uzp_1));
                $(
                    let mask = if $not{
                        mask ^ u16::MAX
                    }else{
                        mask
                    };
                )?
                *dst.add(simd_index) = mask;
                lhs_ptr = lhs_ptr.add(16);
            }
        }
    };
}

macro_rules! cmp_scalar_double_word {
    ($func_name:ident,
        $ty:ty,
        $broadcast_fn:ident,
        $load_fn:ident,
        $cmp_func:ident,
        $($not:expr)?
    ) => {
        #[target_feature(enable = "neon")]
        #[inline]
        pub(crate) unsafe fn $func_name(lhs: &AlignedVec<$ty>, rhs: $ty, dst: *mut BitStore) {
            if lhs.len == 0 {
                return;
            }

            let rhs = $broadcast_fn(rhs);
            let num_loops = roundup_loops(lhs.len, 8);
            let mask_low = vld1q_u16(MASK_16_LOW.as_ptr());
            let mut lhs_ptr = lhs.ptr.as_ptr();
            let dst = dst as *mut u8;

            for simd_index in 0..num_loops {
                let lhs_0 = $load_fn(lhs_ptr);
                let lhs_1 = $load_fn(lhs_ptr.add(2));
                let lhs_2 = $load_fn(lhs_ptr.add(4));
                let lhs_3 = $load_fn(lhs_ptr.add(6));

                let cmp_0 = $cmp_func(lhs_0, rhs);
                let cmp_1 = $cmp_func(lhs_1, rhs);
                let cmp_2 = $cmp_func(lhs_2, rhs);
                let cmp_3 = $cmp_func(lhs_3, rhs);

                let uzp_0 = vuzp1q_u32(vreinterpretq_u32_u64(cmp_0), vreinterpretq_u32_u64(cmp_1));
                let uzp_1 = vuzp1q_u32(vreinterpretq_u32_u64(cmp_2), vreinterpretq_u32_u64(cmp_3));
                let uzp = vuzp1q_u16(vreinterpretq_u16_u32(uzp_0), vreinterpretq_u16_u32(uzp_1));
                let mask = vaddvq_u16(vandq_u16(uzp, mask_low)) as u8;

                $(
                    let mask = if $not{
                        mask ^ u8::MAX
                    }else{
                        mask
                    };
                )?

                *dst.add(simd_index) = mask;
                lhs_ptr = lhs_ptr.add(8);
            }
        }
    };
}

cmp_scalar!(i8, s8, cmp_scalar_byte);
cmp_scalar!(u8, u8, cmp_scalar_byte);
cmp_scalar!(i16, s16, cmp_scalar_half_word);
cmp_scalar!(u16, u16, cmp_scalar_half_word);
cmp_scalar!(i32, s32, cmp_scalar_word);
cmp_scalar!(u32, u32, cmp_scalar_word);
cmp_scalar!(f32, f32, cmp_scalar_word);
cmp_scalar!(i64, s64, cmp_scalar_double_word);
cmp_scalar!(u64, u64, cmp_scalar_double_word);
cmp_scalar!(f64, f64, cmp_scalar_double_word);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::Bitmap;

    #[test]
    fn test_equal_scalar_i8() {
        let lhs = AlignedVec::from_slice(&[
            -1, 5, 122, -12, -33, -22, 45, -1, 90, 100, -11, -66, -33, 7, 0,
        ]);
        let rhs = -1;
        cmp_assert!(&lhs, rhs, eq_scalar_i8_neon, &[0x0081]);
        cmp_assert!(&lhs, rhs, ne_scalar_i8_neon, &[!0x0081]);
    }

    #[test]
    fn test_order_scalar_i8() {
        let lhs = AlignedVec::from_slice(&[
            -1, 5, 122, -12, -33, -22, 45, -1, 90, 100, -11, -66, -33, 7, 0,
        ]);
        let rhs = -1;
        cmp_assert!(&lhs, rhs, gt_scalar_i8_neon, &[0x6346]);
        cmp_assert!(&lhs, rhs, ge_scalar_i8_neon, &[0x63c7]);
        cmp_assert!(&lhs, rhs, lt_scalar_i8_neon, &[!0x63c7]);
        cmp_assert!(&lhs, rhs, le_scalar_i8_neon, &[!0x6346]);
    }

    #[test]
    fn test_equal_scalar_i16() {
        let lhs =
            AlignedVec::from_slice(&[0, -333, -56, 211, 5667, 0, -256, -1222, -333, 777, 444, 65]);
        let rhs = 0;
        cmp_assert!(&lhs, rhs, eq_scalar_i16_neon, &[0x0021]);
        cmp_assert!(&lhs, rhs, ne_scalar_i16_neon, &[!0x0021]);
    }

    #[test]
    fn test_order_scalar_i16() {
        let lhs =
            AlignedVec::from_slice(&[0, -333, -56, 211, 5667, 0, -256, -1222, -333, 777, 444, 65]);
        let rhs = 0;
        cmp_assert!(&lhs, rhs, gt_scalar_i16_neon, &[0xe18]);
        cmp_assert!(&lhs, rhs, ge_scalar_i16_neon, &[0xe39]);
        cmp_assert!(&lhs, rhs, lt_scalar_i16_neon, &[!0xe39]);
        cmp_assert!(&lhs, rhs, le_scalar_i16_neon, &[!0xe18]);
    }

    #[test]
    fn test_equal_scalar_i32() {
        let lhs = AlignedVec::from_slice(&[345, 123, -835, -1, 565, -345, -1, 456, -3]);
        let rhs = -1;
        cmp_assert!(&lhs, rhs, eq_scalar_i32_neon, &[0x048]);
        cmp_assert!(&lhs, rhs, ne_scalar_i32_neon, &[!0x048]);
    }

    #[test]
    fn test_order_scalar_i32() {
        let lhs = AlignedVec::from_slice(&[345, 123, -835, -1, 565, -345, -1, 456, -3]);
        let rhs = -1;
        cmp_assert!(&lhs, rhs, gt_scalar_i32_neon, &[0x093]);
        cmp_assert!(&lhs, rhs, ge_scalar_i32_neon, &[0x0db]);
        cmp_assert!(&lhs, rhs, lt_scalar_i32_neon, &[!0x0db]);
        cmp_assert!(&lhs, rhs, le_scalar_i32_neon, &[!0x093]);
    }

    #[test]
    fn test_equal_scalar_i64() {
        let lhs =
            AlignedVec::from_slice(&[78, 343, -12, -67, -12, -12, 33, 5687, -456, -1, -1, -1, 0]);
        let rhs = -12;
        cmp_assert!(&lhs, rhs, eq_scalar_i64_neon, &[0x0034]);
        cmp_assert!(&lhs, rhs, ne_scalar_i64_neon, &[!0x0034]);
    }

    #[test]
    fn test_order_scalar_i64() {
        let lhs =
            AlignedVec::from_slice(&[78, 343, -12, -67, -12, -12, 33, 5687, -456, -1, -1, -1, 0]);
        let rhs = -12;
        cmp_assert!(&lhs, rhs, gt_scalar_i64_neon, &[0x1ec3]);
        cmp_assert!(&lhs, rhs, ge_scalar_i64_neon, &[0x1ef7]);
        cmp_assert!(&lhs, rhs, lt_scalar_i64_neon, &[!0x1ef7]);
        cmp_assert!(&lhs, rhs, le_scalar_i64_neon, &[!0x1ec3]);
    }
}
