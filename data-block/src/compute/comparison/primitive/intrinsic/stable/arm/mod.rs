//! Using `neon` to implement SIMD, stable code can use SIMD via this module
//!
//! I can not avoid using macro, therefore, I do not use macro mix trait...

use crate::aligned_vec::AlignedVec;
use crate::bitmap::BitStore;
use crate::compute::IntrinsicType;
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
    int8x16_t,
    uint8x16_t,
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
    int16x8_t,
    uint16x8_t,
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
    int32x4_t,
    uint32x4_t,
    vld1q_f32,
    vceqq_f32,
    vcgtq_f32,
    vcgeq_f32,
    vcltq_f32,
    vcleq_f32,
    float32x4_t,
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
    float64x2_t,
    vuzp1q_u32,
    vreinterpretq_u32_u64,
    int64x2_t,
    uint64x2_t
);

const MASK_8: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

trait IntrinsicExt: IntrinsicType {
    type Simd: Sized + Clone + Copy;
    type SimdCmp: Sized;
    const LOAD: unsafe fn(*const Self) -> Self::Simd;
    const BROADCAST: unsafe fn(Self) -> Self::Simd;
    const EQ: unsafe fn(Self::Simd, Self::Simd) -> Self::SimdCmp;
    const GT: unsafe fn(Self::Simd, Self::Simd) -> Self::SimdCmp;
    const GE: unsafe fn(Self::Simd, Self::Simd) -> Self::SimdCmp;
    const LT: unsafe fn(Self::Simd, Self::Simd) -> Self::SimdCmp;
    const LE: unsafe fn(Self::Simd, Self::Simd) -> Self::SimdCmp;
}

macro_rules! impl_intrinsic_ext {
    ($ty:ty, $suffix:ident, $simd:ty, $simd_cmp:ty) => {
        paste::paste! {
            impl IntrinsicExt for $ty {
                type Simd = $simd;
                type SimdCmp = $simd_cmp;
                const LOAD: unsafe fn(*const Self) -> Self::Simd = [<vld1q_ $suffix>];
                const BROADCAST: unsafe fn(Self) -> Self::Simd = [<vdupq_n_ $suffix>];
                const EQ: unsafe fn(Self::Simd, Self::Simd) -> Self::SimdCmp = [<vceqq_ $suffix>] ;
                const GT: unsafe fn(Self::Simd, Self::Simd) -> Self::SimdCmp = [<vcgtq_ $suffix>];
                const GE: unsafe fn(Self::Simd, Self::Simd) -> Self::SimdCmp = [<vcgeq_ $suffix>];
                const LT: unsafe fn(Self::Simd, Self::Simd) -> Self::SimdCmp = [<vcltq_ $suffix>];
                const LE: unsafe fn(Self::Simd, Self::Simd) -> Self::SimdCmp = [<vcleq_ $suffix>];

            }
        }
    };
}

impl_intrinsic_ext!(i8, s8, int8x16_t, uint8x16_t);
impl_intrinsic_ext!(u8, u8, uint8x16_t, uint8x16_t);
impl_intrinsic_ext!(i16, s16, int16x8_t, uint16x8_t);
impl_intrinsic_ext!(u16, u16, uint16x8_t, uint16x8_t);
impl_intrinsic_ext!(i32, s32, int32x4_t, uint32x4_t);
impl_intrinsic_ext!(u32, u32, uint32x4_t, uint32x4_t);
impl_intrinsic_ext!(i64, s64, int64x2_t, uint64x2_t);
impl_intrinsic_ext!(u64, u64, uint64x2_t, uint64x2_t);
impl_intrinsic_ext!(f32, f32, float32x4_t, uint32x4_t);
impl_intrinsic_ext!(f64, f64, float64x2_t, uint64x2_t);

trait ByteIntrinsic: IntrinsicExt<SimdCmp = uint8x16_t> {}

impl ByteIntrinsic for i8 {}
impl ByteIntrinsic for u8 {}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn pack_byte_cmp<const NOT: bool>(
    cmp_0: uint8x16_t,
    cmp_1: uint8x16_t,
    cmp_2: uint8x16_t,
    cmp_3: uint8x16_t,
    mask_8: uint8x16_t,
) -> u64 {
    let cmp_0 = vandq_u8(cmp_0, mask_8);
    let cmp_1 = vandq_u8(cmp_1, mask_8);
    let cmp_2 = vandq_u8(cmp_2, mask_8);
    let cmp_3 = vandq_u8(cmp_3, mask_8);
    let sum_0 = vpaddq_u8(cmp_0, cmp_1);
    let sum_1 = vpaddq_u8(cmp_2, cmp_3);
    let sum_0 = vpaddq_u8(sum_0, sum_1);
    let sum_0 = vpaddq_u8(sum_0, sum_0);
    let mask = vgetq_lane_u64(vreinterpretq_u64_u8(sum_0), 0);
    if NOT {
        mask ^ u64::MAX
    } else {
        mask
    }
}

/// It is slower than auto-vectorization that write to boolean array 😭
///
/// We implement pairwise here, benchmarks on M1 shows that pairwise is faster than
/// interleaved, see [here] for details
///
/// [here]: https://community.arm.com/arm-community-blogs/b/infrastructure-solutions-blog/posts/porting-x86-vector-bitmask-optimizations-to-arm-neon
#[target_feature(enable = "neon")]
#[inline]
unsafe fn byte_cmp_scalar<const NOT: bool, T: ByteIntrinsic>(
    lhs: &AlignedVec<T>,
    rhs: T,
    dst: *mut BitStore,
    cmp: unsafe fn(T::Simd, T::Simd) -> T::SimdCmp,
) {
    if lhs.len == 0 {
        return;
    }

    let rhs = T::BROADCAST(rhs);
    let mask_8 = vld1q_u8(MASK_8.as_ptr());
    let num_loops = roundup_loops(lhs.len, 64);
    let mut lhs_ptr = lhs.ptr.as_ptr();

    for simd_index in 0..num_loops {
        let lhs_0 = T::LOAD(lhs_ptr);
        let lhs_1 = T::LOAD(lhs_ptr.add(16));
        let lhs_2 = T::LOAD(lhs_ptr.add(32));
        let lhs_3 = T::LOAD(lhs_ptr.add(48));
        let mask = pack_byte_cmp::<NOT>(
            cmp(lhs_0, rhs),
            cmp(lhs_1, rhs),
            cmp(lhs_2, rhs),
            cmp(lhs_3, rhs),
            mask_8,
        );

        *dst.add(simd_index) = mask;
        lhs_ptr = lhs_ptr.add(64);
    }
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn byte_cmp<const NOT: bool, T: ByteIntrinsic>(
    lhs: &AlignedVec<T>,
    rhs: &AlignedVec<T>,
    dst: *mut BitStore,
    cmp: unsafe fn(T::Simd, T::Simd) -> T::SimdCmp,
) {
    debug_assert_eq!(lhs.len, rhs.len);
    if lhs.len == 0 {
        return;
    }

    let mask_8 = vld1q_u8(MASK_8.as_ptr());
    let num_loops = roundup_loops(lhs.len, 64);
    let mut lhs_ptr = lhs.ptr.as_ptr();
    let mut rhs_ptr = rhs.ptr.as_ptr();

    for simd_index in 0..num_loops {
        let lhs_0 = T::LOAD(lhs_ptr);
        let lhs_1 = T::LOAD(lhs_ptr.add(16));
        let lhs_2 = T::LOAD(lhs_ptr.add(32));
        let lhs_3 = T::LOAD(lhs_ptr.add(48));
        let rhs_0 = T::LOAD(rhs_ptr);
        let rhs_1 = T::LOAD(rhs_ptr.add(16));
        let rhs_2 = T::LOAD(rhs_ptr.add(32));
        let rhs_3 = T::LOAD(rhs_ptr.add(48));
        let mask = pack_byte_cmp::<NOT>(
            cmp(lhs_0, rhs_0),
            cmp(lhs_1, rhs_1),
            cmp(lhs_2, rhs_2),
            cmp(lhs_3, rhs_3),
            mask_8,
        );

        *dst.add(simd_index) = mask;
        lhs_ptr = lhs_ptr.add(64);
        rhs_ptr = rhs_ptr.add(64);
    }
}

trait HalfWordIntrinsic: IntrinsicExt<SimdCmp = uint16x8_t> {}

impl HalfWordIntrinsic for i16 {}
impl HalfWordIntrinsic for u16 {}

const MASK_16_LOW: [u16; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
const MASK_16_HIGH: [u16; 8] = [
    0x0100, 0x0200, 0x0400, 0x0800, 0x1000, 0x2000, 0x4000, 0x8000,
];

#[target_feature(enable = "neon")]
#[inline]
unsafe fn pack_half_word_cmp<const NOT: bool>(
    cmp_0: uint16x8_t,
    cmp_1: uint16x8_t,
    mask_low: uint16x8_t,
    mask_high: uint16x8_t,
) -> u16 {
    let cmp_0 = vandq_u16(cmp_0, mask_low);
    let cmp_1 = vandq_u16(cmp_1, mask_high);
    let mask = vaddvq_u16(vaddq_u16(cmp_0, cmp_1));
    if NOT {
        mask ^ u16::MAX
    } else {
        mask
    }
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn half_word_cmp_scalar<const NOT: bool, T: HalfWordIntrinsic>(
    lhs: &AlignedVec<T>,
    rhs: T,
    dst: *mut BitStore,
    cmp: unsafe fn(T::Simd, T::Simd) -> T::SimdCmp,
) {
    if lhs.len == 0 {
        return;
    }

    let rhs = T::BROADCAST(rhs);
    let mask_low = vld1q_u16(MASK_16_LOW.as_ptr());
    let mask_high = vld1q_u16(MASK_16_HIGH.as_ptr());
    let num_loops = roundup_loops(lhs.len, 16);
    let mut lhs_ptr = lhs.ptr.as_ptr();
    let dst = dst as *mut u16;

    for simd_index in 0..num_loops {
        let lhs_0 = T::LOAD(lhs_ptr);
        let lhs_1 = T::LOAD(lhs_ptr.add(8));
        let mask = pack_half_word_cmp::<NOT>(cmp(lhs_0, rhs), cmp(lhs_1, rhs), mask_low, mask_high);
        *dst.add(simd_index) = mask as _;
        lhs_ptr = lhs_ptr.add(16);
    }
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn half_word_cmp<const NOT: bool, T: HalfWordIntrinsic>(
    lhs: &AlignedVec<T>,
    rhs: &AlignedVec<T>,
    dst: *mut BitStore,
    cmp: unsafe fn(T::Simd, T::Simd) -> T::SimdCmp,
) {
    debug_assert_eq!(lhs.len, rhs.len);
    if lhs.len == 0 {
        return;
    }

    let mask_low = vld1q_u16(MASK_16_LOW.as_ptr());
    let mask_high = vld1q_u16(MASK_16_HIGH.as_ptr());
    let num_loops = roundup_loops(lhs.len, 16);
    let mut lhs_ptr = lhs.ptr.as_ptr();
    let mut rhs_ptr = rhs.ptr.as_ptr();
    let dst = dst as *mut u16;

    for simd_index in 0..num_loops {
        let lhs_0 = T::LOAD(lhs_ptr);
        let lhs_1 = T::LOAD(lhs_ptr.add(8));
        let rhs_0 = T::LOAD(rhs_ptr);
        let rhs_1 = T::LOAD(rhs_ptr.add(8));
        let mask =
            pack_half_word_cmp::<NOT>(cmp(lhs_0, rhs_0), cmp(lhs_1, rhs_1), mask_low, mask_high);

        *dst.add(simd_index) = mask as _;
        lhs_ptr = lhs_ptr.add(16);
        rhs_ptr = rhs_ptr.add(16);
    }
}

trait WordIntrinsic: IntrinsicExt<SimdCmp = uint32x4_t> {}

impl WordIntrinsic for i32 {}
impl WordIntrinsic for u32 {}
impl WordIntrinsic for f32 {}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn pack_word_cmp<const NOT: bool>(
    cmp_0: uint32x4_t,
    cmp_1: uint32x4_t,
    cmp_2: uint32x4_t,
    cmp_3: uint32x4_t,
    mask_low: uint16x8_t,
    mask_high: uint16x8_t,
) -> u16 {
    let uzp_0 = vuzp1q_u16(vreinterpretq_u16_u32(cmp_0), vreinterpretq_u16_u32(cmp_1));
    let uzp_1 = vuzp1q_u16(vreinterpretq_u16_u32(cmp_2), vreinterpretq_u16_u32(cmp_3));
    let uzp_0 = vandq_u16(uzp_0, mask_low);
    let uzp_1 = vandq_u16(uzp_1, mask_high);
    let mask = vaddvq_u16(vaddq_u16(uzp_0, uzp_1));
    if NOT {
        mask ^ u16::MAX
    } else {
        mask
    }
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn word_cmp_scalar<const NOT: bool, T: WordIntrinsic>(
    lhs: &AlignedVec<T>,
    rhs: T,
    dst: *mut BitStore,
    cmp: unsafe fn(T::Simd, T::Simd) -> T::SimdCmp,
) {
    if lhs.len == 0 {
        return;
    }

    let rhs = T::BROADCAST(rhs);
    let mask_low = vld1q_u16(MASK_16_LOW.as_ptr());
    let mask_high = vld1q_u16(MASK_16_HIGH.as_ptr());
    let num_loops = roundup_loops(lhs.len, 16);
    let mut lhs_ptr = lhs.ptr.as_ptr();
    let dst = dst as *mut u16;

    for simd_index in 0..num_loops {
        let lhs_0 = T::LOAD(lhs_ptr);
        let lhs_1 = T::LOAD(lhs_ptr.add(4));
        let lhs_2 = T::LOAD(lhs_ptr.add(8));
        let lhs_3 = T::LOAD(lhs_ptr.add(12));

        let cmp_0 = cmp(lhs_0, rhs);
        let cmp_1 = cmp(lhs_1, rhs);
        let cmp_2 = cmp(lhs_2, rhs);
        let cmp_3 = cmp(lhs_3, rhs);
        let mask = pack_word_cmp::<NOT>(cmp_0, cmp_1, cmp_2, cmp_3, mask_low, mask_high);
        *dst.add(simd_index) = mask;
        lhs_ptr = lhs_ptr.add(16);
    }
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn word_cmp<const NOT: bool, T: WordIntrinsic>(
    lhs: &AlignedVec<T>,
    rhs: &AlignedVec<T>,
    dst: *mut BitStore,
    cmp: unsafe fn(T::Simd, T::Simd) -> T::SimdCmp,
) {
    debug_assert_eq!(lhs.len, rhs.len);
    if lhs.len == 0 {
        return;
    }

    let mask_low = vld1q_u16(MASK_16_LOW.as_ptr());
    let mask_high = vld1q_u16(MASK_16_HIGH.as_ptr());
    let num_loops = roundup_loops(lhs.len, 16);
    let mut lhs_ptr = lhs.ptr.as_ptr();
    let mut rhs_ptr = rhs.ptr.as_ptr();
    let dst = dst as *mut u16;

    for simd_index in 0..num_loops {
        let lhs_0 = T::LOAD(lhs_ptr);
        let lhs_1 = T::LOAD(lhs_ptr.add(4));
        let lhs_2 = T::LOAD(lhs_ptr.add(8));
        let lhs_3 = T::LOAD(lhs_ptr.add(12));
        let rhs_0 = T::LOAD(rhs_ptr);
        let rhs_1 = T::LOAD(rhs_ptr.add(4));
        let rhs_2 = T::LOAD(rhs_ptr.add(8));
        let rhs_3 = T::LOAD(rhs_ptr.add(12));

        let cmp_0 = cmp(lhs_0, rhs_0);
        let cmp_1 = cmp(lhs_1, rhs_1);
        let cmp_2 = cmp(lhs_2, rhs_2);
        let cmp_3 = cmp(lhs_3, rhs_3);

        let mask = pack_word_cmp::<NOT>(cmp_0, cmp_1, cmp_2, cmp_3, mask_low, mask_high);
        *dst.add(simd_index) = mask;
        lhs_ptr = lhs_ptr.add(16);
        rhs_ptr = rhs_ptr.add(16);
    }
}

trait DoubleWordIntrinsic: IntrinsicExt<SimdCmp = uint64x2_t> {}

impl DoubleWordIntrinsic for i64 {}
impl DoubleWordIntrinsic for u64 {}
impl DoubleWordIntrinsic for f64 {}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn pack_double_word_cmp<const NOT: bool>(
    cmp_0: uint64x2_t,
    cmp_1: uint64x2_t,
    cmp_2: uint64x2_t,
    cmp_3: uint64x2_t,
    mask_low: uint16x8_t,
) -> u8 {
    let uzp_0 = vuzp1q_u32(vreinterpretq_u32_u64(cmp_0), vreinterpretq_u32_u64(cmp_1));
    let uzp_1 = vuzp1q_u32(vreinterpretq_u32_u64(cmp_2), vreinterpretq_u32_u64(cmp_3));
    let uzp = vuzp1q_u16(vreinterpretq_u16_u32(uzp_0), vreinterpretq_u16_u32(uzp_1));
    let mask = vaddvq_u16(vandq_u16(uzp, mask_low)) as u8;

    if NOT {
        mask ^ u8::MAX
    } else {
        mask
    }
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn double_word_cmp_scalar<const NOT: bool, T: DoubleWordIntrinsic>(
    lhs: &AlignedVec<T>,
    rhs: T,
    dst: *mut BitStore,
    cmp: unsafe fn(T::Simd, T::Simd) -> T::SimdCmp,
) {
    if lhs.len == 0 {
        return;
    }

    let rhs = T::BROADCAST(rhs);
    let num_loops = roundup_loops(lhs.len, 8);
    let mask_low = vld1q_u16(MASK_16_LOW.as_ptr());
    let mut lhs_ptr = lhs.ptr.as_ptr();
    let dst = dst as *mut u8;

    for simd_index in 0..num_loops {
        let lhs_0 = T::LOAD(lhs_ptr);
        let lhs_1 = T::LOAD(lhs_ptr.add(2));
        let lhs_2 = T::LOAD(lhs_ptr.add(4));
        let lhs_3 = T::LOAD(lhs_ptr.add(6));

        let cmp_0 = cmp(lhs_0, rhs);
        let cmp_1 = cmp(lhs_1, rhs);
        let cmp_2 = cmp(lhs_2, rhs);
        let cmp_3 = cmp(lhs_3, rhs);

        let mask = pack_double_word_cmp::<NOT>(cmp_0, cmp_1, cmp_2, cmp_3, mask_low);

        *dst.add(simd_index) = mask;
        lhs_ptr = lhs_ptr.add(8);
    }
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn double_word_cmp<const NOT: bool, T: DoubleWordIntrinsic>(
    lhs: &AlignedVec<T>,
    rhs: &AlignedVec<T>,
    dst: *mut BitStore,
    cmp: unsafe fn(T::Simd, T::Simd) -> T::SimdCmp,
) {
    debug_assert_eq!(lhs.len, rhs.len);
    if lhs.len == 0 {
        return;
    }

    let num_loops = roundup_loops(lhs.len, 8);
    let mask_low = vld1q_u16(MASK_16_LOW.as_ptr());
    let mut lhs_ptr = lhs.ptr.as_ptr();
    let mut rhs_ptr = rhs.ptr.as_ptr();
    let dst = dst as *mut u8;

    for simd_index in 0..num_loops {
        let lhs_0 = T::LOAD(lhs_ptr);
        let lhs_1 = T::LOAD(lhs_ptr.add(2));
        let lhs_2 = T::LOAD(lhs_ptr.add(4));
        let lhs_3 = T::LOAD(lhs_ptr.add(6));

        let rhs_0 = T::LOAD(rhs_ptr);
        let rhs_1 = T::LOAD(rhs_ptr.add(2));
        let rhs_2 = T::LOAD(rhs_ptr.add(4));
        let rhs_3 = T::LOAD(rhs_ptr.add(6));

        let cmp_0 = cmp(lhs_0, rhs_0);
        let cmp_1 = cmp(lhs_1, rhs_1);
        let cmp_2 = cmp(lhs_2, rhs_2);
        let cmp_3 = cmp(lhs_3, rhs_3);
        let mask = pack_double_word_cmp::<NOT>(cmp_0, cmp_1, cmp_2, cmp_3, mask_low);

        *dst.add(simd_index) = mask;
        lhs_ptr = lhs_ptr.add(8);
        rhs_ptr = lhs_ptr.add(8);
    }
}

macro_rules! cmp {
    ($ty:ty, $cmp_scalar_func:ident, $cmp_arrays_func:ident) => {
        paste::paste! {
            // Array cmp scalar
            #[target_feature(enable = "neon")]
            pub(super) unsafe fn [<eq_scalar_ $ty _neon>](array: &AlignedVec<$ty>, scalar: $ty, dst: *mut BitStore) {
                $cmp_scalar_func::<false, $ty>(array, scalar, dst, $ty::EQ);
            }

            #[target_feature(enable = "neon")]
            pub(super) unsafe fn [<ne_scalar_ $ty _neon>](array: &AlignedVec<$ty>, scalar: $ty, dst: *mut BitStore) {
                $cmp_scalar_func::<true, $ty>(array, scalar, dst, $ty::EQ);
            }

            #[target_feature(enable = "neon")]
            pub(super) unsafe fn [<gt_scalar_ $ty _neon>](array: &AlignedVec<$ty>, scalar: $ty, dst: *mut BitStore) {
                $cmp_scalar_func::<false, $ty>(array, scalar, dst, $ty::GT);
            }

            #[target_feature(enable = "neon")]
            pub(super) unsafe fn [<ge_scalar_ $ty _neon>](array: &AlignedVec<$ty>, scalar: $ty, dst: *mut BitStore) {
                $cmp_scalar_func::<false, $ty>(array, scalar, dst, $ty::GE);
            }

            #[target_feature(enable = "neon")]
            pub(super) unsafe fn [<lt_scalar_ $ty _neon>](array: &AlignedVec<$ty>, scalar: $ty, dst: *mut BitStore) {
                $cmp_scalar_func::<false, $ty>(array, scalar, dst, $ty::LT);
            }

            #[target_feature(enable = "neon")]
            pub(super) unsafe fn [<le_scalar_ $ty _neon>](array: &AlignedVec<$ty>, scalar: $ty, dst: *mut BitStore) {
                $cmp_scalar_func::<false, $ty>(array, scalar, dst, $ty::LE);
            }

            // Array cmp array

            #[target_feature(enable = "neon")]
            pub(super) unsafe fn [<eq_ $ty _neon>](lhs: &AlignedVec<$ty>, rhs: &AlignedVec<$ty>, dst: *mut BitStore) {
                $cmp_arrays_func::<false, $ty>(lhs, rhs, dst, $ty::EQ);
            }

            #[target_feature(enable = "neon")]
            pub(super) unsafe fn [<ne_ $ty _neon>](lhs: &AlignedVec<$ty>, rhs: &AlignedVec<$ty>, dst: *mut BitStore) {
                $cmp_arrays_func::<true, $ty>(lhs, rhs, dst, $ty::EQ);
            }

            #[target_feature(enable = "neon")]
            pub(super) unsafe fn [<gt_ $ty _neon>](lhs: &AlignedVec<$ty>, rhs: &AlignedVec<$ty>, dst: *mut BitStore) {
                $cmp_arrays_func::<false, $ty>(lhs, rhs, dst, $ty::GT);
            }

            #[target_feature(enable = "neon")]
            pub(super) unsafe fn [<ge_ $ty _neon>](lhs: &AlignedVec<$ty>, rhs: &AlignedVec<$ty>, dst: *mut BitStore) {
                $cmp_arrays_func::<false, $ty>(lhs, rhs, dst, $ty::GE);
            }

            #[target_feature(enable = "neon")]
            pub(super) unsafe fn [<lt_ $ty _neon>](lhs: &AlignedVec<$ty>, rhs: &AlignedVec<$ty>, dst: *mut BitStore) {
                $cmp_arrays_func::<false, $ty>(lhs, rhs, dst, $ty::LT);
            }

            #[target_feature(enable = "neon")]
            pub(super) unsafe fn [<le_ $ty _neon>](lhs: &AlignedVec<$ty>, rhs: &AlignedVec<$ty>, dst: *mut BitStore) {
                $cmp_arrays_func::<false, $ty>(lhs, rhs, dst, $ty::LE);
            }
        }
    };
}

cmp!(i8, byte_cmp_scalar, byte_cmp);
cmp!(u8, byte_cmp_scalar, byte_cmp);
cmp!(i16, half_word_cmp_scalar, half_word_cmp);
cmp!(u16, half_word_cmp_scalar, half_word_cmp);
cmp!(i32, word_cmp_scalar, word_cmp);
cmp!(u32, word_cmp_scalar, word_cmp);
cmp!(f32, word_cmp_scalar, word_cmp);
cmp!(i64, double_word_cmp_scalar, double_word_cmp);
cmp!(u64, double_word_cmp_scalar, double_word_cmp);
cmp!(f64, double_word_cmp_scalar, double_word_cmp);

#[repr(align(16))]
struct AlignedArray([i64; 2]);

impl AlignedArray {
    #[inline]
    fn as_ptr(&self) -> *const i64 {
        self.0.as_ptr()
    }
}

/// Multiply the elements in the array than compare with the scalar
#[target_feature(enable = "neon")]
#[inline]
unsafe fn timestamp_cmp_scalar_neon<const AM: i64, const NOT: bool>(
    array: &AlignedVec<i64>,
    scalar: i64,
    dst: *mut BitStore,
    cmp_func: unsafe fn(int64x2_t, int64x2_t) -> uint64x2_t,
) {
    if array.is_empty() {
        return;
    }

    let scalar = vdupq_n_s64(scalar);
    let num_loops = roundup_loops(array.len(), 8);
    let mask_low = vld1q_u16(MASK_16_LOW.as_ptr());
    let mut lhs_ptr = array.ptr.as_ptr();
    let dst = dst as *mut u8;

    for simd_index in 0..num_loops {
        // neon does not support multiply s64. Using non-simd version for multiply
        // if `AM == 1`, compiler will optimize the following code to load!
        let lhs_0 = vld1q_s64(AlignedArray([(*lhs_ptr) * AM, (*lhs_ptr.add(1)) * AM]).as_ptr());
        let lhs_1 =
            vld1q_s64(AlignedArray([(*lhs_ptr.add(2)) * AM, (*lhs_ptr.add(3)) * AM]).as_ptr());
        let lhs_2 =
            vld1q_s64(AlignedArray([(*lhs_ptr.add(4)) * AM, (*lhs_ptr.add(5)) * AM]).as_ptr());
        let lhs_3 =
            vld1q_s64(AlignedArray([(*lhs_ptr.add(6)) * AM, (*lhs_ptr.add(7)) * AM]).as_ptr());

        // Compare with simd
        let cmp_0 = cmp_func(lhs_0, scalar);
        let cmp_1 = cmp_func(lhs_1, scalar);
        let cmp_2 = cmp_func(lhs_2, scalar);
        let cmp_3 = cmp_func(lhs_3, scalar);

        let mask = pack_double_word_cmp::<NOT>(cmp_0, cmp_1, cmp_2, cmp_3, mask_low);

        *dst.add(simd_index) = mask;
        lhs_ptr = lhs_ptr.add(8);
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn timestamp_eq_scalar_neon<const AM: i64, const SM: i64>(
    array: &AlignedVec<i64>,
    scalar: i64,
    dst: *mut BitStore,
) {
    let scalar = scalar * SM;
    timestamp_cmp_scalar_neon::<AM, false>(array, scalar, dst, vceqq_s64)
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn timestamp_ne_scalar_neon<const AM: i64, const SM: i64>(
    array: &AlignedVec<i64>,
    scalar: i64,
    dst: *mut BitStore,
) {
    let scalar = scalar * SM;
    timestamp_cmp_scalar_neon::<AM, true>(array, scalar, dst, vceqq_s64)
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn timestamp_gt_scalar_neon<const AM: i64, const SM: i64>(
    array: &AlignedVec<i64>,
    scalar: i64,
    dst: *mut BitStore,
) {
    let scalar = scalar * SM;
    timestamp_cmp_scalar_neon::<AM, false>(array, scalar, dst, vcgtq_s64)
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn timestamp_ge_scalar_neon<const AM: i64, const SM: i64>(
    array: &AlignedVec<i64>,
    scalar: i64,
    dst: *mut BitStore,
) {
    let scalar = scalar * SM;
    timestamp_cmp_scalar_neon::<AM, false>(array, scalar, dst, vcgeq_s64)
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn timestamp_lt_scalar_neon<const AM: i64, const SM: i64>(
    array: &AlignedVec<i64>,
    scalar: i64,
    dst: *mut BitStore,
) {
    let scalar = scalar * SM;
    timestamp_cmp_scalar_neon::<AM, false>(array, scalar, dst, vcltq_s64)
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn timestamp_le_scalar_neon<const AM: i64, const SM: i64>(
    array: &AlignedVec<i64>,
    scalar: i64,
    dst: *mut BitStore,
) {
    let scalar = scalar * SM;
    timestamp_cmp_scalar_neon::<AM, false>(array, scalar, dst, vcleq_s64)
}

/// Multiply the elements in the array than compare with the scalar
#[target_feature(enable = "neon")]
#[inline]
unsafe fn timestamp_cmp_neon<const LM: i64, const RM: i64, const NOT: bool>(
    lhs: &AlignedVec<i64>,
    rhs: &AlignedVec<i64>,
    dst: *mut BitStore,
    cmp_func: unsafe fn(int64x2_t, int64x2_t) -> uint64x2_t,
) {
    debug_assert_eq!(lhs.len(), rhs.len());
    if lhs.is_empty() {
        return;
    }

    let num_loops = roundup_loops(lhs.len(), 8);
    let mask_low = vld1q_u16(MASK_16_LOW.as_ptr());
    let mut lhs_ptr = lhs.ptr.as_ptr();
    let mut rhs_ptr = rhs.ptr.as_ptr();
    let dst = dst as *mut u8;

    for simd_index in 0..num_loops {
        // neon does not support multiply s64. Using non-simd version for multiply
        // if `AM == 1`, compiler will optimize the following code to load!
        let lhs_0 = vld1q_s64(AlignedArray([(*lhs_ptr) * LM, (*lhs_ptr.add(1)) * LM]).as_ptr());
        let lhs_1 =
            vld1q_s64(AlignedArray([(*lhs_ptr.add(2)) * LM, (*lhs_ptr.add(3)) * LM]).as_ptr());
        let lhs_2 =
            vld1q_s64(AlignedArray([(*lhs_ptr.add(4)) * LM, (*lhs_ptr.add(5)) * LM]).as_ptr());
        let lhs_3 =
            vld1q_s64(AlignedArray([(*lhs_ptr.add(6)) * LM, (*lhs_ptr.add(7)) * LM]).as_ptr());

        // neon does not support multiply s64. Using non-simd version for multiply
        let rhs_0 = vld1q_s64(AlignedArray([(*rhs_ptr) * RM, (*rhs_ptr.add(1)) * RM]).as_ptr());
        let rhs_1 =
            vld1q_s64(AlignedArray([(*rhs_ptr.add(2)) * RM, (*rhs_ptr.add(3)) * RM]).as_ptr());
        let rhs_2 =
            vld1q_s64(AlignedArray([(*rhs_ptr.add(4)) * RM, (*rhs_ptr.add(5)) * RM]).as_ptr());
        let rhs_3 =
            vld1q_s64(AlignedArray([(*rhs_ptr.add(6)) * RM, (*rhs_ptr.add(7)) * RM]).as_ptr());

        // Compare with simd
        let cmp_0 = cmp_func(lhs_0, rhs_0);
        let cmp_1 = cmp_func(lhs_1, rhs_1);
        let cmp_2 = cmp_func(lhs_2, rhs_2);
        let cmp_3 = cmp_func(lhs_3, rhs_3);

        let mask = pack_double_word_cmp::<NOT>(cmp_0, cmp_1, cmp_2, cmp_3, mask_low);

        *dst.add(simd_index) = mask;
        lhs_ptr = lhs_ptr.add(8);
        rhs_ptr = rhs_ptr.add(8);
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn timestamp_eq_neon<const LM: i64, const RM: i64>(
    lhs: &AlignedVec<i64>,
    rhs: &AlignedVec<i64>,
    dst: *mut BitStore,
) {
    timestamp_cmp_neon::<LM, RM, false>(lhs, rhs, dst, vceqq_s64);
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn timestamp_ne_neon<const LM: i64, const RM: i64>(
    lhs: &AlignedVec<i64>,
    rhs: &AlignedVec<i64>,
    dst: *mut BitStore,
) {
    timestamp_cmp_neon::<LM, RM, true>(lhs, rhs, dst, vceqq_s64);
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn timestamp_gt_neon<const LM: i64, const RM: i64>(
    lhs: &AlignedVec<i64>,
    rhs: &AlignedVec<i64>,
    dst: *mut BitStore,
) {
    timestamp_cmp_neon::<LM, RM, false>(lhs, rhs, dst, vcgtq_s64);
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn timestamp_ge_neon<const LM: i64, const RM: i64>(
    lhs: &AlignedVec<i64>,
    rhs: &AlignedVec<i64>,
    dst: *mut BitStore,
) {
    timestamp_cmp_neon::<LM, RM, false>(lhs, rhs, dst, vcgeq_s64);
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn timestamp_lt_neon<const LM: i64, const RM: i64>(
    lhs: &AlignedVec<i64>,
    rhs: &AlignedVec<i64>,
    dst: *mut BitStore,
) {
    timestamp_cmp_neon::<LM, RM, false>(lhs, rhs, dst, vcltq_s64);
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn timestamp_le_neon<const LM: i64, const RM: i64>(
    lhs: &AlignedVec<i64>,
    rhs: &AlignedVec<i64>,
    dst: *mut BitStore,
) {
    timestamp_cmp_neon::<LM, RM, false>(lhs, rhs, dst, vcleq_s64);
}

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

    #[test]
    fn test_timestamp_equal_scalar() {
        let lhs = AlignedVec::from_slice(&[10, 8, -3, -9, 10, -10, 3, 2, 15, 10]);
        let rhs = 10_000;
        cmp_assert!(&lhs, rhs, timestamp_eq_scalar_neon::<1000, 1>, &[0x211]);
        cmp_assert!(&lhs, rhs, timestamp_ne_scalar_neon::<1000, 1>, &[!0x211]);
    }

    #[test]
    fn test_timestamp_order_scalar() {
        let lhs = AlignedVec::from_slice(&[10, 8, 33, -9, 10, 11, 3, 2, 10, 7]);
        let rhs = 10_000;
        cmp_assert!(&lhs, rhs, timestamp_gt_scalar_neon::<1000, 1>, &[0x024]);
        cmp_assert!(&lhs, rhs, timestamp_ge_scalar_neon::<1000, 1>, &[0x135]);
        cmp_assert!(&lhs, rhs, timestamp_lt_scalar_neon::<1000, 1>, &[!0x135]);
        cmp_assert!(&lhs, rhs, timestamp_le_scalar_neon::<1000, 1>, &[!0x024]);
    }

    #[test]
    fn test_equal_i8() {
        let lhs = AlignedVec::from_slice(&[-5_i8, 9, 4, -7, -111, 111, -5, -4, 4, 4]);
        let rhs = AlignedVec::from_slice(&[-5_i8, 0, 5, -7, 111, -111, -5, 1, 0, 4]);
        cmp_assert!(&lhs, &rhs, eq_i8_neon, &[0x0249]);
        cmp_assert!(&lhs, &rhs, ne_i8_neon, &[!0x0249]);
    }
}
