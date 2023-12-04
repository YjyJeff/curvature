//! SIMD based on `sse2` instructions
use crate::aligned_vec::AlignedVec;
use crate::bitmap::BitStore;
use crate::utils::roundup_loops;
use paste::paste;
use std::mem::transmute;

x86_target_use!(
    __m128i,
    __m128,
    __m128d,
    _mm_load_si128,
    _mm_movemask_epi8,
    _mm_set1_epi8,
    _mm_xor_si128,
    _mm_set1_epi16,
    _mm_packs_epi16,
    _mm_packs_epi32,
    _mm_set1_epi32,
    _mm_set1_ps,
    _mm_load_ps,
    _mm_set1_pd,
    _mm_load_pd,
    _mm_castps_si128,
    _mm_shuffle_ps,
    _mm_castpd_ps,
    _mm_cmpeq_epi8,
    _mm_cmpgt_epi8,
    _mm_cmplt_epi8,
    _mm_cmpeq_epi16,
    _mm_cmpgt_epi16,
    _mm_cmplt_epi16,
    _mm_cmpeq_epi32,
    _mm_cmpgt_epi32,
    _mm_cmplt_epi32,
    _mm_cmpeq_ps,
    _mm_cmpneq_ps,
    _mm_cmpgt_ps,
    _mm_cmpge_ps,
    _mm_cmplt_ps,
    _mm_cmple_ps,
    _mm_cmpeq_pd,
    _mm_cmpneq_pd,
    _mm_cmpgt_pd,
    _mm_cmpge_pd,
    _mm_cmplt_pd,
    _mm_cmple_pd
);

macro_rules! cmp_int_scalar_sse2 {
    ($int_ty:ty, $suffix:ident) => {
        cmp_int_scalar!("sse2", _mm, $int_ty, $suffix);
    };
    ($uint_ty:ty, $int_ty:ty, $suffix:ident) => {
        cmp_int_scalar!("sse2", _mm, $uint_ty, $int_ty, $suffix);
    };
}

macro_rules! cmp_float_scalar_sse2 {
    ($ty:ty, $suffix:ident) => {
        cmp_float_scalar!("sse2", _mm, $ty, $suffix);
    };
}

/// Compare array of i8 with the given i8
///
/// # Safety
///
/// - Caller should ensure `sse2` is supported
/// - dst should have adequate space
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn cmp_scalar_i8_sse2<const NOT: bool, const FLIP_SIGN: bool, F>(
    lhs: &AlignedVec<i8>,
    rhs: i8,
    dst: *mut u16,
    cmp_func: F,
) where
    F: Fn(__m128i, __m128i) -> __m128i,
{
    const SHIFT_BITS: usize = 4;
    const BASE: usize = 1 << SHIFT_BITS;

    if lhs.len == 0 {
        return;
    }

    let num_loops = roundup_loops(lhs.len, BASE);

    // Broadcast rhs to 128 bit
    let mut rhs = _mm_set1_epi8(rhs);

    // These two will be optimized away based on the NOT and FLIP_SIGN
    let flip_sign = _mm_set1_epi8(i8::MIN);
    if FLIP_SIGN {
        rhs = _mm_xor_si128(rhs, flip_sign);
    }

    for simd_index in 0..num_loops {
        // Load 16 i8 into __m128i. Ptr in [`AlignedVec`] is guaranteed to be
        // cache line aligned, therefore, we can use load aligned instruction here. What's
        // more, [`AlignedVec`] is padded to 64 bytes. Even if `lhs.len` is not
        // divisible by 16, load data still succeed and contains some uninitialized data.
        let mut lhs = _mm_load_si128(lhs.ptr.as_ptr().add(simd_index << SHIFT_BITS) as _);
        // If is static, will be optimized away
        if FLIP_SIGN {
            lhs = _mm_xor_si128(lhs, flip_sign);
        }
        // Compare lhs with rhs
        let cmp = cmp_func(lhs, rhs);

        // 16(bit) * 8(element) => 8(bit_per_element) * 2
        // padding the highest bits from 16..32 with zero
        let mut mask = _mm_movemask_epi8(cmp);
        if NOT {
            mask ^= -1;
        }

        // as u16 will truncate the duplicate part and only contains 0..16
        *dst.add(simd_index) = mask as u16;
    }
}

/// Compare array of i16 with the given i16. Adapted from [Getting Bitmasks from SSE Vector Comparisons]
///
///
/// # Safety
///
/// - Caller should ensure `sse2` is supported
/// - dst should have adequate space
///
/// [Getting Bitmasks from SSE Vector Comparisons]: https://giannitedesco.github.io/2019/03/08/simd-cmp-bitmasks.html
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn cmp_scalar_i16_sse2<const NOT: bool, const FLIP_SIGN: bool, F>(
    lhs: &AlignedVec<i16>,
    rhs: i16,
    dst: *mut u16,
    cmp_func: F,
) where
    F: Fn(__m128i, __m128i) -> __m128i,
{
    const SHIFT_BITS: usize = 4;
    const BASE: usize = 1 << SHIFT_BITS;

    if lhs.len == 0 {
        return;
    }

    let num_loops = roundup_loops(lhs.len, BASE);

    // Broadcast rhs to 128 bit
    let mut rhs = _mm_set1_epi16(rhs);

    // These two will be optimized away based on the NOT and FLIP_SIGN
    let flip_sign = _mm_set1_epi16(i16::MIN);
    if FLIP_SIGN {
        rhs = _mm_xor_si128(rhs, flip_sign);
    }

    for simd_index in 0..num_loops {
        // Load 8 i16 into __m128i. Ptr in [`CacheAlineAlignedVec`] is guaranteed to be
        // cache line aligned, therefore, we can use load aligned instruction here. What's
        // more, [`CacheAlignedVec`] is padded to 64 bytes. Even if `lhs.len` is not
        // divisible by 8, load data still succeed and contains some uninitialized data.
        let mut lhs_0 = _mm_load_si128(lhs.ptr.as_ptr().add(simd_index << SHIFT_BITS) as _);
        let mut lhs_1 = _mm_load_si128(lhs.ptr.as_ptr().add((simd_index << SHIFT_BITS) + 8) as _);
        // If is static, will be optimized away
        if FLIP_SIGN {
            lhs_0 = _mm_xor_si128(lhs_0, flip_sign);
            lhs_1 = _mm_xor_si128(lhs_1, flip_sign);
        }
        // Compare lhs with rhs
        let cmp_0 = cmp_func(lhs_0, rhs);
        let cmp_1 = cmp_func(lhs_1, rhs);

        // Now we need to convert cmp_results to bitmap
        // 16(bit) x 8(element) => 8(bit) x 16(element) = 8(bit) * 8(element) x 2
        let packed = _mm_packs_epi16(cmp_0, cmp_1);

        // 8(bit) * 8(element) * 2 => 8(bit_per_element) * 2
        // padding the highest bits from 16..32 with zero
        let mut mask = _mm_movemask_epi8(packed);
        if NOT {
            mask ^= -1
        }

        // as u16 will perform truncate, only contains 0..16
        *dst.add(simd_index) = mask as u16;
    }
}

/// Compare array of i32 with the given i32
///
/// # Safety
///
/// - Caller should ensure `sse2` is supported
/// - dst should have adequate space
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn cmp_scalar_i32_sse2<const NOT: bool, const FLIP_SIGN: bool, F>(
    lhs: &AlignedVec<i32>,
    rhs: i32,
    dst: *mut u16,
    cmp_func: F,
) where
    F: Fn(__m128i, __m128i) -> __m128i,
{
    const SHIFT_BITS: usize = 4;
    const BASE: usize = 1 << SHIFT_BITS;

    if lhs.len == 0 {
        return;
    }

    let num_loops = roundup_loops(lhs.len, BASE);

    // Broadcast rhs to 128 bit
    let mut rhs = _mm_set1_epi32(rhs);

    // These two will be optimized away based on the NOT and FLIP_SIGN
    let flip_sign = _mm_set1_epi32(i32::MIN);
    if FLIP_SIGN {
        rhs = _mm_xor_si128(rhs, flip_sign);
    }

    for simd_index in 0..num_loops {
        // Load 4 i32 into __m128i. Ptr in [`CacheAlineAlignedVec`] is guaranteed to be
        // cache line aligned, therefore, we can use load aligned instruction here. What's
        // more, [`CacheAlignedVec`] is padded to 64 bytes. Even if `lhs.len` is not
        // divisible by 4, load data still succeed and contains some uninitialized data.
        let mut lhs_0 = _mm_load_si128(lhs.ptr.as_ptr().add(simd_index << SHIFT_BITS) as _);
        let mut lhs_1 = _mm_load_si128(lhs.ptr.as_ptr().add((simd_index << SHIFT_BITS) + 4) as _);
        let mut lhs_2 = _mm_load_si128(lhs.ptr.as_ptr().add((simd_index << SHIFT_BITS) + 8) as _);
        let mut lhs_3 = _mm_load_si128(lhs.ptr.as_ptr().add((simd_index << SHIFT_BITS) + 12) as _);

        // If is static, will be optimized away
        if FLIP_SIGN {
            lhs_0 = _mm_xor_si128(lhs_0, flip_sign);
            lhs_1 = _mm_xor_si128(lhs_1, flip_sign);
            lhs_2 = _mm_xor_si128(lhs_2, flip_sign);
            lhs_3 = _mm_xor_si128(lhs_3, flip_sign);
        }

        // Compare lhs with rhs
        let cmp_0 = cmp_func(lhs_0, rhs);
        let cmp_1 = cmp_func(lhs_1, rhs);

        // Following cmp may cause unnecessary computation, but I assume it is rare
        // Padding to 64 bytes allows we can do it !!! What's more, it can make computation
        // about 16% faster
        let cmp_2 = cmp_func(lhs_2, rhs);
        let cmp_3 = cmp_func(lhs_3, rhs);

        // Now we need to convert cmp_result to bitmap
        // 32(bit) x 4(element) => 16(bit) x 8(element)
        let packed_0 = _mm_packs_epi32(cmp_0, cmp_1);
        let packed_1 = _mm_packs_epi32(cmp_2, cmp_3);

        let packed = _mm_packs_epi16(packed_0, packed_1);

        let mut mask = _mm_movemask_epi8(packed);

        if NOT {
            mask ^= -1;
        }

        // as u16 will perform truncate, only contains 0..16
        *dst.add(simd_index) = mask as u16;
    }
}

/// Compare array of f32 with the given f32
///
/// # Safety
///
/// - Caller should ensure `sse2` is supported
/// - dst should have adequate space
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn cmp_scalar_f32_sse2<F>(lhs: &AlignedVec<f32>, rhs: f32, dst: *mut u16, cmp_func: F)
where
    F: Fn(__m128, __m128) -> __m128,
{
    const SHIFT_BITS: usize = 4;
    const BASE: usize = 1 << SHIFT_BITS;

    if lhs.len == 0 {
        return;
    }

    let num_loops = roundup_loops(lhs.len, BASE);

    let rhs = _mm_set1_ps(rhs);

    for simd_index in 0..num_loops {
        let lhs_0 = _mm_load_ps(lhs.ptr.as_ptr().add(simd_index << SHIFT_BITS));
        let lhs_1 = _mm_load_ps(lhs.ptr.as_ptr().add((simd_index << SHIFT_BITS) + 4) as _);
        let lhs_2 = _mm_load_ps(lhs.ptr.as_ptr().add((simd_index << SHIFT_BITS) + 8) as _);
        let lhs_3 = _mm_load_ps(lhs.ptr.as_ptr().add((simd_index << SHIFT_BITS) + 12) as _);

        let cmp_0 = cmp_func(lhs_0, rhs);
        let cmp_1 = cmp_func(lhs_1, rhs);
        let cmp_2 = cmp_func(lhs_2, rhs);
        let cmp_3 = cmp_func(lhs_3, rhs);

        let packed_0 = _mm_packs_epi32(_mm_castps_si128(cmp_0), _mm_castps_si128(cmp_1));
        let packed_1 = _mm_packs_epi32(_mm_castps_si128(cmp_2), _mm_castps_si128(cmp_3));

        let packded = _mm_packs_epi16(packed_0, packed_1);
        let mask = _mm_movemask_epi8(packded);

        // as u16 will perform truncate, only contains 0..16
        *dst.add(simd_index) = mask as u16;
    }
}

/// Compare array of f64 with the given f64
///
/// # Safety
///
/// - Caller should ensure `sse2` is supported
/// - dst should have adequate space
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn cmp_scalar_f64_sse2<F>(lhs: &AlignedVec<f64>, rhs: f64, dst: *mut u16, cmp_func: F)
where
    F: Fn(__m128d, __m128d) -> __m128d,
{
    const SHIFT_BITS: usize = 3;
    const BASE: usize = 1 << SHIFT_BITS;
    const CTRL_BIT: i32 = 0x88;

    if lhs.len == 0 {
        return;
    }

    let num_loops = roundup_loops(lhs.len, BASE);

    let lhs_ptr = lhs.ptr.as_ptr();
    let rhs = _mm_set1_pd(rhs);

    let load_and_compare = |base: usize| {
        let lhs_0 = _mm_load_pd(lhs_ptr.add(base));
        let lhs_1 = _mm_load_pd(lhs_ptr.add(base + 2) as _);
        let lhs_2 = _mm_load_pd(lhs_ptr.add(base + 4) as _);
        let lhs_3 = _mm_load_pd(lhs_ptr.add(base + 6) as _);

        let cmp_0 = cmp_func(lhs_0, rhs);
        let cmp_1 = cmp_func(lhs_1, rhs);
        let cmp_2 = cmp_func(lhs_2, rhs);
        let cmp_3 = cmp_func(lhs_3, rhs);

        let shuffle_0 = _mm_shuffle_ps::<CTRL_BIT>(_mm_castpd_ps(cmp_0), _mm_castpd_ps(cmp_1));
        let shuffle_1 = _mm_shuffle_ps::<CTRL_BIT>(_mm_castpd_ps(cmp_2), _mm_castpd_ps(cmp_3));

        _mm_packs_epi32(_mm_castps_si128(shuffle_0), _mm_castps_si128(shuffle_1))
    };

    if num_loops == 1 {
        let packed = load_and_compare(0);
        let mask = _mm_movemask_epi8(_mm_packs_epi16(packed, packed));
        *dst = mask as u16;
    } else {
        for simd_index in (0..num_loops - 1).step_by(2) {
            let packed_0 = load_and_compare(simd_index << SHIFT_BITS);
            let packed_1 = load_and_compare((simd_index + 1) << SHIFT_BITS);
            let mask = _mm_movemask_epi8(_mm_packs_epi16(packed_0, packed_1));
            *dst.add(simd_index >> 1) = mask as u16;
        }
        if num_loops & 0x01 != 0 {
            // Handle last SIMD chunk
            let simd_index = num_loops - 1;
            let packed = load_and_compare(simd_index << SHIFT_BITS);
            let mask = _mm_movemask_epi8(_mm_packs_epi16(packed, packed));
            *dst.add(simd_index >> 1) = mask as u16;
        }
    }
}

cmp_int_scalar_sse2!(i8, epi8);
cmp_int_scalar_sse2!(u8, i8, epi8);
cmp_int_scalar_sse2!(i16, epi16);
cmp_int_scalar_sse2!(u16, i16, epi16);
cmp_int_scalar_sse2!(i32, epi32);
cmp_int_scalar_sse2!(u32, i32, epi32);
cmp_float_scalar_sse2!(f32, ps);
cmp_float_scalar_sse2!(f64, pd);

#[cfg(target_feature = "sse2")]
#[cfg(test)]
mod tests {

    use super::*;
    use crate::aligned_vec::AllocType;
    use crate::bitmap::Bitmap;
    use num_traits::AsPrimitive;
    use std::ops::{Add, Rem};

    fn enumerate_assign<T>(lhs: &mut AlignedVec<T>, len: usize, offset: T)
    where
        T: AllocType + Add<Output = T>,
        usize: AsPrimitive<T>,
    {
        lhs.clear_and_resize(len)
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| {
                *v = i.as_() + offset;
            });
    }

    fn remain_assign<T>(lhs: &mut AlignedVec<T>, len: usize, rem: T)
    where
        T: AllocType + Rem<Output = T>,
        usize: AsPrimitive<T>,
    {
        lhs.clear_and_resize(len)
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| {
                *v = i.as_() % rem;
            });
    }

    #[test]
    fn test_equal_scalar_i8() {
        let len = 37;
        let mut lhs = AlignedVec::<i8>::new();
        remain_assign(&mut lhs, len, 10);
        let rhs = 3;

        // Equal
        cmp_assert!(&lhs, rhs, eq_scalar_i8_sse2_, &[0x0200802008]);
        cmp_assert!(&lhs, rhs, ne_scalar_i8_sse2_, &[!0x0200802008]);
    }

    #[test]
    fn test_order_scalar_i8() {
        let len = 7;
        let mut lhs = AlignedVec::<i8>::new();
        enumerate_assign(&mut lhs, len, -2);

        let rhs = 3;

        cmp_assert!(&lhs, rhs, gt_scalar_i8_sse2_, &[0x40]);
        cmp_assert!(&lhs, rhs, ge_scalar_i8_sse2_, &[0x60]);
        cmp_assert!(&lhs, rhs, lt_scalar_i8_sse2_, &[!0x60]);
        cmp_assert!(&lhs, rhs, le_scalar_i8_sse2_, &[!0x40]);
    }

    #[test]
    fn test_equal_scalar_u8() {
        let len = 37;
        let mut lhs = AlignedVec::<u8>::new();
        remain_assign(&mut lhs, len, 10);

        let rhs = 3;

        cmp_assert!(&lhs, rhs, eq_scalar_u8_sse2_, &[0x0200802008]);
        cmp_assert!(&lhs, rhs, ne_scalar_u8_sse2_, &[!0x0200802008]);
    }

    #[test]
    fn test_order_scalar_u8() {
        let len = 7;
        let mut lhs = AlignedVec::<u8>::new();
        // Contains signed is 0 and 1
        enumerate_assign(&mut lhs, len, 125);

        let rhs = 128;

        cmp_assert!(&lhs, rhs, gt_scalar_u8_sse2_, &[0x70]);
        cmp_assert!(&lhs, rhs, ge_scalar_u8_sse2_, &[0x78]);
        cmp_assert!(&lhs, rhs, lt_scalar_u8_sse2_, &[!0x78]);
        cmp_assert!(&lhs, rhs, le_scalar_u8_sse2_, &[!0x70]);
    }

    #[test]
    fn test_equal_scalar_i16() {
        let len = 37;
        let mut lhs = AlignedVec::<i16>::new();
        remain_assign(&mut lhs, len, 8);

        let rhs = 4;

        cmp_assert!(&lhs, rhs, eq_scalar_i16_sse2_, &[0x1010101010]);
        cmp_assert!(&lhs, rhs, ne_scalar_i16_sse2_, &[!0x1010101010]);
    }

    #[test]
    fn test_order_scalar_i16() {
        let len = 24;
        let mut lhs = AlignedVec::<i16>::new();
        enumerate_assign(&mut lhs, len, -5);

        let rhs = 14;

        cmp_assert!(&lhs, rhs, gt_scalar_i16_sse2_, &[0x00f00000]);
        cmp_assert!(&lhs, rhs, ge_scalar_i16_sse2_, &[0x00f80000]);
        cmp_assert!(&lhs, rhs, lt_scalar_i16_sse2_, &[!0x00f80000]);
        cmp_assert!(&lhs, rhs, le_scalar_i16_sse2_, &[!0x00f00000]);
    }

    #[test]
    fn test_equal_scalar_u16() {
        let len = 37;
        let mut lhs = AlignedVec::<u16>::new();
        remain_assign(&mut lhs, len, 8);

        let rhs = 4;

        cmp_assert!(&lhs, rhs, eq_scalar_u16_sse2_, &[0x1010101010]);
        cmp_assert!(&lhs, rhs, ne_scalar_u16_sse2_, &[!0x1010101010]);
    }

    #[test]
    fn test_order_scalar_u16() {
        let len = 24;
        let mut lhs = AlignedVec::<u16>::new();
        // contains signed 0 and 1
        enumerate_assign(&mut lhs, len, 32760);

        let rhs = 32770;

        cmp_assert!(&lhs, rhs, gt_scalar_u16_sse2_, &[0x00fff800]);
        cmp_assert!(&lhs, rhs, ge_scalar_u16_sse2_, &[0x00fffc00]);
        cmp_assert!(&lhs, rhs, lt_scalar_u16_sse2_, &[!0x00fffc00]);
        cmp_assert!(&lhs, rhs, le_scalar_u16_sse2_, &[!0x00fff800]);
    }

    #[test]
    fn test_equal_scalar_i32() {
        let len = 17;
        let mut lhs = AlignedVec::<i32>::new();
        remain_assign(&mut lhs, len, 6);

        let rhs = 1;

        cmp_assert!(&lhs, rhs, eq_scalar_i32_sse2_, &[0x2082]);
        cmp_assert!(&lhs, rhs, ne_scalar_i32_sse2_, &[!0x2082]);
    }

    #[test]
    fn test_order_scalar_i32() {
        let len = 14;
        let mut lhs = AlignedVec::<i32>::new();
        enumerate_assign(&mut lhs, len, -1);

        let rhs = 7;

        cmp_assert!(&lhs, rhs, gt_scalar_i32_sse2_, &[0xfe00]);
        cmp_assert!(&lhs, rhs, ge_scalar_i32_sse2_, &[0xff00]);
        cmp_assert!(&lhs, rhs, lt_scalar_i32_sse2_, &[!0xff00]);
        cmp_assert!(&lhs, rhs, le_scalar_i32_sse2_, &[!0xfe00]);
    }

    #[test]
    fn test_equal_scalar_u32() {
        let len = 17;
        let mut lhs = AlignedVec::<u32>::new();
        remain_assign(&mut lhs, len, 6);

        let rhs = 1;

        cmp_assert!(&lhs, rhs, eq_scalar_u32_sse2_, &[0x2082]);
        cmp_assert!(&lhs, rhs, ne_scalar_u32_sse2_, &[!0x2082]);
    }

    #[test]
    fn test_order_scalar_u32() {
        let len = 14;
        let mut lhs = AlignedVec::<u32>::new();
        enumerate_assign(&mut lhs, len, 2147483643);

        let rhs = 2147483654;

        cmp_assert!(&lhs, rhs, gt_scalar_u32_sse2_, &[0x3000]);
        cmp_assert!(&lhs, rhs, ge_scalar_u32_sse2_, &[0x3800]);
        cmp_assert!(&lhs, rhs, lt_scalar_u32_sse2_, &[!0x3800]);
        cmp_assert!(&lhs, rhs, le_scalar_u32_sse2_, &[!0x3000]);
    }

    #[test]
    fn test_equal_scalar_f32() {
        let lhs = AlignedVec::<f32>::from_slice(&[
            1.0, 7.4, -1.3, -8.9, -10.1, -255.6, 877.44, -1.3, -8.9, 344.7, 99.99, 77.77, -1.3,
        ]);

        let rhs = -1.3;

        cmp_assert!(&lhs, rhs, eq_scalar_f32_sse2_, &[0x1084]);
        cmp_assert!(&lhs, rhs, ne_scalar_f32_sse2_, &[!0x1084]);
    }

    #[test]
    fn test_order_scalar_f32() {
        let lhs = AlignedVec::<f32>::from_slice(&[
            1.0, 7.4, -1.3, -8.9, -10.1, -255.6, 877.44, -1.3, -8.9, 344.7, 99.99, 77.77, -1.3,
        ]);

        let rhs = -1.3;

        cmp_assert!(&lhs, rhs, gt_scalar_f32_sse2_, &[0x0e43]);
        cmp_assert!(&lhs, rhs, ge_scalar_f32_sse2_, &[0x1ec7]);
        cmp_assert!(&lhs, rhs, lt_scalar_f32_sse2_, &[!0x1ec7]);
        cmp_assert!(&lhs, rhs, le_scalar_f32_sse2_, &[!0x0e43]);
    }

    #[test]
    fn test_equal_scalar_f64() {
        let lhs = AlignedVec::<f64>::from_slice(&[
            1.0, 7.4, -1.3, -8.9, -10.1, -255.6, 877.44, -1.3, -8.9, 344.7, 99.99, 77.77, -1.3,
        ]);

        let rhs = -1.3;
        cmp_assert!(&lhs, rhs, eq_scalar_f64_sse2_, &[0x1084]);
        cmp_assert!(&lhs, rhs, ne_scalar_f64_sse2_, &[!0x1084]);
    }

    #[test]
    fn test_order_scalar_f64() {
        let lhs = AlignedVec::<f64>::from_slice(&[
            1.0, 7.4, -1.3, -8.9, -10.1, -255.6, 877.44, -1.3, -8.9, 344.7, 99.99, 77.77, -1.3,
        ]);

        let rhs = -1.3;

        cmp_assert!(&lhs, rhs, gt_scalar_f64_sse2_, &[0x0e43]);
        cmp_assert!(&lhs, rhs, ge_scalar_f64_sse2_, &[0x1ec7]);
        cmp_assert!(&lhs, rhs, lt_scalar_f64_sse2_, &[!0x1ec7]);
        cmp_assert!(&lhs, rhs, le_scalar_f64_sse2_, &[!0x0e43]);
    }
}
