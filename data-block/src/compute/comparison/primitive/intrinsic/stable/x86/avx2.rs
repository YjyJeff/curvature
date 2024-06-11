//! SIMD based on `avx2` instructions

use crate::aligned_vec::AlignedVec;
use crate::bitmap::BitStore;
use crate::utils::roundup_loops;
use std::mem::transmute;

x86_target_use!(
    __m256i,
    __m256,
    __m256d,
    _mm256_set1_epi8,
    _mm256_load_si256,
    _mm256_movemask_epi8,
    _mm256_xor_si256,
    _mm256_cmpeq_epi8,
    _mm256_cmpgt_epi8,
    _mm256_set1_epi16,
    _mm256_packs_epi16,
    _mm256_permute4x64_epi64,
    _mm256_cmpeq_epi16,
    _mm256_cmpgt_epi16,
    _mm256_set1_epi32,
    _mm256_packs_epi32,
    _mm256_cmpeq_epi32,
    _mm256_cmpgt_epi32,
    _mm256_permutevar8x32_epi32,
    _mm256_load_ps,
    _mm256_set1_ps,
    _mm256_castps_si256,
    _mm256_cmp_ps,
    _CMP_EQ_OS,
    _CMP_NEQ_OS,
    _CMP_GT_OS,
    _CMP_GE_OS,
    _CMP_LT_OS,
    _CMP_LE_OS,
    _mm256_set1_pd,
    _mm256_load_pd,
    _mm256_cmp_pd,
    _mm256_movemask_ps,
    _mm256_castpd_si256,
    _mm256_castsi256_ps,
    _mm256_extracti128_si256,
    _mm_packs_epi16,
    _mm_shuffle_epi32,
    _mm_movemask_epi8,
    _mm256_set1_epi64x,
    _mm256_blend_epi32,
    _mm256_cmpeq_epi64,
    _mm256_cmpgt_epi64,
    _mm256_blend_ps,
    _mm256_castpd_ps,
    _mm256_shuffle_epi8
);

/// Compare integer scalar with avx2
macro_rules! cmp_int_scalar_avx2 {
    // Signed integer match this pattern
    ($int_ty:ty, $suffix:ident) => {
        cmp_int_scalar!("avx2", _mm256, $int_ty, $suffix);
    };
    // Unsigned integer match this pattern
    ($uint_ty:ty, $int_ty:ty, $suffix:ident) => {
        cmp_int_scalar!("avx2", _mm256, $uint_ty, $int_ty, $suffix);
    };
}

/// Compare float scalar with avx2
macro_rules! cmp_float_scalar_avx2 {
    ($ty:ty, $suffix:ident) => {
        cmp_float_scalar!("avx2", _mm256, $ty, $suffix);
    };
}

/// Compare array of i8 with the given i8
///
/// # Safety
///
/// - Caller should ensure `avx2` is supported
/// - dst should have adequate space
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn cmp_scalar_i8_avx2<const NOT: bool, const FLIP_SIGN: bool, F>(
    lhs: &AlignedVec<i8>,
    rhs: i8,
    dst: *mut u32,
    cmp_func: F,
) where
    F: Fn(__m256i, __m256i) -> __m256i,
{
    const SHIFT_BITS: usize = 5;
    const BASE: usize = 1 << SHIFT_BITS;

    if lhs.len == 0 {
        return;
    }

    let num_loops = roundup_loops(lhs.len(), BASE);
    let mut rhs = _mm256_set1_epi8(rhs);

    let flip_sign = _mm256_set1_epi8(i8::MIN);
    if FLIP_SIGN {
        rhs = _mm256_xor_si256(rhs, flip_sign);
    }

    for simd_index in 0..num_loops {
        // AlignedVec's capacity is always multiple of 512bit/64byte, using tumbling window
        // with window size 256bit/32byte is always valid
        let mut lhs = _mm256_load_si256(lhs.ptr.as_ptr().add(simd_index << SHIFT_BITS) as _);
        if FLIP_SIGN {
            lhs = _mm256_xor_si256(lhs, flip_sign);
        }
        let cmp = cmp_func(lhs, rhs);
        // Convert to bitmap of i32
        let mut mask = _mm256_movemask_epi8(cmp);
        if NOT {
            mask ^= -1;
        }
        *dst.add(simd_index) = mask as u32;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn _mm256_cmplt_epi8(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmpgt_epi8(b, a)
}

/// Compare array of i32 with the given i32
///
/// # Safety
///
/// - Caller should ensure `avx2` is supported
/// - dst should have adequate space
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn cmp_scalar_i16_avx2<const NOT: bool, const FLIP_SIGN: bool, F>(
    lhs: &AlignedVec<i16>,
    rhs: i16,
    dst: *mut u32,
    cmp_func: F,
) where
    F: Fn(__m256i, __m256i) -> __m256i,
{
    const SHIFT_BITS: usize = 5;
    const BASE: usize = 1 << SHIFT_BITS;
    const CTRL_BIT: i32 = 0xd8;

    if lhs.len == 0 {
        return;
    }

    let num_loops = roundup_loops(lhs.len(), BASE);

    let mut rhs = _mm256_set1_epi16(rhs);

    // These two will be optimized away based on the NOT and FLIP_SIGN
    let flip_sign = _mm256_set1_epi16(i16::MIN);
    if FLIP_SIGN {
        rhs = _mm256_xor_si256(rhs, flip_sign);
    }

    for simd_index in 0..num_loops {
        // AlignedVec's capacity is always multiple of 512bit/64byte, using tumbling window
        // with window size 512bit/64byte is always valid
        let mut lhs_0 = _mm256_load_si256(lhs.ptr.as_ptr().add(simd_index << SHIFT_BITS) as _);
        let mut lhs_1 =
            _mm256_load_si256(lhs.ptr.as_ptr().add((simd_index << SHIFT_BITS) + 16) as _);

        if FLIP_SIGN {
            lhs_0 = _mm256_xor_si256(lhs_0, flip_sign);
            lhs_1 = _mm256_xor_si256(lhs_1, flip_sign);
        }

        // 00 11 22 33 44 55 66 77 88 99 aa bb cc dd ee ff
        let cmp_0 = cmp_func(lhs_0, rhs);
        // gg hh ii jj kk ll mm nn oo pp qq rr ss tt uu vv
        let cmp_1 = cmp_func(lhs_1, rhs);

        // Pack the 32 compare results into following bytes:
        // 0 1 2 3 4 5 6 7 g h i j k l m n 8 9 a b c d e f o p q r s t u v
        let mut packed = _mm256_packs_epi16(cmp_0, cmp_1);
        // Permute the 32 compare results into following bytes:
        // 0 1 2 3 4 5 6 7 8 9 a b c d e f g h i j k l m n o p q r s t u v
        packed = _mm256_permute4x64_epi64::<CTRL_BIT>(packed);

        // Convert to bitmap of i32
        let mut mask = _mm256_movemask_epi8(packed);

        if NOT {
            mask ^= -1;
        }
        *dst.add(simd_index) = mask as u32;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn _mm256_cmplt_epi16(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmpgt_epi16(b, a)
}

/// 64 byte aligned, such that we can use aligned load
#[repr(align(64))]
struct CacheLineAlignArray([i32; 8]);
/// Permute the __mm256i such that packed has correct order
const PERMUTE: CacheLineAlignArray =
    CacheLineAlignArray([0x00, 0x04, 0x01, 0x05, 0x02, 0x06, 0x03, 0x07]);

/// Compare array of i32 with the given i32
///
/// # Safety
///
/// - Caller should ensure `avx2` is supported
/// - dst should have adequate space
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn cmp_scalar_i32_avx2<const NOT: bool, const FLIP_SIGN: bool, F>(
    lhs: &AlignedVec<i32>,
    rhs: i32,
    dst: *mut u32,
    cmp_func: F,
) where
    F: Fn(__m256i, __m256i) -> __m256i,
{
    const SHIFT_BITS: usize = 4;
    const BASE: usize = 1 << SHIFT_BITS;
    const CTRL_BIT: i32 = 0xd8;

    if lhs.len == 0 {
        return;
    }

    let num_loops = roundup_loops(lhs.len(), BASE);

    let mut rhs = _mm256_set1_epi32(rhs);
    let lhs_ptr = lhs.ptr.as_ptr();

    // These two will be optimized away based on the NOT and FLIP_SIGN
    let flip_sign = _mm256_set1_epi32(i32::MIN);
    if FLIP_SIGN {
        rhs = _mm256_xor_si256(rhs, flip_sign);
    }

    let permute = _mm256_load_si256(PERMUTE.0.as_ptr() as _);

    let load_and_compare = |base: usize| {
        // AlignedVec's capacity is always multiple of 512bit/64byte, using tumbling window
        // with window size 512bit/64byte is always valid
        let mut lhs_0 = _mm256_load_si256(lhs_ptr.add(base) as _);
        let mut lhs_1 = _mm256_load_si256(lhs_ptr.add(base + 8) as _);
        if FLIP_SIGN {
            lhs_0 = _mm256_xor_si256(lhs_0, flip_sign);
            lhs_1 = _mm256_xor_si256(lhs_1, flip_sign);
        }
        // 0000 1111 2222 3333 4444 5555 6666 7777
        let cmp_0 = cmp_func(lhs_0, rhs);
        // 8888 9999 aaaa bbbb cccc dddd eeee ffff
        let cmp_1 = cmp_func(lhs_1, rhs);
        // Packs the 16 compare results into following bytes:
        // 00 11 22 33 88 99 aa bb 44 55 66 77 cc dd ee ff
        _mm256_packs_epi32(cmp_0, cmp_1)
    };

    let packs_to_mask = |packed_0: __m256i, packed_1: __m256i| {
        let packed = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(packed_0, packed_1), permute);
        let mask = _mm256_movemask_epi8(packed);
        if NOT {
            (mask ^ -1) as u32
        } else {
            mask as u32
        }
    };

    // packed: 00 11 22 33 88 99 aa bb 44 55 66 77 cc dd ee ff
    let pack_to_mask = |packed: __m256i| {
        // Compiler will only extract hi, lo is the xmm register
        // hi: 44 55 66 77 cc dd ee ff
        let hi = _mm256_extracti128_si256(packed, 1);
        // lo: 00 11 22 33 88 99 aa bb
        let lo = _mm256_extracti128_si256(packed, 0);
        // packed: 0 1 2 3 8 9 a b 4 5 6 7 c d e f
        let mut packed = _mm_packs_epi16(lo, hi);
        // shuffle the packed to following bytes:
        // 0 1 2 3 4 5 6 7 8 9 a b c d e f
        packed = _mm_shuffle_epi32::<CTRL_BIT>(packed);

        let mask = _mm_movemask_epi8(packed);
        if NOT {
            (mask ^ -1) as u32
        } else {
            mask as u32
        }
    };

    if num_loops > 1 {
        for simd_index in (0..num_loops - 1).step_by(2) {
            let packed_0 = load_and_compare(simd_index << SHIFT_BITS);
            // There exist
            let packed_1 = load_and_compare((simd_index + 1) << SHIFT_BITS);
            *dst.add(simd_index >> 1) = packs_to_mask(packed_0, packed_1);
        }
    }
    if num_loops & 0x01 != 0 {
        // Handle last
        let simd_index = num_loops - 1;
        let packed = load_and_compare(simd_index << SHIFT_BITS);
        *dst.add(simd_index >> 1) = pack_to_mask(packed);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn _mm256_cmplt_epi32(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmpgt_epi32(b, a)
}

/// Compare array of f32 with the given f32
///
/// # Safety
///
/// - Caller should ensure `avx2` is supported
/// - dst should have adequate space
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn cmp_scalar_f32_avx2<F>(lhs: &AlignedVec<f32>, rhs: f32, dst: *mut u32, cmp_func: F)
where
    F: Fn(__m256, __m256) -> __m256,
{
    const SHIFT_BITS: usize = 4;
    const BASE: usize = 1 << SHIFT_BITS;
    const CTRL_BIT: i32 = 0xd8;

    if lhs.len == 0 {
        return;
    }

    let num_loops = roundup_loops(lhs.len(), BASE);
    let lhs_ptr = lhs.ptr.as_ptr();

    let rhs = _mm256_set1_ps(rhs);
    let permute = _mm256_load_si256(PERMUTE.0.as_ptr() as _);

    let load_and_compare = |base: usize| {
        let lhs_0 = _mm256_load_ps(lhs_ptr.add(base) as _);
        let lhs_1 = _mm256_load_ps(lhs_ptr.add(base + 8) as _);
        let cmp_0 = cmp_func(lhs_0, rhs);
        let cmp_1 = cmp_func(lhs_1, rhs);
        _mm256_packs_epi32(_mm256_castps_si256(cmp_0), _mm256_castps_si256(cmp_1))
    };

    if num_loops > 1 {
        for simd_index in (0..num_loops - 1).step_by(2) {
            let packed_0 = load_and_compare(simd_index << SHIFT_BITS);
            let packed_1 = load_and_compare((simd_index + 1) << SHIFT_BITS);
            let packed =
                _mm256_permutevar8x32_epi32(_mm256_packs_epi16(packed_0, packed_1), permute);
            let mask = _mm256_movemask_epi8(packed) as u32;
            *dst.add(simd_index >> 1) = mask;
        }
    }

    if num_loops & 0x01 != 0 {
        // Handle last
        let simd_index = num_loops - 1;
        let packed = load_and_compare(simd_index << SHIFT_BITS);
        // Compiler will only extract hi, lo is the xmm register
        // hi: 44 55 66 77 cc dd ee ff
        let hi = _mm256_extracti128_si256(packed, 1);
        // lo: 00 11 22 33 88 99 aa bb
        let lo = _mm256_extracti128_si256(packed, 0);
        // packed: 0 1 2 3 8 9 a b 4 5 6 7 c d e f
        let mut packed = _mm_packs_epi16(lo, hi);
        // shuffle the packed to following bytes:
        // 0 1 2 3 4 5 6 7 8 9 a b c d e f
        packed = _mm_shuffle_epi32::<CTRL_BIT>(packed);
        let mask = _mm_movemask_epi8(packed) as u32;

        *dst.add(simd_index >> 1) = mask;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn _mm256_cmpeq_ps(a: __m256, b: __m256) -> __m256 {
    _mm256_cmp_ps::<_CMP_EQ_OS>(a, b)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn _mm256_cmpneq_ps(a: __m256, b: __m256) -> __m256 {
    _mm256_cmp_ps::<_CMP_NEQ_OS>(a, b)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn _mm256_cmpgt_ps(a: __m256, b: __m256) -> __m256 {
    _mm256_cmp_ps::<_CMP_GT_OS>(a, b)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn _mm256_cmpge_ps(a: __m256, b: __m256) -> __m256 {
    _mm256_cmp_ps::<_CMP_GE_OS>(a, b)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn _mm256_cmplt_ps(a: __m256, b: __m256) -> __m256 {
    _mm256_cmp_ps::<_CMP_LT_OS>(a, b)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn _mm256_cmple_ps(a: __m256, b: __m256) -> __m256 {
    _mm256_cmp_ps::<_CMP_LE_OS>(a, b)
}

/// Permute the __mm256i such that packed has correct order
const SHUFFLE_64: CacheLineAlignArray = CacheLineAlignArray([
    0x0a080200, 0x0b090301, 0x0e0c0604, 0x0f0d0705, 0x0a080200, 0x0b090301, 0x0e0c0604, 0x0f0d0705,
]);

/// Compare array of f64 with the given f64
///
/// # Safety
///
/// - Caller should ensure `avx2` is supported
/// - dst should have adequate space
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn cmp_scalar_f64_avx2<F>(lhs: &AlignedVec<f64>, rhs: f64, dst: *mut u32, cmp_func: F)
where
    F: Fn(__m256d, __m256d) -> __m256d + Copy,
{
    const SHIFT_BITS: usize = 3;
    const BASE: usize = 1 << SHIFT_BITS;

    if lhs.len == 0 {
        return;
    }

    let num_loops = roundup_loops(lhs.len, BASE);

    let rhs = _mm256_set1_pd(rhs);
    let mut lhs_ptr = lhs.ptr.as_ptr();

    // https://stackoverflow.com/questions/70974476/efficiently-load-compute-pack-64-double-comparison-results-in-uint64-t-bitmask

    let shuffle = _mm256_load_si256(SHUFFLE_64.0.as_ptr() as _);

    // Function instead of closure here. Closure can not be inlined 😂 Bug?
    #[inline(always)]
    unsafe fn load_and_compare_4<F>(lhs_ptr: *mut f64, rhs: __m256d, cmp_func: F) -> __m256d
    where
        F: Fn(__m256d, __m256d) -> __m256d + Copy,
    {
        cmp_func(_mm256_load_pd(lhs_ptr), rhs)
    }

    // Compare 8 numbers, combine 8 results into the following bytes:
    // 0000 4444 1111 5555  2222 6666 3333 7777
    #[inline(always)]
    unsafe fn load_and_compare_8<F>(lhs_ptr: *mut f64, rhs: __m256d, cmp_func: F) -> __m256i
    where
        F: Fn(__m256d, __m256d) -> __m256d + Copy,
    {
        let cmp_0 = load_and_compare_4(lhs_ptr, rhs, cmp_func);
        let cmp_1 = load_and_compare_4(lhs_ptr.add(4), rhs, cmp_func);
        _mm256_castps_si256(_mm256_blend_ps::<0xaa>(
            _mm256_castpd_ps(cmp_0),
            _mm256_castpd_ps(cmp_1),
        ))
    }

    // Compare 16 numbers, combine 16 results into following bytes:
    // 00 44 11 55  88 CC 99 DD  22 66 33 77  AA EE BB FF
    #[inline(always)]
    unsafe fn load_and_compare_16<F>(lhs_ptr: *mut f64, rhs: __m256d, cmp_func: F) -> __m256i
    where
        F: Fn(__m256d, __m256d) -> __m256d + Copy,
    {
        _mm256_packs_epi32(
            load_and_compare_8(lhs_ptr, rhs, cmp_func),
            load_and_compare_8(lhs_ptr.add(8), rhs, cmp_func),
        )
    }

    // Compare 16 numbers, combine 16 results into following bytes:
    // 00 44 11 55  88 CC 99 DD  22 66 33 77  AA EE BB FF
    #[inline(always)]
    unsafe fn load_and_compare_32<F>(
        lhs_ptr: *mut f64,
        rhs: __m256d,
        cmp_func: F,
        shuffle: __m256i,
    ) -> u32
    where
        F: Fn(__m256d, __m256d) -> __m256d + Copy,
    {
        // We got the 32 bytes, but the byte order is screwed up in that vector:
        // 0   4   1   5   8   12  9   13  16  20  17  21  24  28  25  29
        // 2   6   3   7   10  14  11  15  18  22  19  23  26  30  27  31
        // The following 2 instructions are fixing the order
        let mut scrwed = _mm256_packs_epi16(
            load_and_compare_16(lhs_ptr, rhs, cmp_func),
            load_and_compare_16(lhs_ptr.add(16), rhs, cmp_func),
        );
        // Shuffle 8-byte pieces across the complete vector
        // That instruction is relatively expensive on most CPUs, but we only doing it once per 32 numbers
        scrwed = _mm256_permute4x64_epi64::<0xd8>(scrwed);
        // The order of bytes in the vector is still wrong:
        // 0    4   1   5   8  12   9  13    2   6   3   7  10  14  11  15
        // 13  16  20  17  21  24  28  25   18  22  19  23  26  30  27  31
        // Need one more shuffle instruction
        let ordered = _mm256_shuffle_epi8(scrwed, shuffle);
        _mm256_movemask_epi8(ordered) as u32
    }

    if num_loops >= 4 {
        for simd_index in (0..num_loops - 3).step_by(4) {
            let mask = load_and_compare_32(lhs_ptr, rhs, cmp_func, shuffle);
            *dst.add(simd_index >> 2) = mask as _;
            lhs_ptr = lhs_ptr.add(32);
        }
    }
    // Handle remained
    if num_loops & 0x03 != 0 {
        let mut mask = 0;
        for simd_index in 0..num_loops {
            let lhs_0 = _mm256_load_pd(lhs_ptr);
            let lhs_1 = _mm256_load_pd(lhs_ptr.add(4));
            let cmp_0 = cmp_func(lhs_0, rhs);
            let cmp_1 = cmp_func(lhs_1, rhs);
            let mut packed =
                _mm256_packs_epi32(_mm256_castpd_si256(cmp_0), _mm256_castpd_si256(cmp_1));
            packed = _mm256_permute4x64_epi64::<0xd8>(packed);
            let mask_8 = _mm256_movemask_ps(_mm256_castsi256_ps(packed)) as u32;
            mask |= mask_8 << (simd_index << 3);
            lhs_ptr = lhs_ptr.add(8);
        }
        *dst.add(num_loops >> 2) = mask;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn _mm256_cmpeq_pd(a: __m256d, b: __m256d) -> __m256d {
    _mm256_cmp_pd::<_CMP_EQ_OS>(a, b)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn _mm256_cmpneq_pd(a: __m256d, b: __m256d) -> __m256d {
    _mm256_cmp_pd::<_CMP_NEQ_OS>(a, b)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn _mm256_cmpgt_pd(a: __m256d, b: __m256d) -> __m256d {
    _mm256_cmp_pd::<_CMP_GT_OS>(a, b)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn _mm256_cmpge_pd(a: __m256d, b: __m256d) -> __m256d {
    _mm256_cmp_pd::<_CMP_GE_OS>(a, b)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn _mm256_cmplt_pd(a: __m256d, b: __m256d) -> __m256d {
    _mm256_cmp_pd::<_CMP_LT_OS>(a, b)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn _mm256_cmple_pd(a: __m256d, b: __m256d) -> __m256d {
    _mm256_cmp_pd::<_CMP_LE_OS>(a, b)
}

/// Compare array of i64 with the given i64
///
/// # Safety
///
/// - Caller should ensure `avx2` is supported
/// - dst should have adequate space
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn cmp_scalar_i64_avx2<const NOT: bool, const FLIP_SIGN: bool, F>(
    lhs: &AlignedVec<i64>,
    rhs: i64,
    dst: *mut u32,
    cmp_func: F,
) where
    F: Fn(__m256i, __m256i) -> __m256i + Copy,
{
    const SHIFT_BITS: usize = 3;
    const BASE: usize = 1 << SHIFT_BITS;

    if lhs.len == 0 {
        return;
    }

    let num_loops = roundup_loops(lhs.len, BASE);
    let flip_sign = _mm256_set1_epi64x(i64::MIN);

    let mut rhs = _mm256_set1_epi64x(rhs);
    if FLIP_SIGN {
        rhs = _mm256_xor_si256(rhs, flip_sign);
    }
    let mut lhs_ptr = lhs.ptr.as_ptr();

    // https://stackoverflow.com/questions/70974476/efficiently-load-compute-pack-64-double-comparison-results-in-uint64-t-bitmask

    let shuffle = _mm256_load_si256(SHUFFLE_64.0.as_ptr() as _);

    // Function instead of closure here. Closure can not be inlined 😂 Bug?
    #[inline(always)]
    unsafe fn load_and_compare_4<const FLIP_SIGN: bool, F>(
        lhs_ptr: *mut i64,
        rhs: __m256i,
        cmp_func: F,
        flip_sign: __m256i,
    ) -> __m256i
    where
        F: Fn(__m256i, __m256i) -> __m256i + Copy,
    {
        let mut lhs = _mm256_load_si256(lhs_ptr as _);
        if FLIP_SIGN {
            lhs = _mm256_xor_si256(lhs, flip_sign);
        }
        cmp_func(lhs, rhs)
    }

    // Compare 8 numbers, combine 8 results into the following bytes:
    // 0000 4444 1111 5555  2222 6666 3333 7777
    #[inline(always)]
    unsafe fn load_and_compare_8<const FLIP_SIGN: bool, F>(
        lhs_ptr: *mut i64,
        rhs: __m256i,
        cmp_func: F,
        flip_sign: __m256i,
    ) -> __m256i
    where
        F: Fn(__m256i, __m256i) -> __m256i + Copy,
    {
        let cmp_0 = load_and_compare_4::<FLIP_SIGN, _>(lhs_ptr, rhs, cmp_func, flip_sign);
        let cmp_1 = load_and_compare_4::<FLIP_SIGN, _>(lhs_ptr.add(4), rhs, cmp_func, flip_sign);
        _mm256_blend_epi32::<0xaa>(cmp_0, cmp_1)
    }

    // Compare 16 numbers, combine 16 results into following bytes:
    // 00 44 11 55  88 CC 99 DD  22 66 33 77  AA EE BB FF
    #[inline(always)]
    unsafe fn load_and_compare_16<const FLIP_SIGN: bool, F>(
        lhs_ptr: *mut i64,
        rhs: __m256i,
        cmp_func: F,
        flip_sign: __m256i,
    ) -> __m256i
    where
        F: Fn(__m256i, __m256i) -> __m256i + Copy,
    {
        _mm256_packs_epi32(
            load_and_compare_8::<FLIP_SIGN, _>(lhs_ptr, rhs, cmp_func, flip_sign),
            load_and_compare_8::<FLIP_SIGN, _>(lhs_ptr.add(8), rhs, cmp_func, flip_sign),
        )
    }

    // Compare 16 numbers, combine 16 results into following bytes:
    // 00 44 11 55  88 CC 99 DD  22 66 33 77  AA EE BB FF
    #[inline(always)]
    unsafe fn load_and_compare_32<const FLIP_SIGN: bool, F>(
        lhs_ptr: *mut i64,
        rhs: __m256i,
        cmp_func: F,
        flip_sign: __m256i,
        shuffle: __m256i,
    ) -> i32
    where
        F: Fn(__m256i, __m256i) -> __m256i + Copy,
    {
        // We got the 32 bytes, but the byte order is screwed up in that vector:
        // 0   4   1   5   8   12  9   13  16  20  17  21  24  28  25  29
        // 2   6   3   7   10  14  11  15  18  22  19  23  26  30  27  31
        // The following 2 instructions are fixing the order
        let mut scrwed = _mm256_packs_epi16(
            load_and_compare_16::<FLIP_SIGN, _>(lhs_ptr, rhs, cmp_func, flip_sign),
            load_and_compare_16::<FLIP_SIGN, _>(lhs_ptr.add(16), rhs, cmp_func, flip_sign),
        );
        // Shuffle 8-byte pieces across the complete vector
        // That instruction is relatively expensive on most CPUs, but we only doing it once per 32 numbers
        scrwed = _mm256_permute4x64_epi64::<0xd8>(scrwed);
        // The order of bytes in the vector is still wrong:
        // 0    4   1   5   8  12   9  13    2   6   3   7  10  14  11  15
        // 13  16  20  17  21  24  28  25   18  22  19  23  26  30  27  31
        // Need one more shuffle instruction
        let ordered = _mm256_shuffle_epi8(scrwed, shuffle);
        _mm256_movemask_epi8(ordered)
    }

    if num_loops >= 4 {
        for simd_index in (0..num_loops - 3).step_by(4) {
            let mut mask =
                load_and_compare_32::<FLIP_SIGN, _>(lhs_ptr, rhs, cmp_func, flip_sign, shuffle);
            if NOT {
                mask ^= -1;
            }
            *dst.add(simd_index >> 2) = mask as _;
            lhs_ptr = lhs_ptr.add(32);
        }
    }
    // Handle remained
    if num_loops & 0x03 != 0 {
        let mut mask = 0;
        for simd_index in 0..num_loops {
            let mut lhs_0 = _mm256_load_si256(lhs_ptr as _);
            let mut lhs_1 = _mm256_load_si256(lhs_ptr.add(4) as _);
            if FLIP_SIGN {
                lhs_0 = _mm256_xor_si256(lhs_0, flip_sign);
                lhs_1 = _mm256_xor_si256(lhs_1, flip_sign);
            }
            let cmp_0 = cmp_func(lhs_0, rhs);
            let cmp_1 = cmp_func(lhs_1, rhs);
            let mut packed = _mm256_packs_epi32(cmp_0, cmp_1);
            packed = _mm256_permute4x64_epi64::<0xd8>(packed);
            let mask_8 = _mm256_movemask_ps(_mm256_castsi256_ps(packed));
            mask |= mask_8 << (simd_index << 3);
            lhs_ptr = lhs_ptr.add(8);
        }
        if NOT {
            mask ^= -1;
        }
        *dst.add(num_loops >> 2) = mask as _;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn _mm256_cmplt_epi64(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmpgt_epi64(b, a)
}

cmp_int_scalar_avx2!(i8, epi8);
cmp_int_scalar_avx2!(u8, i8, epi8);
cmp_int_scalar_avx2!(i16, epi16);
cmp_int_scalar_avx2!(u16, i16, epi16);
cmp_int_scalar_avx2!(i32, epi32);
cmp_int_scalar_avx2!(u32, i32, epi32);
cmp_int_scalar_avx2!(i64, epi64);
cmp_int_scalar_avx2!(u64, i64, epi64);

cmp_float_scalar_avx2!(f32, ps);
cmp_float_scalar_avx2!(f64, pd);

#[cfg(target_feature = "avx2")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::Bitmap;

    #[test]
    fn test_equal_scalar_i8() {
        let lhs = AlignedVec::<i8>::from_slice(&[
            -1, -3, -7, 9, 100, 44, 32, -1, -7, -122, 8, 24, -33, 21, -3, -1, 99, 88, -126, 127,
            -1, -5, 33, 44, 22, -56, -76, -99, 11, -84, 0, 0, -43, -54, 32,
        ]);

        let rhs = -1;
        cmp_assert!(&lhs, rhs, eq_scalar_i8_avx2_, &[0x0000108081]);
        cmp_assert!(&lhs, rhs, ne_scalar_i8_avx2_, &[!0x0000108081]);
    }

    #[test]
    fn test_order_scalar_i8() {
        let lhs = AlignedVec::<i8>::from_slice(&[
            -1, -3, -7, 9, 100, 44, 32, -1, -7, -122, 8, 24, -33, 21, -3, -1, 99, 88, -126, 127,
            -1, -5, 33, 44, 22, -56, -76, -99, 11, -84, 0, 0, -43, -54, 32,
        ]);

        let rhs = -1;
        cmp_assert!(&lhs, rhs, gt_scalar_i8_avx2_, &[0x04d1cb2c78]);
        cmp_assert!(&lhs, rhs, ge_scalar_i8_avx2_, &[0x04d1dbacf9]);
        cmp_assert!(&lhs, rhs, lt_scalar_i8_avx2_, &[!0x04d1dbacf9]);
        cmp_assert!(&lhs, rhs, le_scalar_i8_avx2_, &[!0x04d1cb2c78]);
    }

    #[test]
    fn test_equal_scalar_u8() {
        let lhs = AlignedVec::<u8>::from_slice(&[
            7, 9, 233, 11, 200, 120, 244, 233, 1, 4, 9, 255, 127, 11, 39, 42,
        ]);

        let rhs = 233;
        cmp_assert!(&lhs, rhs, eq_scalar_u8_avx2_, &[0x0084]);
        cmp_assert!(&lhs, rhs, ne_scalar_u8_avx2_, &[!0x0084]);
    }

    #[test]
    fn test_order_scalar_u8() {
        let lhs = AlignedVec::<u8>::from_slice(&[
            7, 9, 233, 11, 200, 120, 244, 233, 1, 4, 9, 255, 127, 11, 39, 42,
        ]);

        let rhs = 200;
        cmp_assert!(&lhs, rhs, gt_scalar_u8_avx2_, &[0x08c4]);
        cmp_assert!(&lhs, rhs, ge_scalar_u8_avx2_, &[0x08d4]);
        cmp_assert!(&lhs, rhs, lt_scalar_u8_avx2_, &[!0x08d4]);
        cmp_assert!(&lhs, rhs, le_scalar_u8_avx2_, &[!0x08c4]);
    }

    #[test]
    fn test_equal_scalar_i16() {
        let lhs = AlignedVec::<i16>::from_slice(&[
            -32768, 32767, -444, -333, -678, 631, 23, -7, -4, 444, 777, 32666, 8999, -4, -4, -321,
            -33, 990, 453, 4232, 6763, -3223, 122, -1, -44, -98, -32, -11, 23, 465, -44, -4, 122,
            3344, 890, -122, 78, 89, -4,
        ]);

        let rhs = -4;
        cmp_assert!(&lhs, rhs, eq_scalar_i16_avx2_, &[0x4080006100]);
        cmp_assert!(&lhs, rhs, ne_scalar_i16_avx2_, &[!0x4080006100]);
    }

    #[test]
    fn test_order_scalar_i16() {
        let lhs = AlignedVec::<i16>::from_slice(&[
            -32768, 32767, -444, -333, -678, 631, 23, -7, -4, 444, 777, 32666, 8999, -4, -4, -321,
            -33, 990, 453, 4232, 6763, -3223, 122, -1, -44, -98, -32, -11, 23, 465, -44, -4, 122,
            3344, 890, -122, 78, 89, -4,
        ]);

        let rhs = -4;

        cmp_assert!(&lhs, rhs, gt_scalar_i16_avx2_, &[0x3730de1e62]);
        cmp_assert!(&lhs, rhs, ge_scalar_i16_avx2_, &[0x77b0de7f62]);
        cmp_assert!(&lhs, rhs, lt_scalar_i16_avx2_, &[!0x77b0de7f62]);
        cmp_assert!(&lhs, rhs, le_scalar_i16_avx2_, &[!0x3730de1e62]);
    }

    #[test]
    fn test_equal_scalar_u16() {
        let lhs = AlignedVec::<u16>::from_slice(&[
            3375, 32766, 19999, 11, 1, 9783, 33, 19999, 21, 34, 11,
        ]);

        let rhs = 19999;
        cmp_assert!(&lhs, rhs, eq_scalar_u16_avx2_, &[0x084]);
        cmp_assert!(&lhs, rhs, ne_scalar_u16_avx2_, &[!0x084]);
    }

    #[test]
    fn test_order_scalar_u16() {
        let lhs = AlignedVec::<u16>::from_slice(&[
            3375, 32766, 19999, 11, 1, 9783, 33, 19999, 21, 34, 11,
        ]);

        let rhs = 19999;

        cmp_assert!(&lhs, rhs, gt_scalar_u16_avx2_, &[0x002]);
        cmp_assert!(&lhs, rhs, ge_scalar_u16_avx2_, &[0x086]);
        cmp_assert!(&lhs, rhs, lt_scalar_u16_avx2_, &[!0x086]);
        cmp_assert!(&lhs, rhs, le_scalar_u16_avx2_, &[!0x002]);
    }

    #[test]
    fn test_equal_scalar_i32() {
        let lhs = AlignedVec::<i32>::from_slice(&[
            -256, -255, -127, 134, 77, -1, -3, 89, -255, -128, 33, 90, 73, -34, -83, -354, 673,
            345, -566,
        ]);

        let rhs = -255;
        cmp_assert!(&lhs, rhs, eq_scalar_i32_avx2_, &[0x000102]);
        cmp_assert!(&lhs, rhs, ne_scalar_i32_avx2_, &[!0x000102]);
    }

    #[test]
    fn test_order_scalar_i32() {
        let lhs = AlignedVec::<i32>::from_slice(&[
            -256, -255, -127, 134, 77, -1, -3, 89, -255, -128, 33, 90, 73, -34, -83, -354, 673,
            345, -566,
        ]);

        let rhs = -255;

        cmp_assert!(&lhs, rhs, gt_scalar_i32_avx2_, &[0x037efc]);
        cmp_assert!(&lhs, rhs, ge_scalar_i32_avx2_, &[0x037ffe]);
        cmp_assert!(&lhs, rhs, lt_scalar_i32_avx2_, &[!0x037ffe]);
        cmp_assert!(&lhs, rhs, le_scalar_i32_avx2_, &[!0x037efc]);
    }

    #[test]
    fn test_equal_scalar_u32() {
        let lhs = AlignedVec::<u32>::from_slice(&[
            2147483648, 2147483648, 2147483641, 0, 0, 32, 1, 2147483650, 2247483650, 2147493650, 0,
            12, 673, 3147483650, 23242, 1, 234,
        ]);

        let rhs = 2147483648;
        cmp_assert!(&lhs, rhs, eq_scalar_u32_avx2_, &[0x000003]);
        cmp_assert!(&lhs, rhs, ne_scalar_u32_avx2_, &[!0x000003]);
    }

    #[test]
    fn test_order_scalar_u32() {
        let lhs = AlignedVec::<u32>::from_slice(&[
            2147483648, 2147483648, 2147483641, 0, 0, 32, 1, 2147483650, 2247483650, 2147493650, 0,
            12, 673, 3147483650, 23242, 1, 234,
        ]);

        let rhs = 2147483648;

        cmp_assert!(&lhs, rhs, gt_scalar_u32_avx2_, &[0x002380]);
        cmp_assert!(&lhs, rhs, ge_scalar_u32_avx2_, &[0x002383]);
        cmp_assert!(&lhs, rhs, lt_scalar_u32_avx2_, &[!0x002383]);
        cmp_assert!(&lhs, rhs, le_scalar_u32_avx2_, &[!0x002380]);
    }

    #[test]
    fn test_equal_scalar_f32() {
        let lhs = AlignedVec::<f32>::from_slice(&[
            1.0, 7.4, -1.3, -8.9, -10.1, -255.6, 877.44, -1.3, -8.9, 344.7, 99.99, 77.77, -1.3,
        ]);

        let rhs = -1.3;

        cmp_assert!(&lhs, rhs, eq_scalar_f32_avx2_, &[0x1084]);
        cmp_assert!(&lhs, rhs, ne_scalar_f32_avx2_, &[!0x1084]);
    }

    #[test]
    fn test_order_scalar_f32() {
        let lhs = AlignedVec::<f32>::from_slice(&[
            1.0, 7.4, -1.3, -8.9, -10.1, -255.6, 877.44, -1.3, -8.9, 344.7, 99.99, 77.77, -1.3,
        ]);

        let rhs = -1.3;

        cmp_assert!(&lhs, rhs, gt_scalar_f32_avx2_, &[0x0e43]);
        cmp_assert!(&lhs, rhs, ge_scalar_f32_avx2_, &[0x1ec7]);
        cmp_assert!(&lhs, rhs, lt_scalar_f32_avx2_, &[!0x1ec7]);
        cmp_assert!(&lhs, rhs, le_scalar_f32_avx2_, &[!0x0e43]);
    }

    #[test]
    fn test_equal_scalar_f64() {
        let lhs = AlignedVec::<f64>::from_slice(&[
            1.0, 7.4, -1.3, -8.9, -10.1, -255.6, 877.44, -1.3, -8.9, 344.7, 99.99, 77.77, -1.3,
        ]);

        let rhs = -1.3;

        cmp_assert!(&lhs, rhs, eq_scalar_f64_avx2_, &[0x1084]);
        cmp_assert!(&lhs, rhs, ne_scalar_f64_avx2_, &[!0x1084]);

        let lhs = AlignedVec::<f64>::from_slice(&[
            1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0,
            -1.0,
        ]);
        let rhs = 1.0;
        cmp_assert!(&lhs, rhs, eq_scalar_f64_avx2_, &[0x77918039]);
        cmp_assert!(&lhs, rhs, ne_scalar_f64_avx2_, &[!0x77918039]);
    }

    #[test]
    fn test_order_scalar_f64() {
        let lhs = AlignedVec::<f64>::from_slice(&[
            1.0, 7.4, -1.3, -8.9, -10.1, -255.6, 877.44, -1.3, -8.9, 344.7, 99.99, 77.77, -1.3,
        ]);

        let rhs = -1.3;

        cmp_assert!(&lhs, rhs, gt_scalar_f64_avx2_, &[0x0e43]);
        cmp_assert!(&lhs, rhs, ge_scalar_f64_avx2_, &[0x1ec7]);
        cmp_assert!(&lhs, rhs, lt_scalar_f64_avx2_, &[!0x1ec7]);
        cmp_assert!(&lhs, rhs, le_scalar_f64_avx2_, &[!0x0e43]);

        let lhs = AlignedVec::<f64>::from_slice(&[
            1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0,
            0.0,
        ]);

        let rhs = 0.0;
        cmp_assert!(&lhs, rhs, gt_scalar_f64_avx2_, &[0x77918039]);
        cmp_assert!(&lhs, rhs, ge_scalar_f64_avx2_, &[0xf7918039]);
        cmp_assert!(&lhs, rhs, lt_scalar_f64_avx2_, &[!0xf7918039]);
        cmp_assert!(&lhs, rhs, le_scalar_f64_avx2_, &[!0x77918039]);
    }

    #[test]
    fn test_equal_scalar_i64() {
        let lhs = AlignedVec::<i64>::from_slice(&[
            7, 9, 100, 234, 45, 100, -10, -99, -123, -56, 100, 345, -34554,
        ]);

        let rhs = 100;
        cmp_assert!(&lhs, rhs, eq_scalar_i64_avx2_, &[0x0424]);
        cmp_assert!(&lhs, rhs, ne_scalar_i64_avx2_, &[!0x0424]);

        let lhs = AlignedVec::<i64>::from_slice(&[
            1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1,
            1, 1, 1, -1, 1, 1, 1, -1,
        ]);

        let rhs = 1;
        cmp_assert!(&lhs, rhs, eq_scalar_i64_avx2_, &[0x77918039]);
        cmp_assert!(&lhs, rhs, ne_scalar_i64_avx2_, &[!0x77918039]);
    }

    #[test]
    fn test_order_scalar_i64() {
        let lhs = AlignedVec::<i64>::from_slice(&[
            7, 9, 100, 234, 45, 100, -10, -99, -123, -56, 100, 345, -34554,
        ]);

        let rhs = 100;

        cmp_assert!(&lhs, rhs, gt_scalar_i64_avx2_, &[0x0808]);
        cmp_assert!(&lhs, rhs, ge_scalar_i64_avx2_, &[0x0c2c]);
        cmp_assert!(&lhs, rhs, lt_scalar_i64_avx2_, &[!0x0c2c]);
        cmp_assert!(&lhs, rhs, le_scalar_i64_avx2_, &[!0x0808]);

        let lhs = AlignedVec::<i64>::from_slice(&[
            1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1,
            1, 1, 1, -1, 1, 1, 1, 0,
        ]);

        let rhs = 0;
        cmp_assert!(&lhs, rhs, gt_scalar_i64_avx2_, &[0x77918039]);
        cmp_assert!(&lhs, rhs, ge_scalar_i64_avx2_, &[0xf7918039]);
        cmp_assert!(&lhs, rhs, lt_scalar_i64_avx2_, &[!0xf7918039]);
        cmp_assert!(&lhs, rhs, le_scalar_i64_avx2_, &[!0x77918039]);
    }

    #[test]
    fn test_equal_scalar_u64() {
        let lhs = AlignedVec::<u64>::from_slice(&[
            2147483648, 2147483638, 2247483648, 2347483648, 2147483648,
        ]);

        let rhs = 2147483648;
        cmp_assert!(&lhs, rhs, eq_scalar_u64_avx2_, &[0x11]);
        cmp_assert!(&lhs, rhs, ne_scalar_u64_avx2_, &[!0x11]);
    }

    #[test]
    fn test_order_scalar_u64() {
        let lhs = AlignedVec::<u64>::from_slice(&[
            2147483648, 2147483638, 2247483648, 2347483648, 2147483648,
        ]);

        let rhs = 2147483648;

        cmp_assert!(&lhs, rhs, gt_scalar_u64_avx2_, &[0x0c]);
        cmp_assert!(&lhs, rhs, ge_scalar_u64_avx2_, &[0x1d]);
        cmp_assert!(&lhs, rhs, lt_scalar_u64_avx2_, &[!0x1d]);
        cmp_assert!(&lhs, rhs, le_scalar_u64_avx2_, &[!0x0c]);
    }
}
