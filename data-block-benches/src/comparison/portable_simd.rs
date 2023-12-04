// Looks like it outperforms manually SIMD

use std::simd::{
    f32x16, f64x8, i16x32, i32x16, i64x8, i8x64, Mask, SimdElement, SimdPartialOrd, ToBitMask,
};

pub trait SimdNative {
    type SimdXLanes: SimdXLanes<Native = Self>;
}

pub trait SimdXLanes: SimdPartialOrd<Mask = Self::M> + Copy {
    const LANES: usize;
    type Native: SimdNative<SimdXLanes = Self>;
    type BM: Default + Clone;
    type M: ToBitMask<BitMask = Self::BM>;

    fn splat(v: Self::Native) -> Self;

    fn from_slice(v: &[Self::Native]) -> Self;
}
macro_rules! impl_simd {
    ($ty:ty, $md:ty, $bm:ty, $lanes:expr) => {
        impl SimdNative for $ty {
            type SimdXLanes = $md;
        }

        impl SimdXLanes for $md {
            const LANES: usize = $lanes;
            type Native = $ty;
            type BM = $bm;
            type M = Mask<<$ty as SimdElement>::Mask, $lanes>;

            #[inline]
            fn splat(v: $ty) -> Self {
                <$md>::splat(v)
            }

            #[inline]
            fn from_slice(v: &[$ty]) -> Self {
                <$md>::from_slice(v)
            }
        }
    };
}

impl_simd!(f64, f64x8, u8, 8);
impl_simd!(f32, f32x16, u16, 16);
impl_simd!(i64, i64x8, u8, 8);
impl_simd!(i32, i32x16, u16, 16);
impl_simd!(i16, i16x32, u32, 32);
impl_simd!(i8, i8x64, u64, 64);

#[inline]
fn cmp_scalar<T, F>(
    lhs: &[T],
    rhs: T,
    dst: &mut [<<T as SimdNative>::SimdXLanes as SimdXLanes>::BM],
    cmp: F,
) where
    T: SimdNative,
    F: Fn(T::SimdXLanes, T::SimdXLanes) -> <<T as SimdNative>::SimdXLanes as SimdXLanes>::BM,
{
    let rhs = T::SimdXLanes::splat(rhs);
    let chunks = lhs.chunks_exact(<T::SimdXLanes as SimdXLanes>::LANES);

    chunks.zip(dst).for_each(|(chunk, dst)| {
        let lhs = T::SimdXLanes::from_slice(chunk);
        *dst = cmp(lhs, rhs);
    });
}

pub unsafe fn ge_scalar<T: SimdNative>(
    lhs: &[T],
    rhs: T,
    dst: &mut [<<T as SimdNative>::SimdXLanes as SimdXLanes>::BM],
) {
    cmp_scalar(lhs, rhs, dst, |a: T::SimdXLanes, b: T::SimdXLanes| {
        a.simd_ge(b).to_bitmask()
    })
}
