//! SIMD with stable rust
//!
//! LLVM/Rust/C++ can not perform auto-vectorization for comparing primitive types and
//! write the result to bitmap. It sounds counterintuitive but it is the [truth] 😂.
//! We manually implement this case with SIMD instructions to accelerate the comparison.
//! Benchmarks shows we got a huge performance improvement 😊.
//!
//! After writing SIMD code manually, I know why LLVM generates strange SIMD code
//!
//! [truth]: (https://stackoverflow.com/questions/77350870/why-llvm-can-not-auto-vectorize-comparing-two-arrays-and-write-result-to-vector?noredirect=1#comment136366602_77350870)

use crate::aligned_vec::AlignedVec;
use crate::array::primitive::PrimitiveType;
use crate::array::{Array, BooleanArray, PrimitiveArray};
use crate::bitmap::BitStore;
use crate::compute::comparison::primitive::{
    eq_scalar_default_, ge_scalar_default_, gt_scalar_default_, le_scalar_default_,
    lt_scalar_default_, ne_scalar_default_,
};
use crate::mutate_array_func;

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod arm;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

type CmpFunc<T> = unsafe fn(&AlignedVec<T>, T, *mut BitStore);

/// TODO:
/// - sse4.1 and neon
/// - Should we provide two traits: IntrinsicCmpScalar and PrimitiveCmpScalar?
///
/// Comparison array with scalar functions associated with each primitive type
pub trait PrimitiveCmpScalar: PrimitiveType + PartialOrd {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// Equal function implemented with avx2
    const EQ_FUNC_AVX2: CmpFunc<Self>;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// Equal function implemented with sse2
    const EQ_FUNC_SSE2: CmpFunc<Self>;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// NotEqual function implemented with avx2
    const NE_FUNC_AVX2: CmpFunc<Self>;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// NotEqual function implemented with sse2
    const NE_FUNC_SSE2: CmpFunc<Self>;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// Greater than function implemented with avx2
    const GT_FUNC_AVX2: CmpFunc<Self>;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// Greater than function implemented with sse2
    const GT_FUNC_SSE2: CmpFunc<Self>;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// Greater than or equal to function implemented with avx2
    const GE_FUNC_AVX2: CmpFunc<Self>;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// Greater than or equal to function implemented with sse2
    const GE_FUNC_SSE2: CmpFunc<Self>;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// Less than function implemented with avx2
    const LT_FUNC_AVX2: CmpFunc<Self>;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// Less than function implemented with sse2
    const LT_FUNC_SSE2: CmpFunc<Self>;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// Less than or equal to function implemented with avx2
    const LE_FUNC_AVX2: CmpFunc<Self>;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// Less than or equal to function implemented with sse2
    const LE_FUNC_SSE2: CmpFunc<Self>;
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    /// Equal function implemented with neon
    const EQ_FUNC_NEON: CmpFunc<Self>;
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    /// NotEqual function implemented with neon
    const NE_FUNC_NEON: CmpFunc<Self>;
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    /// Greater than function implemented with neon
    const GT_FUNC_NEON: CmpFunc<Self>;
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    /// Greater than or equal to function implemented with neon
    const GE_FUNC_NEON: CmpFunc<Self>;
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    /// Less than function implemented with neon
    const LT_FUNC_NEON: CmpFunc<Self>;
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    /// Less than or equal to function implemented with neon
    const LE_FUNC_NEON: CmpFunc<Self>;
}

macro_rules! cmp_scalar {
    ($func_name:ident, $cmp_avx2:ident, $cmp_sse2:ident, $cmp_neon:ident, $cmp_default:ident, $op:tt) => {
        mutate_array_func!(
            #[cfg_attr(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"), allow(unreachable_code))]
            #[doc = concat!("Perform `lhs ", stringify!($op), " rhs` between `PrimitiveArray<T>` and `T`")]
            pub unsafe fn $func_name<T>(lhs: &PrimitiveArray<T>, rhs: T, dst: &mut BooleanArray)
            where
                T: PrimitiveCmpScalar,
                PrimitiveArray<T>: Array,
            {
                // Reference dst's validity to lhs'validity
                dst.validity.reference(&lhs.validity);

                let uninitialized = dst.data.exactly_once_mut().clear_and_resize(lhs.len()).as_mut_ptr();

                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    if std::arch::is_x86_feature_detected!("avx2") {
                        return T::$cmp_avx2(&lhs.data, rhs, uninitialized);
                    } else {
                        return T::$cmp_sse2(&lhs.data, rhs, uninitialized);
                    }
                }

                #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
                {
                    if std::arch::is_aarch64_feature_detected!("neon"){
                        return T::$cmp_neon(&lhs.data, rhs, uninitialized);
                    }
                }

                $cmp_default(&lhs.data, rhs, uninitialized);
            }
        );
    };
}

cmp_scalar!(eq_scalar, EQ_FUNC_AVX2, EQ_FUNC_SSE2, EQ_FUNC_NEON, eq_scalar_default_, ==);
cmp_scalar!(ne_scalar, NE_FUNC_AVX2, NE_FUNC_SSE2, NE_FUNC_NEON, ne_scalar_default_, !=);
cmp_scalar!(gt_scalar, GT_FUNC_AVX2, GT_FUNC_SSE2, GT_FUNC_NEON, gt_scalar_default_, >);
cmp_scalar!(ge_scalar, GE_FUNC_AVX2, GE_FUNC_SSE2, GE_FUNC_NEON, ge_scalar_default_, >=);
cmp_scalar!(lt_scalar, LT_FUNC_AVX2, LT_FUNC_SSE2, LT_FUNC_NEON, lt_scalar_default_, <);
cmp_scalar!(le_scalar, LE_FUNC_AVX2, LE_FUNC_SSE2, LE_FUNC_NEON, le_scalar_default_, <=);
