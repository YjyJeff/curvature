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
use crate::bitmap::{BitStore, Bitmap};
use crate::compute::comparison::primitive::{
    eq_scalar_default_, ge_scalar_default_, gt_scalar_default_, le_scalar_default_,
    lt_scalar_default_, ne_scalar_default_,
};
use crate::compute::logical::and_inplace;

// TBD: Following implementation heavily depends on the reading intrinsic types from
// uninitialized  memory optimization. However, it is undefined behavior in the Rust.
// See following links for details. In our cases, the cpu is the main reason to avoid
// reading uninitialized memory. Should we remove this optimization?
//
// https://doc.rust-lang.org/nightly/reference/behavior-considered-undefined.html
// https://www.ralfj.de/blog/2019/07/14/uninit.html
// https://stackoverflow.com/questions/11962457/why-is-using-an-uninitialized-variable-undefined-behavior
// https://langdev.stackexchange.com/questions/2870/why-would-accessing-uninitialized-memory-necessarily-be-undefined-behavior

/// Miri does not support neon, see [issue](https://github.com/rust-lang/miri/issues/3243)
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod arm;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

macro_rules! cmp_scalar {
    ($func:ident, $op:tt) => {
        paste::paste! {
            #[doc = concat!("Perform `", stringify!($op), "` operation between array and scalar, write the result to dst")]
            /// # Panic
            ///
            /// Invariance: `dst.len() = roundup_loops(array.len(), 64)` should hold
            #[inline]
            fn $func(array: &AlignedVec<Self>, scalar: Self, dst: &mut [BitStore]) {
                debug_assert_eq!(dst.len(), crate::utils::roundup_loops(array.len(), 64));

                let uninitialized = dst.as_mut_ptr();

                // SAFETY: The uninitialized slice has adequate valid space via `clear_and_resize`
                unsafe {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        if std::arch::is_x86_feature_detected!("avx2") {
                            return Self::[<$func:upper _AVX2>](array, scalar, uninitialized);
                        } else {
                            return Self::[<$func:upper _SSE2>](array, scalar, uninitialized);
                        }
                    }

                    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
                    {
                        if std::arch::is_aarch64_feature_detected!("neon") {
                            return Self::[<$func:upper _NEON>](array, scalar, uninitialized);
                        }
                    }

                    [<$func _default_>](array, scalar, uninitialized)
                }
            }
        }
    };
}

type CmpFunc<T> = unsafe fn(&AlignedVec<T>, T, *mut BitStore);

/// TODO:
/// - sse4.1 for i64/u64
/// - Should we provide two traits: IntrinsicCmpElement and PrimitiveCmpElement?
///
/// Comparison array with scalar functions associated with each primitive type
pub trait PrimitiveCmpElement: PrimitiveType + PartialOrd {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// Equal function implemented with avx2
    const EQ_SCALAR_AVX2: CmpFunc<Self>;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// Equal function implemented with sse2
    const EQ_SCALAR_SSE2: CmpFunc<Self>;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// NotEqual function implemented with avx2
    const NE_SCALAR_AVX2: CmpFunc<Self>;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// NotEqual function implemented with sse2
    const NE_SCALAR_SSE2: CmpFunc<Self>;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// Greater than function implemented with avx2
    const GT_SCALAR_AVX2: CmpFunc<Self>;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// Greater than function implemented with sse2
    const GT_SCALAR_SSE2: CmpFunc<Self>;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// Greater than or equal to function implemented with avx2
    const GE_SCALAR_AVX2: CmpFunc<Self>;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// Greater than or equal to function implemented with sse2
    const GE_SCALAR_SSE2: CmpFunc<Self>;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// Less than function implemented with avx2
    const LT_SCALAR_AVX2: CmpFunc<Self>;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// Less than function implemented with sse2
    const LT_SCALAR_SSE2: CmpFunc<Self>;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// Less than or equal to function implemented with avx2
    const LE_SCALAR_AVX2: CmpFunc<Self>;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    /// Less than or equal to function implemented with sse2
    const LE_SCALAR_SSE2: CmpFunc<Self>;
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    /// Equal function implemented with neon
    const EQ_SCALAR_NEON: CmpFunc<Self>;
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    /// NotEqual function implemented with neon
    const NE_SCALAR_NEON: CmpFunc<Self>;
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    /// Greater than function implemented with neon
    const GT_SCALAR_NEON: CmpFunc<Self>;
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    /// Greater than or equal to function implemented with neon
    const GE_SCALAR_NEON: CmpFunc<Self>;
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    /// Less than function implemented with neon
    const LT_SCALAR_NEON: CmpFunc<Self>;
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    /// Less than or equal to function implemented with neon
    const LE_SCALAR_NEON: CmpFunc<Self>;

    cmp_scalar!(eq_scalar, ==);
    cmp_scalar!(ne_scalar, !=);
    cmp_scalar!(gt_scalar, >);
    cmp_scalar!(ge_scalar, >=);
    cmp_scalar!(lt_scalar, <);
    cmp_scalar!(le_scalar, <=);
}

/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
///
/// - `array`'s data and validity should not reference `dst`'s data and validity. In the computation
/// graph, `lhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
/// computation graph, it will never happens
#[inline]
unsafe fn cmp_scalar<T>(
    selection: &mut Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    dst: &mut BooleanArray,
    cmp_func: impl FnOnce(&AlignedVec<T>, T, &mut [BitStore]),
    scalar_cmp_func: impl Fn(T, T) -> bool,
) where
    T: PrimitiveCmpElement,
    PrimitiveArray<T>: Array<Element = T>,
{
    debug_assert_selection_is_valid!(selection, array);

    and_inplace(selection, array.validity());

    // FIXME: According to selection, determine full or partial. Tune the parameter.
    if selection.ones_ratio() < 0.0 {
        // Partial computation
        selection
            .mutate()
            .mutate_ones(|index| scalar_cmp_func(array.get_value_unchecked(index), scalar));
    } else {
        // Full computation
        cmp_func(
            &array.data,
            scalar,
            dst.data_mut().mutate().clear_and_resize(array.len()),
        );
        and_inplace(selection, dst.data());
    }
}

/// Perform `array == scalar` between `PrimitiveArray<T>` and `T`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
///
/// - `array`'s data and validity should not reference `dst`'s data and validity. In the computation
/// graph, `lhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn eq_scalar<T>(
    selection: &mut Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    dst: &mut BooleanArray,
) where
    T: PrimitiveCmpElement,
    PrimitiveArray<T>: Array<Element = T>,
{
    cmp_scalar(selection, array, scalar, dst, T::eq_scalar, |lhs, rhs| {
        lhs == rhs
    })
}

/// Perform `array != scalar` between `PrimitiveArray<T>` and `T`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
///
/// - `array`'s data and validity should not reference `dst`'s data and validity. In the computation
/// graph, `lhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn ne_scalar<T>(
    selection: &mut Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    dst: &mut BooleanArray,
) where
    T: PrimitiveCmpElement,
    PrimitiveArray<T>: Array<Element = T>,
{
    cmp_scalar(selection, array, scalar, dst, T::ne_scalar, |lhs, rhs| {
        lhs != rhs
    })
}

/// Perform `array > scalar` between `PrimitiveArray<T>` and `T`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
///
/// - `array`'s data and validity should not reference `dst`'s data and validity. In the computation
/// graph, `lhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn gt_scalar<T>(
    selection: &mut Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    dst: &mut BooleanArray,
) where
    T: PrimitiveCmpElement,
    PrimitiveArray<T>: Array<Element = T>,
{
    cmp_scalar(selection, array, scalar, dst, T::gt_scalar, |lhs, rhs| {
        lhs > rhs
    })
}

/// Perform `array >= scalar` between `PrimitiveArray<T>` and `T`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
///
/// - `array`'s data and validity should not reference `dst`'s data and validity. In the computation
/// graph, `lhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn ge_scalar<T>(
    selection: &mut Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    dst: &mut BooleanArray,
) where
    T: PrimitiveCmpElement,
    PrimitiveArray<T>: Array<Element = T>,
{
    cmp_scalar(selection, array, scalar, dst, T::ge_scalar, |lhs, rhs| {
        lhs >= rhs
    })
}

/// Perform `array < scalar` between `PrimitiveArray<T>` and `T`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
///
/// - `array`'s data and validity should not reference `dst`'s data and validity. In the computation
/// graph, `lhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn lt_scalar<T>(
    selection: &mut Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    dst: &mut BooleanArray,
) where
    T: PrimitiveCmpElement,
    PrimitiveArray<T>: Array<Element = T>,
{
    cmp_scalar(selection, array, scalar, dst, T::lt_scalar, |lhs, rhs| {
        lhs < rhs
    })
}

/// Perform `array <= scalar` between `PrimitiveArray<T>` and `T`
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
/// Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
///
/// - `array`'s data and validity should not reference `dst`'s data and validity. In the computation
/// graph, `lhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn le_scalar<T>(
    selection: &mut Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    dst: &mut BooleanArray,
) where
    T: PrimitiveCmpElement,
    PrimitiveArray<T>: Array<Element = T>,
{
    cmp_scalar(selection, array, scalar, dst, T::le_scalar, |lhs, rhs| {
        lhs <= rhs
    })
}
