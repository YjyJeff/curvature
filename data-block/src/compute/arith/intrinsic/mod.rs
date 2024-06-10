//! Arithmetic on IntrinsicType

mod add;
mod div;
mod mul;
mod rem;
mod sub;

use crate::array::{Array, PrimitiveArray};
use crate::bitmap::Bitmap;
use crate::compute::combine_validities;
use crate::types::IntrinsicType;
pub use add::{add, add_scalar, AddExt};
pub use div::{div, div_scalar, DivExt};
pub use mul::{mul, mul_scalar, MulExt};
pub use rem::{rem, rem_scalar, RemCast, RemExt};
pub use sub::{sub, sub_scalar, SubExt};

// Add, Sub, Mul operations are supported by cpu, benchmark shows that they usually
// have same threshold based on number of bits
const NATIVE_PARTIAL_ARITH_THRESHOLD_8_BIT: f64 = 0.013;
const NATIVE_PARTIAL_ARITH_THRESHOLD_16_BIT: f64 = 0.026;
const NATIVE_PARTIAL_ARITH_THRESHOLD_32_BIT: f64 = 0.05;
const NATIVE_PARTIAL_ARITH_THRESHOLD_64_BIT: f64 = 0.1;

/// Generic array arith scalar function
#[inline(always)]
unsafe fn arith_scalar<T, U, V>(
    selection: &Bitmap,
    array: &PrimitiveArray<T>,
    scalar: T,
    dst: &mut PrimitiveArray<V>,
    partial_arith_threshold: f64,
    arith_func: impl Fn(T, U) -> V,
    transformer: impl Fn(T) -> U,
) where
    PrimitiveArray<T>: Array,
    T: IntrinsicType,
    V: IntrinsicType,
    U: Copy,
{
    debug_assert_selection_is_valid!(selection, array);
    dst.validity.reference(&array.validity);

    let dst = dst.data.as_mut();
    let uninitialized = dst.clear_and_resize(array.len());

    let array_data = array.data.as_slice();
    let scalar = transformer(scalar);

    if selection.ones_ratio() < partial_arith_threshold {
        selection.iter_ones().for_each(|index| {
            *uninitialized.get_unchecked_mut(index) =
                arith_func(*array_data.get_unchecked(index), scalar);
        });
    } else {
        // Following code will be optimized for different target-feature to perform
        // auto-vectorization
        array_data
            .iter()
            .zip(uninitialized)
            .for_each(|(&element, uninit)| *uninit = arith_func(element, scalar));
    }
}

/// Arithmetic between two arrays
///
/// Allow `into_iter_on_ref` here because rust analyzer has bug. If we use `iter` instead of
/// `into_iter` in the block, the analyzer will tell us an error here. However, the code can
/// compile!
#[inline(always)]
#[allow(clippy::into_iter_on_ref)]
unsafe fn arith_arrays<T, U, V>(
    selection: &Bitmap,
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveArray<U>,
    dst: &mut PrimitiveArray<V>,
    partial_arith_threshold: f64,
    arith_func: impl Fn(T, U) -> V,
) where
    PrimitiveArray<T>: Array,
    PrimitiveArray<U>: Array,
    PrimitiveArray<V>: Array,
    T: IntrinsicType,
    V: IntrinsicType,
    U: IntrinsicType,
{
    debug_assert_selection_is_valid!(selection, lhs);
    debug_assert_eq!(lhs.len(), rhs.len());

    combine_validities(&lhs.validity, &rhs.validity, &mut dst.validity);

    let uninitialized = dst.data.as_mut().clear_and_resize(lhs.len());

    let lhs = lhs.data.as_slice();
    let rhs = rhs.data.as_slice();

    if selection.ones_ratio() < partial_arith_threshold {
        selection.iter_ones().for_each(|index| {
            *uninitialized.get_unchecked_mut(index) =
                arith_func(*lhs.get_unchecked(index), *rhs.get_unchecked(index));
        });
    } else {
        // Following code will be optimized for different target-feature to perform
        // auto-vectorization
        lhs.into_iter()
            .zip(rhs)
            .zip(uninitialized)
            .for_each(|((&lhs, &rhs), uninit)| *uninit = arith_func(lhs, rhs));
    }
}
