//! Check null

use super::logical::not_bitmap_dynamic;
use crate::array::{Array, ArrayImpl, BooleanArray};
use crate::bitmap::{BitStore, Bitmap};

/// # Safety
///
/// - `array` data and validity should not reference `dst`'s data and validity. In the computation
/// graph, `array` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn is_null<const NOT: bool>(array: &ArrayImpl, dst: &mut BooleanArray) {
    dst.validity_mut().mutate().clear();
    if NOT {
        dst.validity.reference(array.swar_validity());
    } else {
        not_bitmap_dynamic(
            array.validity().as_raw_slice(),
            dst.data.as_mut().mutate().as_mut_slice(),
        )
    }
}

/// Check the validity of the selected value in the array.
///
/// Note that the implementation will not reference the array's validity. It may copy data
/// from it because the bitmap is pretty small, it is efficient
///
/// # Safety
///
/// - `array` and `selection` should have same length. Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn selected_is_null<const NOT: bool>(selection: &mut Bitmap, array: &ArrayImpl) {
    let validity = array.validity();
    let mut guard = selection.mutate();
    if NOT {
        // Compiler will optimize it to memset
        if validity.all_valid() {
            guard
                .as_mut_slice()
                .iter_mut()
                .for_each(|bit_store| *bit_store = BitStore::MAX);
        } else {
            // Copy from arrays validity. Compiler will optimize it to memcpy
            guard
                .as_mut_slice()
                .iter_mut()
                .zip(validity.as_raw_slice())
                .for_each(|(dst, &src)| {
                    *dst = src;
                });
        }
    } else if validity.all_valid() {
        // Compiler will optimize it to memset
        guard
            .as_mut_slice()
            .iter_mut()
            .for_each(|bit_store| *bit_store = 0);
    } else {
        // Using SIMD to accelerate
        not_bitmap_dynamic(array.validity().as_raw_slice(), guard.as_mut_slice())
    }
}
