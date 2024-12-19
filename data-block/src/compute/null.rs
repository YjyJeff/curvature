//! Check null

use super::logical::not_bitmap_dynamic;
use crate::array::ArrayImpl;
use crate::bitmap::{BitStore, Bitmap};

/// Check the validity of the selected value in the array.
///
/// Note that the implementation will not reference the array's validity. It may copy data
/// from it because the bitmap is pretty small, it is efficient
///
/// # Safety
///
/// - If the `selection` is not empty, `array` and `selection` should have same length.
///   Otherwise, undefined behavior happens
///
/// - `selection` should not be referenced by any array
pub unsafe fn is_null<const NOT: bool>(selection: &mut Bitmap, array: &ArrayImpl) {
    #[cfg(feature = "verify")]
    assert_selection_is_valid!(selection, array);

    let validity = array.validity();
    let mut guard = selection.mutate();
    let uninitialized = guard.clear_and_resize(array.len());
    if NOT {
        // Compiler will optimize it to memset
        if validity.all_valid() {
            uninitialized
                .iter_mut()
                .for_each(|bit_store| *bit_store = BitStore::MAX);
        } else {
            // Copy from arrays validity. Compiler will optimize it to memcpy
            uninitialized
                .iter_mut()
                .zip(validity.as_raw_slice())
                .for_each(|(dst, &src)| {
                    *dst = src;
                });
        }
    } else if validity.all_valid() {
        // Compiler will optimize it to memset
        uninitialized
            .iter_mut()
            .for_each(|bit_store| *bit_store = 0);
    } else {
        // Using SIMD to accelerate
        not_bitmap_dynamic(array.validity().as_raw_slice(), uninitialized)
    }
}
