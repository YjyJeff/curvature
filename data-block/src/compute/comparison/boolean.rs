//! Comparison between boolean arrays

use crate::array::{Array, BooleanArray};
use crate::bitmap::BitStore;

macro_rules! not_bitmap {
    ($lhs:ident, $dst:ident) => {
        $lhs.iter().zip($dst).for_each(|(&lhs, dst)| {
            *dst = !lhs;
        })
    };
}

crate::dynamic_func!(
    not_bitmap,
    ,
    (lhs: &[BitStore], dst: &mut [BitStore]),
);

/// Perform `lhs == rhs` between [`BooleanArray`] and bool
///
/// # Safety
///
/// - `lhs` data and validity should not reference `dst`'s data and validity. In the computation
/// graph, `lhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn eq_scalar(lhs: &BooleanArray, rhs: bool, dst: &mut BooleanArray) {
    dst.validity.reference(&lhs.validity);

    if rhs {
        dst.data.reference(&lhs.data)
    } else {
        not_bitmap_dynamic(
            lhs.data.as_raw_slice(),
            dst.data.as_mut().mutate().clear_and_resize(lhs.len()),
        );
    }
}

/// Perform `lhs != rhs` between [`BooleanArray`] and bool
///
/// # Safety
///
/// - `lhs`'s data and validity should not reference `dst`'s data and validity. In the computation
/// graph, `lhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn ne_scalar(lhs: &BooleanArray, rhs: bool, dst: &mut BooleanArray) {
    dst.validity.reference(&lhs.validity);

    if !rhs {
        dst.data.reference(&lhs.data)
    } else {
        not_bitmap_dynamic(
            lhs.data.as_raw_slice(),
            dst.data.as_mut().mutate().clear_and_resize(lhs.len()),
        );
    }
}

/// Perform `lhs > rhs` between [`BooleanArray`] and bool
///
/// # Safety
///
/// - `lhs`'s data and validity should not reference `dst`'s data and validity. In the computation
/// graph, `lhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn gt_scalar(lhs: &BooleanArray, rhs: bool, dst: &mut BooleanArray) {
    dst.validity.reference(&lhs.validity);

    if !rhs {
        dst.data.reference(&lhs.data);
    } else {
        // Compiler will optimize it to memset
        dst.data
            .as_mut()
            .mutate()
            .clear_and_resize(lhs.len())
            .iter_mut()
            .for_each(|v| *v = 0);
    }
}

/// Perform `lhs >= rhs` between [`BooleanArray`] and bool
///
/// # Safety
///
/// - `lhs` data and validity should not reference `dst`'s data and validity. In the computation
/// graph, `lhs`/`rhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn ge_scalar(lhs: &BooleanArray, rhs: bool, dst: &mut BooleanArray) {
    dst.validity.reference(&lhs.validity);

    if rhs {
        dst.data.reference(&lhs.data);
    } else {
        // Compiler will optimize it to memset
        dst.data
            .as_mut()
            .mutate()
            .clear_and_resize(lhs.len())
            .iter_mut()
            .for_each(|v| *v = u64::MAX);
    }
}

/// Perform `lhs < rhs` between [`BooleanArray`] and bool
///
/// # Safety
///
/// - `lhs`'s data and validity should not reference `dst`'s data and validity. In the computation
/// graph, `lhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn lt_scalar(lhs: &BooleanArray, rhs: bool, dst: &mut BooleanArray) {
    dst.validity.reference(&lhs.validity);

    let mut dst = dst.data.as_mut().mutate();
    if rhs {
        not_bitmap_dynamic(lhs.data.as_raw_slice(), dst.clear_and_resize(lhs.len()));
    } else {
        // Compiler will optimize it to memset
        dst.clear_and_resize(lhs.len())
            .iter_mut()
            .for_each(|v| *v = 0);
    }
}

/// Perform `lhs <= rhs` between [`BooleanArray`] and bool
///
/// # Safety
///
/// - `lhs`'s data and validity should not reference `dst`'s data and validity. In the computation
/// graph, `lhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
/// computation graph, it will never happens
pub unsafe fn le_scalar(lhs: &BooleanArray, rhs: bool, dst: &mut BooleanArray) {
    dst.validity.reference(&lhs.validity);

    let mut dst = dst.data.as_mut().mutate();
    if rhs {
        // Compiler will optimize it to memset
        dst.clear_and_resize(lhs.len())
            .iter_mut()
            .for_each(|v| *v = u64::MAX);
    } else {
        not_bitmap_dynamic(lhs.data.as_raw_slice(), dst.clear_and_resize(lhs.len()))
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Deref;

    use crate::bitmap::Bitmap;
    use crate::types::LogicalType;

    use super::*;

    fn test_reverse(dst: &Bitmap, lhs: &Bitmap) {
        assert_eq!(
            dst.iter().map(|v| !v).collect::<Vec<_>>(),
            lhs.iter().collect::<Vec<_>>()
        );
    }

    fn test_all_true(dst: &Bitmap) {
        dst.iter().all(|v| v);
    }

    fn test_all_false(dst: &Bitmap) {
        dst.iter().all(|v| !v);
    }

    #[test]
    fn test_equal() {
        let lhs = BooleanArray::new_with_data(Bitmap::from_slice_and_len(&[0xf371], 16));

        let mut dst = BooleanArray::new(LogicalType::Boolean).unwrap();
        unsafe { eq_scalar(&lhs, true, &mut dst) };
        assert_eq!(dst.data.deref(), lhs.data.deref());

        let lhs = BooleanArray::new_with_data(Bitmap::from_slice_and_len(&[0xf371], 40));

        unsafe { eq_scalar(&lhs, false, &mut dst) };
        test_reverse(&dst.data, &lhs.data);

        unsafe { ne_scalar(&lhs, true, &mut dst) };
        test_reverse(&dst.data, &lhs.data);

        unsafe { ne_scalar(&lhs, false, &mut dst) };
        assert_eq!(dst.data.deref(), lhs.data.deref());
    }

    #[test]
    fn test_order() {
        let lhs = BooleanArray::new_with_data(Bitmap::from_slice_and_len(&[0x52e3, 0xd2c8], 100));

        let mut dst = BooleanArray::new(LogicalType::Boolean).unwrap();
        unsafe { gt_scalar(&lhs, false, &mut dst) };
        assert_eq!(dst.data.deref(), lhs.data.deref());

        unsafe { gt_scalar(&lhs, true, &mut dst) };
        test_all_false(&dst.data);

        unsafe { ge_scalar(&lhs, false, &mut dst) };
        test_all_true(&dst.data);

        unsafe { ge_scalar(&lhs, true, &mut dst) };
        assert_eq!(dst.data.deref(), lhs.data.deref());

        unsafe { lt_scalar(&lhs, false, &mut dst) };
        test_all_false(&dst.data);

        unsafe { lt_scalar(&lhs, true, &mut dst) };
        test_reverse(&dst.data, &lhs.data);

        unsafe { le_scalar(&lhs, false, &mut dst) };
        test_reverse(&dst.data, &lhs.data);

        unsafe { le_scalar(&lhs, true, &mut dst) };
        test_all_true(&dst.data)
    }
}
