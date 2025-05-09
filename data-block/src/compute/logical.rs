//! Null preserving logical operations
//!
//! TBD: Do we really need `avx2`? Bitmap is pretty small

use crate::array::{Array, BooleanArray};
use crate::bitmap::{BitStore, Bitmap};
use crate::compute::combine_validities;
use std::ops::{BitAnd, BitOr, BitXor};

/// This will be optimized for different target feature
#[inline(always)]
fn raw_bitmap_compute<F>(lhs: &[BitStore], rhs: &[BitStore], dst: &mut [BitStore], op: F)
where
    F: Fn(BitStore, BitStore) -> BitStore,
{
    #[cfg(feature = "verify")]
    {
        assert_eq!(lhs.len(), rhs.len());
        assert_eq!(lhs.len(), dst.len());
    }

    lhs.iter()
        .zip(rhs)
        .zip(dst)
        .for_each(|((&lhs, &rhs), dst)| {
            *dst = op(lhs, rhs);
        });
}

macro_rules! and_bitmaps {
    ($lhs:ident, $rhs:ident, $dst:ident) => {
        raw_bitmap_compute($lhs, $rhs, $dst, BitAnd::bitand)
    };
}

crate::dynamic_func!(
    and_bitmaps,
    ,
    (lhs: &[BitStore], rhs: &[BitStore], dst: &mut [BitStore]),
);

macro_rules! or_bitmaps {
    ($lhs:ident, $rhs:ident, $dst:ident) => {
        raw_bitmap_compute($lhs, $rhs, $dst, BitOr::bitor)
    };
}

crate::dynamic_func!(
    or_bitmaps,
    ,
    (lhs: &[BitStore], rhs: &[BitStore], dst: &mut [BitStore]),
);

macro_rules! xor_bitmaps {
    ($lhs:ident, $rhs:ident, $dst:ident) => {
        raw_bitmap_compute($lhs, $rhs, $dst, BitXor::bitxor)
    };
}

crate::dynamic_func!(
    xor_bitmaps,
    ,
    (lhs: &[BitStore], rhs: &[BitStore], dst: &mut [BitStore]),
);

macro_rules! not_xor_bitmaps {
    ($lhs:ident, $rhs:ident, $dst:ident) => {
        raw_bitmap_compute($lhs, $rhs, $dst, |lhs, rhs| !(lhs ^ rhs))
    };
}

crate::dynamic_func!(
    not_xor_bitmaps,
    ,
    (lhs: &[BitStore], rhs: &[BitStore], dst: &mut [BitStore]),
);

macro_rules! and_not_bitmaps {
    ($lhs:ident, $rhs:ident, $dst:ident) => {
        raw_bitmap_compute($lhs, $rhs, $dst, |lhs, rhs| lhs & !rhs)
    };
}

crate::dynamic_func!(
    and_not_bitmaps,
    ,
    (lhs: &[BitStore], rhs: &[BitStore], dst: &mut [BitStore]),
);

macro_rules! or_not_bitmaps {
    ($lhs:ident, $rhs:ident, $dst:ident) => {
        raw_bitmap_compute($lhs, $rhs, $dst, |lhs, rhs| lhs | !rhs)
    };
}

crate::dynamic_func!(
    or_not_bitmaps,
    ,
    (lhs: &[BitStore], rhs: &[BitStore], dst: &mut [BitStore]),
);

macro_rules! not_and_bitmaps {
    ($lhs:ident, $rhs:ident, $dst:ident) => {
        raw_bitmap_compute($lhs, $rhs, $dst, |lhs, rhs| !lhs & rhs)
    };
}

crate::dynamic_func!(
    not_and_bitmaps,
    ,
    (lhs: &[BitStore], rhs: &[BitStore], dst: &mut [BitStore]),
);

macro_rules! not_or_bitmaps {
    ($lhs:ident, $rhs:ident, $dst:ident) => {
        raw_bitmap_compute($lhs, $rhs, $dst, |lhs, rhs| !lhs | rhs)
    };
}

crate::dynamic_func!(
    not_or_bitmaps,
    ,
    (lhs: &[BitStore], rhs: &[BitStore], dst: &mut [BitStore]),
);

/// Perform `&&` operation on two [`BooleanArray`]s, combine the validity
///
/// Note that caller should guarantee `lhs` and `rhs` have same length
///
/// # Safety
///
/// - `lhs`/`rhs`'s data and validity should not reference `dst`'s data and validity. In the computation
///   graph, `lhs`/`rhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
///   computation graph, it will never happens
pub unsafe fn and(lhs: &BooleanArray, rhs: &BooleanArray, dst: &mut BooleanArray) {
    unsafe {
        #[cfg(feature = "verify")]
        assert_eq!(lhs.len(), rhs.len());

        if lhs.validity.count_zeros() == 0 && rhs.validity.count_zeros() == 0 {
            match (lhs.data.count_zeros(), rhs.data.count_zeros()) {
                (0, 0) => {
                    // all values are true on both side
                    dst.reference(lhs);
                    return;
                }
                (l, _) if l == lhs.len() => {
                    // all values are `false` on left side
                    dst.reference(lhs);
                    return;
                }
                (_, r) if r == rhs.len() => {
                    // all values are `false` on right side
                    dst.reference(rhs);
                    return;
                }
                (_, _) => (),
            }
        }

        combine_validities(&lhs.validity, &rhs.validity, &mut dst.validity);
        and_bitmaps_dynamic(
            lhs.data.as_raw_slice(),
            rhs.data.as_raw_slice(),
            dst.data.as_mut().mutate().clear_and_resize(lhs.len()),
        )
    }
}

/// Perform `||` operation on two [`BooleanArray`]s, combine the validity
///
/// Note that caller should guarantee `lhs` and `rhs` have same length
///
/// # Safety
///
/// - `lhs`/`rhs`'s data and validity should not reference `dst`'s data and validity. In the computation
///   graph, `lhs`/`rhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
///   computation graph, it will never happens
pub unsafe fn or(lhs: &BooleanArray, rhs: &BooleanArray, dst: &mut BooleanArray) {
    unsafe {
        #[cfg(feature = "verify")]
        assert_eq!(lhs.len(), rhs.len());

        if lhs.validity.count_zeros() == 0 && rhs.validity.count_zeros() == 0 {
            match (lhs.data.count_zeros(), rhs.data.count_zeros()) {
                (0, _) => {
                    // all values are true on left side
                    dst.data.reference(&lhs.data);
                    dst.validity.reference(&lhs.validity);
                    return;
                }
                (_, 0) => {
                    // all values are `true` on right side
                    dst.data.reference(&rhs.data);
                    dst.validity.reference(&rhs.validity);
                    return;
                }
                (l, r) if l == lhs.len() && r == rhs.len() => {
                    // all values are `false` on both sides
                    dst.data.reference(&rhs.data);
                    dst.validity.reference(&rhs.validity);
                    return;
                }
                (_, _) => (),
            }
        }

        // SAFETY: dst is the unique owner
        combine_validities(&lhs.validity, &rhs.validity, &mut dst.validity);
        or_bitmaps_dynamic(
            lhs.data.as_raw_slice(),
            rhs.data.as_raw_slice(),
            dst.data.as_mut().mutate().clear_and_resize(lhs.len()),
        )
    }
}

/// Perform `^` operation on two [`BooleanArray`]s, combine the validity
///
/// Note that caller should guarantee `lhs` and `rhs` have same length
///
/// # Safety
///
/// - `lhs`/`rhs`'s data and validity should not reference `dst`'s data and validity. In the computation
///   graph, `lhs`/`rhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
///   computation graph, it will never happens
pub unsafe fn xor(lhs: &BooleanArray, rhs: &BooleanArray, dst: &mut BooleanArray) {
    unsafe {
        #[cfg(feature = "verify")]
        assert_eq!(lhs.len(), rhs.len());

        combine_validities(&lhs.validity, &rhs.validity, &mut dst.validity);
        xor_bitmaps_dynamic(
            lhs.data.as_raw_slice(),
            rhs.data.as_raw_slice(),
            dst.data.as_mut().mutate().clear_and_resize(lhs.len()),
        );
    }
}

/// Perform `and` operation on [`BooleanArray`] and scalar
///
/// # Safety
///
/// - `lhs` data and validity should not reference `dst`'s data and validity. In the computation
///   graph, `lhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
///   computation graph, it will never happens
pub unsafe fn and_scalar(lhs: &BooleanArray, rhs: bool, dst: &mut BooleanArray) {
    unsafe {
        if rhs {
            dst.reference(lhs)
        } else {
            dst.validity.reference(&lhs.validity);
            // Compiler will optimize set zero to memset
            dst.data
                .as_mut()
                .mutate()
                .clear_and_resize(lhs.len())
                .iter_mut()
                .for_each(|v| *v = 0);
        }
    }
}

/// Perform `or` operation on [`BooleanArray`] and scalar
///
/// # Safety
///
/// - `lhs` data and validity should not reference `dst`'s data and validity. In the computation
///   graph, `lhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
///   computation graph, it will never happens
pub unsafe fn or_scalar(lhs: &BooleanArray, rhs: bool, dst: &mut BooleanArray) {
    unsafe {
        if rhs {
            dst.validity.reference(&lhs.validity);
            // Compiler will optimize set u64::MAX to memset
            dst.data
                .as_mut()
                .mutate()
                .clear_and_resize(lhs.len())
                .iter_mut()
                .for_each(|v| *v = u64::MAX);
        } else {
            dst.reference(lhs);
        }
    }
}

macro_rules! not_bitmap {
    ($input:ident, $dst:ident) => {
        $input
            .iter()
            .zip($dst)
            .for_each(|(input, dst)| *dst = !input)
    };
}

crate::dynamic_func!(
    not_bitmap,
    ,
    (input: &[BitStore], dst: &mut [BitStore]),
);

/// Perform `NOT` operation on [`BooleanArray`]. If the value is null, then the result
/// is also null
///
/// # Safety
///
/// - `lhs` data and validity should not reference `dst`'s data and validity. In the computation
///   graph, `lhs` must be the descendant of `dst`
///
/// - No other arrays that reference the `dst`'s data and validity are accessed! In the
///   computation graph, it will never happens
pub unsafe fn not(array: &BooleanArray, dst: &mut BooleanArray) {
    unsafe {
        dst.validity.reference(&array.validity);
        not_bitmap_dynamic(
            array.data.as_raw_slice(),
            dst.data.as_mut().mutate().clear_and_resize(array.len()),
        )
    }
}

/// This will be optimized for different target feature
#[inline(always)]
fn raw_bitmap_compute_inplace<F>(dst: &mut [BitStore], other: &[BitStore], op: F)
where
    F: Fn(BitStore, BitStore) -> BitStore,
{
    #[cfg(feature = "verify")]
    assert_eq!(dst.len(), other.len());

    dst.iter_mut().zip(other).for_each(|(dst, &rhs)| {
        *dst = op(*dst, rhs);
    });
}

macro_rules! and_bitmaps_inplace {
    ($dst:ident, $other:ident) => {
        raw_bitmap_compute_inplace($dst, $other, BitAnd::bitand)
    };
}
crate::dynamic_func!(
    and_bitmaps_inplace,
    ,
    (dst: &mut [BitStore], other: &[BitStore]),
);

macro_rules! and_not_bitmaps_inplace {
    ($dst:ident, $other:ident) => {
        raw_bitmap_compute_inplace($dst, $other, |dst, rhs| dst & !rhs)
    };
}

crate::dynamic_func!(
    and_not_bitmaps_inplace,
    ,
    (dst: &mut [BitStore], other: &[BitStore]),
);

macro_rules! or_bitmaps_inplace {
    ($dst:ident, $other:ident) => {
        raw_bitmap_compute_inplace($dst, $other, BitOr::bitor)
    };
}
crate::dynamic_func!(
    or_bitmaps_inplace,
    ,
    (dst: &mut [BitStore], other: &[BitStore]),
);

/// Perform `&` operation on two [`Bitmap`]s
///
/// # Safety
///
/// - `dst` should not referenced by any array
///
/// - If `dst` is not empty, `dst` and `other` should have same length
#[inline]
pub unsafe fn and_inplace(dst: &mut Bitmap, other: &Bitmap) {
    #[cfg(feature = "verify")]
    {
        if !dst.is_empty() && !other.is_empty() {
            assert_eq!(dst.len(), other.len());
        }
    }

    if other.all_valid() {
        // Do nothing
        return;
    }
    if dst.all_valid() {
        // Copy from other. Compiler will optimize it to memory copy
        let mut guard = dst.mutate();
        let uninitialized = guard.clear_and_resize(other.len());
        uninitialized
            .iter_mut()
            .zip(other.as_raw_slice())
            .for_each(|(dst, &src)| *dst = src);
    } else {
        // Perform and operation
        let mut guard = dst.mutate();
        and_bitmaps_inplace_dynamic(guard.as_mut_slice(), other.as_raw_slice());
    }
}

/// Perform `&!` operation on two [`Bitmap`]s
///
/// # Safety
///
/// - `dst` should not referenced by any array
///
/// - If `dst`/`other` is not empty, they should have same length with len
#[inline]
pub unsafe fn and_not_inplace(dst: &mut Bitmap, other: &Bitmap, len: usize) {
    #[cfg(feature = "verify")]
    {
        if !dst.is_empty() {
            assert_eq!(dst.len(), len);
        }

        if !other.is_empty() {
            assert_eq!(other.len(), len);
        }
    }

    if other.all_valid() {
        // & false, set to false
        dst.mutate()
            .clear_and_resize(len)
            .iter_mut()
            .for_each(|v| *v = 0);
    } else if other.count_zeros() != len {
        if dst.is_empty() {
            not_bitmap_dynamic(other.as_raw_slice(), dst.mutate().clear_and_resize(len));
        } else {
            and_not_bitmaps_inplace_dynamic(dst.mutate().as_mut_slice(), other.as_raw_slice());
        }
    } else {
        // & true, do nothing
    }
}

/// Perform `||` operation on two [`Bitmap`]s
///
/// Note that caller should guarantee `dst` and `other` have same length
///
/// # Safety
///
/// - `dst` should not referenced by any array
///
/// - If `dst` is not all valid, `dst` and `other` should have same length
pub unsafe fn or_inplace(dst: &mut Bitmap, other: &Bitmap) {
    #[cfg(feature = "verify")]
    {
        if !dst.is_empty() && !other.is_empty() {
            assert_eq!(dst.len(), other.len());
        }
    }

    if dst.all_valid() {
        // Do nothing
        return;
    }
    if other.all_valid() {
        // Set dst to all true
        let mut guard = dst.mutate();
        let uninitialized = guard.clear_and_resize(other.len());
        uninitialized.iter_mut().for_each(|dst| *dst = 0);
    } else {
        let mut guard = dst.mutate();
        or_bitmaps_inplace_dynamic(guard.as_mut_slice(), other.as_raw_slice());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::LogicalType;

    #[test]
    fn test_and() {
        let lhs = BooleanArray::from_iter([
            Some(true),
            Some(false),
            Some(true),
            Some(true),
            Some(false),
            Some(false),
            Some(false),
        ]);

        let rhs = BooleanArray::from_iter([
            Some(false),
            Some(false),
            Some(true),
            Some(false),
            Some(false),
            Some(true),
            Some(true),
        ]);

        let mut dst = BooleanArray::try_new(LogicalType::Boolean).unwrap();
        unsafe { and(&lhs, &rhs, &mut dst) };
        assert_eq!(
            dst.values_iter().collect::<Vec<_>>(),
            [false, false, true, false, false, false, false]
        );

        // Fast path
        let other = BooleanArray::from_iter(vec![Some(false); 7]);
        unsafe { and(&lhs, &other, &mut dst) };
        assert_eq!(
            dst.values_iter().collect::<Vec<_>>(),
            other.values_iter().collect::<Vec<_>>()
        );

        unsafe { and(&other, &lhs, &mut dst) };
        assert_eq!(
            dst.values_iter().collect::<Vec<_>>(),
            other.values_iter().collect::<Vec<_>>()
        );

        let other_ = BooleanArray::from_iter(vec![Some(true); 7]);
        unsafe { and(&other_, &other_, &mut dst) };
        assert_eq!(
            dst.values_iter().collect::<Vec<_>>(),
            other_.values_iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_or() {
        let lhs = BooleanArray::from_iter([
            Some(true),
            Some(false),
            Some(true),
            Some(true),
            Some(false),
            Some(false),
            Some(false),
        ]);

        let rhs = BooleanArray::from_iter([
            Some(false),
            Some(false),
            Some(true),
            Some(false),
            Some(false),
            Some(true),
            Some(true),
        ]);

        let mut dst = BooleanArray::try_new(LogicalType::Boolean).unwrap();
        unsafe { or(&lhs, &rhs, &mut dst) };
        assert_eq!(
            dst.values_iter().collect::<Vec<_>>(),
            [true, false, true, true, false, true, true]
        );

        // Fast path
        let other = BooleanArray::from_iter(vec![Some(true); 7]);
        unsafe { or(&lhs, &other, &mut dst) };
        assert_eq!(
            dst.values_iter().collect::<Vec<_>>(),
            other.values_iter().collect::<Vec<_>>()
        );

        unsafe { or(&other, &lhs, &mut dst) };
        assert_eq!(
            dst.values_iter().collect::<Vec<_>>(),
            other.values_iter().collect::<Vec<_>>()
        );

        let other_ = BooleanArray::from_iter(vec![Some(false); 7]);
        unsafe { or(&other_, &other_, &mut dst) };
        assert_eq!(
            dst.values_iter().collect::<Vec<_>>(),
            other_.values_iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_logical_scalar() {
        let lhs = BooleanArray::from_iter([
            Some(true),
            Some(false),
            Some(true),
            Some(true),
            Some(false),
            Some(false),
            Some(false),
        ]);

        let dst = &mut BooleanArray::try_new(LogicalType::Boolean).unwrap();
        unsafe { and_scalar(&lhs, true, dst) };
        assert_eq!(
            dst.values_iter().collect::<Vec<_>>(),
            lhs.values_iter().collect::<Vec<_>>()
        );

        unsafe { and_scalar(&lhs, false, dst) };
        assert!(dst.values_iter().all(|v| !v));

        unsafe { or_scalar(&lhs, true, dst) };
        assert!(dst.values_iter().all(|v| v));

        unsafe { or_scalar(&lhs, false, dst) };
        assert_eq!(
            dst.values_iter().collect::<Vec<_>>(),
            lhs.values_iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_inplace() {
        let mut dst = Bitmap::from_slice_and_len(&[0xff, 0x01], 65);
        let other = Bitmap::from_slice_and_len(&[0xff01, 0xff], 65);
        unsafe {
            and_inplace(&mut dst, &other);
        }
        assert_eq!(dst.as_raw_slice(), &[0x01, 0x01]);

        unsafe {
            or_inplace(&mut dst, &other);
        }
        assert_eq!(dst.as_raw_slice(), &[0xff01, 0xff]);
    }
}
