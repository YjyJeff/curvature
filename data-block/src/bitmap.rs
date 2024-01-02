//! Bitmap

use std::fmt::Debug;

use crate::aligned_vec::AlignedVec;
use bitvec::index::BitIdx;
use bitvec::order::Lsb0;
use bitvec::ptr::BitPtr;
use bitvec::slice::{from_raw_parts_unchecked, from_raw_parts_unchecked_mut, BitSlice};
use wyz::Address;

/// Underling type that stores the bitmap
pub(crate) type BitStore = u64;

/// Iterator of the bitmap
pub type BitValIter<'a> = bitvec::slice::BitValIter<'a, BitStore, Lsb0>;
/// Iterator that produce the index of the true
pub type IterOnes<'a> = bitvec::slice::IterOnes<'a, BitStore, Lsb0>;
/// A mutable reference to the bit
pub type BitRef<'a> = bitvec::prelude::BitRef<'a, bitvec::ptr::Mut, BitStore, Lsb0>;

/// Bitmap in data-block, each boolean is stored as a single bit
///
/// Note that array all of the elements are not null, the [`Bitmap`] could be empty
pub struct Bitmap {
    /// Internal buffer stores the bits
    buffer: AlignedVec<BitStore>,
    /// Number of live bits in the allocation
    num_bits: usize,
}

/// FIXME: Do not use bitvec, 💩!💩!💩!💩!
impl Bitmap {
    /// Create a new [`Bitmap`]
    #[inline]
    pub fn new() -> Self {
        Self {
            buffer: AlignedVec::new(),
            num_bits: 0,
        }
    }

    /// Create a new [`Bitmap`] with given capacity
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: AlignedVec::with_capacity(elts(capacity)),
            num_bits: 0,
        }
    }

    #[inline]
    pub(crate) fn as_bitslice(&self) -> &BitSlice<BitStore, Lsb0> {
        unsafe {
            from_raw_parts_unchecked(
                BitPtr::new_unchecked(Address::new(self.buffer.ptr), BitIdx::MIN),
                self.num_bits,
            )
        }
    }

    #[inline]
    pub(crate) fn as_bitslice_mut(&mut self) -> &mut BitSlice<BitStore, Lsb0> {
        unsafe {
            from_raw_parts_unchecked_mut(
                BitPtr::new_unchecked(Address::new(self.buffer.ptr), BitIdx::MIN),
                self.num_bits,
            )
        }
    }

    /// Get the underling raw_slice for the bitmap
    #[inline]
    pub fn as_raw_slice(&self) -> &[BitStore] {
        self.buffer.as_slice()
    }

    /// Returns true if the bitmap is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_bits == 0
    }

    /// Get number of bits in the bitmap
    #[inline]
    pub fn len(&self) -> usize {
        self.num_bits
    }

    /// Get the iterator that produce bool
    #[inline]
    pub fn iter(&self) -> BitValIter<'_> {
        self.as_bitslice().iter().by_vals()
    }

    /// Get the iterator that produce the index that is true
    #[inline]
    pub fn iter_ones(&self) -> IterOnes<'_> {
        self.as_bitslice().iter_ones()
    }

    /// Get a reference to a single bit without bound check
    ///
    /// # Safety
    ///
    /// Same with [BitSlice::get_unchecked]
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> bool {
        *self.as_bitslice().get_unchecked(index).as_ref()
    }

    /// Get a mutable reference to a single bit without bound check
    ///
    /// # Safety
    ///
    /// Same with [BitSlice::get_unchecked]
    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> BitRef<'_> {
        self.as_bitslice_mut().get_unchecked_mut(index)
    }

    /// Count the number of zeros in the bitmap
    #[inline]
    pub fn count_zeros(&self) -> usize {
        self.as_bitslice().count_zeros()
    }

    /// Count the number of oness in the bitmap
    #[inline]
    pub fn count_ones(&self) -> usize {
        self.as_bitslice().count_ones()
    }

    /// Resize the [`AlignedVec`] to new_len, all of the visible length will
    /// be uninitialized(Actually, partial of the Vec remain the old value 😊). It is
    /// caller's responsibility to init the visible region.
    #[inline]
    #[must_use]
    pub fn clear_and_resize(&mut self, new_len: usize) -> &mut [u64] {
        self.num_bits = new_len;
        self.buffer.clear_and_resize(elts(new_len))
    }

    /// Clear the bitmap, it only set the num_bits to 0 and do not free the
    /// underling buffer
    #[inline]
    pub(crate) fn clear(&mut self) {
        self.num_bits = 0;
    }

    /// Appends a single bit into the vec
    ///
    /// Do not use it in the query engine !!!! It is very slow !!!!
    pub(crate) fn push(&mut self, val: bool) {
        let old_len = self.num_bits;
        self.num_bits += 1;
        if old_len == 0 || (self.num_bits >> 6) > (old_len >> 6) {
            // We need to place the new bit in a new element
            self.buffer.reserve(1);
            self.buffer.len += 1;
        }
        // SAFETY: we increase the number of bits and reserve the space
        unsafe {
            self.as_bitslice_mut()
                .get_unchecked_mut(old_len)
                .commit(val);
        }
    }

    /// # Safety
    ///
    /// - len should equal to trusted_iter's length
    #[inline]
    pub(crate) unsafe fn reset(
        &mut self,
        len: usize,
        trusted_len_iter: impl Iterator<Item = bool>,
    ) {
        let uninitialized = self.clear_and_resize(len);
        reset_bitmap_raw(uninitialized.as_mut_ptr(), len, trusted_len_iter)
    }
}

impl Default for Bitmap {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Debug for Bitmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Bitmap {{ data: {}, len: {} }}",
            self.as_bitslice(),
            self.num_bits
        )
    }
}

impl PartialEq for Bitmap {
    fn eq(&self, other: &Self) -> bool {
        self.as_bitslice().eq(other.as_bitslice())
    }
}

/// Compute the number of [`BitStore`] to store the required number of bits
#[inline]
fn elts(num_bits: usize) -> usize {
    (num_bits >> 6) + (num_bits % 64 != 0) as usize
}

/// Copied from arrow2
/// Awesome! Use this function to mutate the bitmap is much much faster than mutating
/// the [`BitRef`] produced by [`BitSlice`]
///
/// # Safety
///
/// len should equal to trusted_iter.len()
///
/// [`BitRef`]: bitvec::prelude::BitRef
/// [`BitSlice`]: bitvec::prelude::BitSlice
#[inline]
pub(crate) unsafe fn reset_bitmap_raw(
    ptr: *mut BitStore,
    len: usize,
    mut trusted_len_iter: impl Iterator<Item = bool>,
) {
    let num_bit_store = len >> 6;
    for index in 0..num_bit_store {
        let mut mask;
        let mut tmp = BitStore::default();
        for i in 0..8 {
            mask = 1 << (i << 3);
            for _ in 0..8 {
                match trusted_len_iter.next() {
                    Some(value) => {
                        tmp |= if value { mask } else { 0 };
                        mask <<= 1;
                    }
                    None => std::hint::unreachable_unchecked(),
                };
            }
        }
        *ptr.add(index) = tmp;
    }

    let remainder = len & 63;
    if remainder > 0 {
        let mut tmp = BitStore::default();
        let num_bytes = remainder / 8;
        for i in 0..num_bytes {
            let mut mask = 1 << (i << 3);
            for _ in 0..8 {
                match trusted_len_iter.next() {
                    Some(value) => {
                        tmp |= if value { mask } else { 0 };
                        mask <<= 1;
                    }
                    None => std::hint::unreachable_unchecked(),
                };
            }
        }

        let remainder = remainder & 7;
        let mut mask = 1 << (num_bytes << 3);
        for _ in 0..remainder {
            match trusted_len_iter.next() {
                Some(value) => {
                    tmp |= if value { mask } else { 0 };
                    mask <<= 1;
                }
                None => std::hint::unreachable_unchecked(),
            };
        }

        *ptr.add(num_bit_store) = tmp;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl Bitmap {
        /// Construct self from slice of [`BitStore`] and len. Caller should
        /// guarantee the len is valid. Otherwise, panic
        pub fn from_slice_and_len(slice: &[BitStore], len: usize) -> Self {
            if len >= std::mem::size_of_val(slice) * 8 {
                panic!(
                    "len: {} out of bounds, slice's length is: {}",
                    len,
                    slice.len()
                );
            }

            Self {
                buffer: AlignedVec::from_slice(slice),
                num_bits: len,
            }
        }
    }

    #[test]
    fn test_empty_bit_slice() {
        let bitmap = Bitmap::new();
        bitmap.iter().for_each(|v| println!("{v}"));
    }
}
