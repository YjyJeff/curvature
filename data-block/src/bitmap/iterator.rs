//! Iterator of the bitmap

use std::iter::FusedIterator;

use super::{get_bit_unchecked, BitStore, Bitmap, BIT_STORE_BITS};
use crate::aligned_vec::AlignedVec;

#[derive(Debug)]
/// Iterator of the bitmap
pub struct BitmapIter<'a> {
    buffer: &'a AlignedVec<BitStore>,
    bit_index: usize,
    end: usize,
}

impl<'a> BitmapIter<'a> {
    /// Create a new iterator of the bitmap
    pub fn new(bitmap: &'a Bitmap) -> Self {
        Self {
            buffer: &bitmap.buffer,
            bit_index: 0,
            end: bitmap.num_bits,
        }
    }

    /// Create a new iterator from the bitmap with given offset and length
    pub fn new_with_offset_and_len(bitmap: &'a Bitmap, offset: usize, length: usize) -> Self {
        let end = offset + length;
        assert!(end <= bitmap.num_bits);
        Self {
            buffer: &bitmap.buffer,
            bit_index: offset,
            end,
        }
    }
}

impl Iterator for BitmapIter<'_> {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.bit_index == self.end {
            return None;
        }

        let old = self.bit_index;
        self.bit_index += 1;
        Some(unsafe { get_bit_unchecked(self.buffer, old) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.end - self.bit_index;
        (size, Some(size))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let new_index = self.bit_index + n;
        if new_index >= self.end {
            self.bit_index = self.end;
            None
        } else {
            self.bit_index = new_index;
            self.next()
        }
    }
}

impl ExactSizeIterator for BitmapIter<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.end - self.bit_index
    }
}

impl FusedIterator for BitmapIter<'_> {}

/// Iterator of the index that is set in the bitmap
#[derive(Debug)]
pub struct BitmapOnesIter<'a> {
    bitmap: &'a Bitmap,
    current: BitStore,
    bit_store_index: usize,
}

impl<'a> BitmapOnesIter<'a> {
    /// Create a new [`BitmapOnesIter`]
    pub(super) fn new(bitmap: &'a Bitmap) -> Self {
        if bitmap.num_bits == 0 {
            Self {
                bitmap,
                current: 0,
                bit_store_index: 0,
            }
        } else {
            // SAFETY: this branch guarantees the bitmap at least have one BitStore
            let current = unsafe { *bitmap.buffer.get_unchecked(0) };
            Self {
                bitmap,
                current,
                bit_store_index: 0,
            }
        }
    }
}

impl Iterator for BitmapOnesIter<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.current == 0 {
            self.bit_store_index += 1;
            if self.bit_store_index == self.bitmap.buffer.len() {
                return None;
            }

            self.current = unsafe { *self.bitmap.buffer.get_unchecked(self.bit_store_index) };

            if ((self.bit_store_index + 1) * BIT_STORE_BITS) > self.bitmap.num_bits {
                // Last BitStore, only remain the valid data
                self.current &= (1 << (self.bitmap.num_bits & (BIT_STORE_BITS - 1))) - 1;
            }
        }

        let index =
            (self.bit_store_index * BIT_STORE_BITS) + self.current.trailing_zeros() as usize;

        self.current &= self.current - 1;
        Some(index)
    }
}

impl FusedIterator for BitmapOnesIter<'_> {}
