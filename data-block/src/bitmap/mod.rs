//! Bitmap

use std::fmt::Debug;

use crate::aligned_vec::AlignedVec;
use crate::dynamic_func;
mod iterator;

/// Underling type that stores the bitmap
pub type BitStore = u64;
/// Number of bits the bit store contains
const BIT_STORE_BITS: usize = std::mem::size_of::<BitStore>() * 8;

pub use self::iterator::{BitmapIter, BitmapOnesIter};

/// Bitmap in data-block, each boolean is stored as a single bit
///
/// Note that array all of the elements are not null, the [`Bitmap`] could be empty
pub struct Bitmap {
    /// Internal buffer stores the bits
    buffer: AlignedVec<BitStore>,
    /// Number of live bits in the allocation
    num_bits: usize,
}

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
    pub fn iter(&self) -> BitmapIter<'_> {
        BitmapIter::new(self)
    }

    /// Get the iterator that produce the index that is true
    #[inline]
    pub fn iter_ones(&self) -> BitmapOnesIter<'_> {
        BitmapOnesIter::new(self)
    }

    /// Get a reference to a single bit without bound check
    ///
    /// # Safety
    ///
    /// `index <= self.len()`, otherwise undefined behavior happens
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> bool {
        get_bit_unchecked(&self.buffer, index)
    }

    /// Set the given bit index with given value without bound check
    ///
    /// # Safety
    ///
    /// `index <= self.len()`, otherwise undefined behavior happens
    #[inline]
    pub unsafe fn set_unchecked(&mut self, index: usize, val: bool) {
        set_bit_unchecked(&mut self.buffer, index, val)
    }

    /// Count the number of zeros in the bitmap
    #[inline]
    pub fn count_zeros(&self) -> usize {
        self.num_bits - self.count_ones()
    }

    /// Count the number of oness in the bitmap
    pub fn count_ones(&self) -> usize {
        macro_rules! count_ones {
            ($vals:ident, $count:expr) => {
                *$count += $vals.iter().map(|v| v.count_ones()).sum::<u32>()
            };
        }
        dynamic_func!(count_ones, , (vals: &[BitStore], count: &mut u32), );

        if self.num_bits % BIT_STORE_BITS == 0 {
            let vals = self.buffer.as_slice();
            let mut count = 0;
            count_ones_dynamic(vals, &mut count);
            return count as usize;
        }
        let mut length = 0;
        let mut count = 0;
        if self.num_bits > BIT_STORE_BITS {
            length = self.buffer.len - 1;
            // SAFETY: length is the last element index
            let vals = unsafe { self.buffer.get_slice_unchecked(0, length) };
            count_ones_dynamic(vals, &mut count);
        }
        // Last
        let mask = (1 << (self.num_bits & (BIT_STORE_BITS - 1))) - 1;
        // SAFETY: length is the last element index
        count += (unsafe { self.buffer.get_unchecked(length) } & mask).count_ones();

        count as usize
    }

    /// Resize the [`AlignedVec`] to new_len, all of the visible length will
    /// be uninitialized(Actually, partial of the Vec remain the old value 😊). It is
    /// caller's responsibility to init the visible region.
    #[inline]
    #[must_use]
    pub fn clear_and_resize(&mut self, new_len: usize) -> &mut [u64] {
        self.num_bits = new_len;
        let tmp = self.buffer.clear_and_resize(elts(new_len));
        // Hack😊: The computation may cause last BitStore partially initialized, we initialize it now!
        if let Some(last) = tmp.last_mut() {
            *last = 0;
        }
        tmp
    }

    /// Clear the bitmap, it only set the num_bits to 0 and do not free the
    /// underling buffer
    #[inline]
    pub fn clear(&mut self) {
        self.num_bits = 0;
    }

    /// Appends a single bit into the vec
    ///
    /// Do not use it in the query engine !!!! It is very slow !!!!
    pub(crate) fn push(&mut self, val: bool) {
        let old_len = self.num_bits;
        self.num_bits += 1;
        if old_len == 0 || (self.num_bits / BIT_STORE_BITS) > (old_len / BIT_STORE_BITS) {
            // We need to place the new bit in a new element
            self.buffer.reserve(1);
            self.buffer.len += 1;
        }
        // According to the [book](https://doc.rust-lang.org/nightly/reference/behavior-considered-undefined.html)
        // read integer from uninitialized memory is undefined behavior.
        // [What The Hardware Does" is not What Your Program Does: Uninitialized Memory](https://www.ralfj.de/blog/2019/07/14/uninit.html)
        // [(Why) is using an uninitialized variable undefined behavior?](https://stackoverflow.com/questions/11962457/why-is-using-an-uninitialized-variable-undefined-behavior)

        // We gonna to init the uninitialized memory
        if (old_len % BIT_STORE_BITS) == 0 {
            let bit_store = unsafe { self.buffer.get_unchecked_mut(old_len / BIT_STORE_BITS) };
            *bit_store = 0;
        }

        // SAFETY: we increase the number of bits and reserve the space
        unsafe { set_bit_unchecked(&mut self.buffer, old_len, val) }
    }

    /// # Safety
    ///
    /// - len should equal to trusted_iter's length
    #[inline]
    pub unsafe fn reset(&mut self, len: usize, trusted_len_iterator: impl Iterator<Item = bool>) {
        let uninitialized = self.clear_and_resize(len);
        reset_bitmap_raw(uninitialized.as_mut_ptr(), len, trusted_len_iterator)
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
        write!(f, "Bitmap {{ len: {}, data: ", self.num_bits)?;
        f.debug_list().entries(self.iter()).finish()?;
        write!(f, "}}")
    }
}

impl PartialEq for Bitmap {
    fn eq(&self, other: &Self) -> bool {
        if self.num_bits != other.num_bits {
            return false;
        }
        if self.num_bits % BIT_STORE_BITS == 0 {
            return self.buffer.as_slice() == other.buffer.as_slice();
        }

        let mut length = 0;
        if self.num_bits > BIT_STORE_BITS {
            length = self.buffer.len - 1;
            // SAFETY: length is the last element index
            let lhs = unsafe { self.buffer.get_slice_unchecked(0, length) };
            let rhs = unsafe { self.buffer.get_slice_unchecked(0, length) };
            if lhs != rhs {
                return false;
            }
        }
        // Compare last bit store
        let mask = (1 << (self.num_bits & (BIT_STORE_BITS - 1))) - 1;
        // SAFETY: length is the last element index
        let lhs = unsafe { *self.buffer.get_unchecked(length) } & mask;
        let rhs = unsafe { *other.buffer.get_unchecked(length) } & mask;
        lhs == rhs
    }
}

impl Eq for Bitmap {}

/// Compute the number of [`BitStore`] to store the required number of bits
#[inline]
pub fn elts(num_bits: usize) -> usize {
    (num_bits / BIT_STORE_BITS) + (num_bits % BIT_STORE_BITS != 0) as usize
}

/// Get the given bit value
#[inline]
unsafe fn get_bit_unchecked(buffer: &AlignedVec<BitStore>, index: usize) -> bool {
    *buffer.get_unchecked(index / BIT_STORE_BITS) & (1 << (index % BIT_STORE_BITS)) != 0
}

/// Set the given bit
#[inline]
unsafe fn set_bit_unchecked(buffer: &mut AlignedVec<BitStore>, index: usize, val: bool) {
    let bit_store = buffer.get_unchecked_mut(index / BIT_STORE_BITS);
    let mask = 1 << (index % BIT_STORE_BITS);
    if val {
        *bit_store |= mask;
    } else {
        *bit_store &= !mask;
    }
}

/// Copied from arrow2
///
/// Awesome! Use this function to mutate the bitmap is pretty fast!
///
/// # Safety
///
/// len should equal to trusted_iter's length
#[inline]
pub(crate) unsafe fn reset_bitmap_raw(
    ptr: *mut BitStore,
    len: usize,
    mut trusted_len_iterator: impl Iterator<Item = bool>,
) {
    let num_bit_store = len >> 6;
    for index in 0..num_bit_store {
        let mut mask;
        let mut tmp = BitStore::default();
        for i in 0..8 {
            mask = 1 << (i << 3);
            for _ in 0..8 {
                match trusted_len_iterator.next() {
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
                match trusted_len_iterator.next() {
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
            match trusted_len_iterator.next() {
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

/// Ergonomic helper functions
impl Bitmap {
    /// Construct self from slice of [`BitStore`] and len. Caller should
    /// guarantee the len is valid. Otherwise, panic
    pub fn from_slice_and_len(slice: &[BitStore], len: usize) -> Self {
        if len > slice.len() * BIT_STORE_BITS {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_ones() {
        // Empty
        let bitmap = Bitmap::new();
        assert_eq!(bitmap.count_zeros(), 0);

        // Multiple of 64
        let mut bitmap = Bitmap::from_slice_and_len(&[0x31cd, 0x10], 128);
        assert_eq!(bitmap.count_ones(), 9);

        // > 64
        bitmap.num_bits = 65;
        assert_eq!(bitmap.count_ones(), 8);

        // < 64
        bitmap.num_bits = 13;
        assert_eq!(bitmap.count_ones(), 7);
    }

    #[test]
    fn test_bitmap_eq() {
        // Empty
        let lhs = Bitmap::new();
        let rhs = Bitmap::new();
        assert_eq!(lhs, rhs);

        // Multiple of 64
        let mut lhs = Bitmap::from_slice_and_len(&[0x31cd, 0x10], 128);
        let mut rhs = Bitmap::from_slice_and_len(&[0x31cd, 0x10], 128);
        assert_eq!(lhs, rhs);

        // > 64
        lhs.num_bits = 90;
        rhs.num_bits = 90;
        assert_eq!(lhs, rhs);

        // < 64
        lhs.num_bits = 20;
        rhs.num_bits = 20;
        assert_eq!(lhs, rhs);

        // Not have same length
        lhs.num_bits = 120;
        rhs.num_bits = 127;
        assert_ne!(lhs, rhs);

        // > 64 and the first bit store is equal
        let lhs = Bitmap::from_slice_and_len(&[0x31cd, 0x30], 120);
        let rhs = Bitmap::from_slice_and_len(&[0x31cd, 0x33], 120);
        assert_ne!(lhs, rhs);
    }
}
