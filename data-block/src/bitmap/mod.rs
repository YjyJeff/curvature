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

    // INVARIANT: `count_zeros + count_ones = num_bits`
    /// Number of zeros in the bitmap.
    count_zeros: usize,
    /// Number of ones in the bitmap
    count_ones: usize,
}

impl Bitmap {
    /// Create a new [`Bitmap`]
    #[inline]
    pub fn new() -> Self {
        Self {
            buffer: AlignedVec::new(),
            num_bits: 0,
            count_zeros: 0,
            count_ones: 0,
        }
    }

    /// Create a new [`Bitmap`] with given capacity
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: AlignedVec::with_capacity(elts(capacity)),
            num_bits: 0,
            count_zeros: 0,
            count_ones: 0,
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

    /// Returns true if all of the bits in the bitmap are set
    pub fn all_valid(&self) -> bool {
        self.count_zeros == 0
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
    /// `index < self.len()`, otherwise undefined behavior happens
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> bool {
        unsafe {
            #[cfg(feature = "verify")]
            assert!(index < self.len());

            get_bit_unchecked(&self.buffer, index)
        }
    }

    /// Count the number of zeros in the bitmap
    #[inline]
    pub fn count_zeros(&self) -> usize {
        self.count_zeros
    }

    /// Count the number of ones in the bitmap
    ///
    /// If `self.is_empty()` it will return None. Because it represents all of the
    /// bitmap is valid
    #[inline]
    pub fn count_ones(&self) -> Option<usize> {
        if self.is_empty() {
            None
        } else {
            Some(self.count_ones)
        }
    }

    /// Get the number of ones in the bitmap without check
    ///
    /// # Safety
    ///
    /// Caller should guarantee `!self.is_empty()`
    #[inline]
    pub unsafe fn count_ones_unchecked(&self) -> usize {
        #[cfg(feature = "verify")]
        assert!(!self.is_empty());

        self.count_ones
    }

    /// Get the ratio of ones
    #[inline]
    pub fn ones_ratio(&self) -> f64 {
        if self.num_bits == 0 {
            1.0
        } else {
            self.count_ones as f64 / self.num_bits as f64
        }
    }

    /// Get the bit store with given bit store index without check
    ///
    /// # Safety
    ///
    /// `bit_store_index < self.buffer.len()`
    #[inline]
    unsafe fn valid_bit_store_unchecked(&self, bit_store_index: usize) -> BitStore {
        unsafe {
            #[cfg(feature = "verify")]
            assert!(bit_store_index < self.buffer.len());

            let mut val = *self.buffer.get_unchecked(bit_store_index);

            if ((bit_store_index + 1) * BIT_STORE_BITS) > self.num_bits {
                // Last BitStore, only remain the valid data
                val &= (1 << (self.num_bits & (BIT_STORE_BITS - 1))) - 1;
            }

            val
        }
    }

    /// Get a guard to mutate the bitmap
    #[inline]
    pub fn mutate(&mut self) -> MutateBitmapGuard<'_> {
        MutateBitmapGuard(self)
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
    unsafe {
        #[cfg(feature = "verify")]
        assert!(index / BIT_STORE_BITS < buffer.len());

        *buffer.get_unchecked(index / BIT_STORE_BITS) & (1 << (index % BIT_STORE_BITS)) != 0
    }
}

/// Set the given bit
#[inline]
unsafe fn set_bit_unchecked(buffer: &mut AlignedVec<BitStore>, index: usize, val: bool) {
    unsafe {
        #[cfg(feature = "verify")]
        assert!(index / BIT_STORE_BITS < buffer.len());

        let bit_store = buffer.get_unchecked_mut(index / BIT_STORE_BITS);
        let mask = 1 << (index % BIT_STORE_BITS);
        if val {
            *bit_store |= mask;
        } else {
            *bit_store &= !mask;
        }
    }
}

/// Guard for mutating the [`Bitmap`], it will calibrate the `count_ones` and `count_zeros`
/// when it goes out of scope. It is generated by the [`Bitmap::mutate()`] method
#[derive(Debug)]
#[repr(transparent)]
pub struct MutateBitmapGuard<'a>(&'a mut Bitmap);

/// Calibrate the `count_ones` and `count_zeros`
impl Drop for MutateBitmapGuard<'_> {
    fn drop(&mut self) {
        let count_ones = count_ones(self.0);
        self.0.count_ones = count_ones;
        self.0.count_zeros = self.0.num_bits - count_ones;
    }
}

impl MutateBitmapGuard<'_> {
    /// Set the given bit index with given value without bound check
    ///
    /// # Safety
    ///
    /// `index < self.len()`, otherwise undefined behavior happens
    #[inline]
    pub unsafe fn set_unchecked(&mut self, index: usize, val: bool) {
        unsafe { set_bit_unchecked(&mut self.0.buffer, index, val) }
    }

    /// Resize the [`Bitmap`]
    #[inline]
    #[must_use]
    pub fn clear_and_resize(&mut self, new_len: usize) -> &mut [u64] {
        self.0.num_bits = new_len;
        self.0.buffer.clear_and_resize(elts(new_len))
    }

    /// As raw slice
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u64] {
        self.0.buffer.as_mut_slice()
    }

    /// Clear the bitmap, it only set the num_bits to 0 and do not free the
    /// underling buffer
    #[inline]
    pub fn clear(&mut self) {
        self.0.num_bits = 0;
        self.0.buffer.clear();
    }

    /// Appends a single bit into the vec
    ///
    /// Do not use it in the query engine !!!! It is very slow !!!!
    pub(crate) fn push(&mut self, val: bool) {
        let old_len = self.0.num_bits;
        self.0.num_bits += 1;
        if old_len == 0 || (self.0.num_bits / BIT_STORE_BITS) > (old_len / BIT_STORE_BITS) {
            // We need to place the new bit in a new element
            self.0.buffer.reserve(1);
            self.0.buffer.len += 1;
        }
        // According to the [book](https://doc.rust-lang.org/nightly/reference/behavior-considered-undefined.html)
        // read integer from uninitialized memory is undefined behavior.
        // [What The Hardware Does" is not What Your Program Does: Uninitialized Memory](https://www.ralfj.de/blog/2019/07/14/uninit.html)
        // [(Why) is using an uninitialized variable undefined behavior?](https://stackoverflow.com/questions/11962457/why-is-using-an-uninitialized-variable-undefined-behavior)

        // We gonna to init the uninitialized memory
        if (old_len % BIT_STORE_BITS) == 0 {
            let bit_store = unsafe { self.0.buffer.get_unchecked_mut(old_len / BIT_STORE_BITS) };
            *bit_store = 0;
        }

        // SAFETY: we increase the number of bits and reserve the space
        unsafe { set_bit_unchecked(&mut self.0.buffer, old_len, val) }
    }

    /// Reset the bitmap with given trusted len iterator.
    ///
    /// # Safety
    ///
    /// - len should equal to trusted_iter's length
    #[inline]
    pub unsafe fn reset(&mut self, len: usize, trusted_len_iterator: impl Iterator<Item = bool>) {
        unsafe {
            let uninitialized = self.clear_and_resize(len);
            reset_bitmap_raw(uninitialized.as_mut_ptr(), len, trusted_len_iterator)
        }
    }

    /// Mutate the ones in the bitmap with given function. The function take the index of the
    /// ones in the bitmap and return valid or not for overwriting the ones
    ///
    /// NOTE: For iterating time, it is two times slower than [`reset`](Self::reset()) if
    /// all of the bits are set. It can reduce the computation count by only invoke the func
    /// on the selected index
    pub fn mutate_ones(&mut self, func: impl Fn(usize) -> bool) {
        if self.0.count_ones == 0 {
            return;
        }

        // Manually inline next ones
        for bit_store_index in 0..self.0.buffer.len {
            let base = bit_store_index * BIT_STORE_BITS;
            let mut current = unsafe { self.0.valid_bit_store_unchecked(bit_store_index) };
            let mut new_bit_store = 0;
            while current != 0 {
                let trailing_zeros = current.trailing_zeros();
                let index = base + trailing_zeros as usize;
                if func(index) {
                    new_bit_store |= 1 << trailing_zeros;
                };
                current &= current - 1;
            }

            unsafe {
                *self.0.buffer.get_unchecked_mut(bit_store_index) = new_bit_store;
            }
        }
    }

    /// Set all of the bits as invalid
    #[inline]
    pub fn set_all_invalid(&mut self, len: usize) {
        self.clear_and_resize(len).iter_mut().for_each(|v| *v = 0);
    }

    /// Copy into self
    pub fn copy(&mut self, source: &Bitmap, start: usize, len: usize) {
        if source.is_empty() {
            self.clear();
        } else {
            #[cfg(feature = "verify")]
            assert!(start + len <= source.len());

            let shift = start % BIT_STORE_BITS;
            let new_len = elts(len);
            let dst = self.0.buffer.clear_and_resize(new_len);
            if shift == 0 {
                // Fast path, copy the bit store directly
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        source.buffer.as_ptr().add(start / BIT_STORE_BITS),
                        dst.as_mut_ptr(),
                        new_len,
                    );
                }
            } else if shift % 8 == 0 {
                // Fast path, view the buffer as byte, copy the byte
                // The source address may do not have enough space: `new_len * 8`
                let byte_copy_len = (len / 8) + (len % 8 != 0) as usize;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        (source.buffer.as_ptr() as *const u8).add(start / 8),
                        dst.as_mut_ptr() as *mut u8,
                        byte_copy_len,
                    )
                }
            } else {
                macro_rules! shift_bit_store_slice {
                    ($src:ident, $dst:ident, $bits:ident) => {
                        unsafe {
                            let len = $src.len();
                            (0..len - 1).for_each(|i| {
                                let next = *$src.get_unchecked(i + 1);
                                let v = *$src.get_unchecked(i);
                                *$dst.get_unchecked_mut(i) = (v >> $bits) | (next << (64 - $bits));
                            });
                            *$dst.get_unchecked_mut(len - 1) =
                                (*$src.get_unchecked(len - 1)) >> $bits;
                        }
                    };
                }
                crate::dynamic_func!(
                    shift_bit_store_slice,
                    ,
                    (src: &[u64], dst: &mut [u64], bits:usize),
                );

                let src = unsafe {
                    source
                        .buffer
                        .get_slice_unchecked(start / BIT_STORE_BITS, new_len)
                };

                shift_bit_store_slice_dynamic(src, dst, shift);
            }

            self.0.num_bits = len;
        }
    }
}

fn count_ones(bitmap: &Bitmap) -> usize {
    macro_rules! count_ones {
        ($vals:ident, $count:expr) => {
            *$count += $vals.iter().map(|v| v.count_ones()).sum::<u32>()
        };
    }
    dynamic_func!(count_ones, , (vals: &[BitStore], count: &mut u32), );

    // Cleared
    if bitmap.num_bits == 0 {
        return 0;
    }

    if bitmap.num_bits % BIT_STORE_BITS == 0 {
        let vals = bitmap.buffer.as_slice();
        let mut count = 0;
        count_ones_dynamic(vals, &mut count);
        return count as usize;
    }
    let mut length = 0;
    let mut count = 0;
    if bitmap.num_bits > BIT_STORE_BITS {
        length = bitmap.buffer.len - 1;
        // SAFETY: length is the last element index
        let vals = unsafe { bitmap.buffer.get_slice_unchecked(0, length) };
        count_ones_dynamic(vals, &mut count);
    }
    // Last
    let mask = (1 << (bitmap.num_bits & (BIT_STORE_BITS - 1))) - 1;
    // SAFETY: length is the last element index
    count += (unsafe { bitmap.buffer.get_unchecked(length) } & mask).count_ones();

    count as usize
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
    unsafe {
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
                        None => {
                            #[cfg(feature = "verify")]
                            unreachable!(
                                "Trusted len iterator have same length with len. Not enough elements for BitStore"
                            );

                            #[cfg(not(feature = "verify"))]
                            std::hint::unreachable_unchecked()
                        }
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
                        None => {
                            #[cfg(feature = "verify")]
                            unreachable!(
                                "Trusted len iterator have same length with len. Not enough elements for Bytes"
                            );

                            #[cfg(not(feature = "verify"))]
                            std::hint::unreachable_unchecked()
                        }
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
                    None => {
                        #[cfg(feature = "verify")]
                        unreachable!(
                            "Trusted len iterator have same length with len. Not enough elements for remainder bits"
                        );

                        #[cfg(not(feature = "verify"))]
                        std::hint::unreachable_unchecked()
                    }
                };
            }

            *ptr.add(num_bit_store) = tmp;
        }
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

        // SAFETY: the len is checked
        unsafe { Self::from_slice_and_len_unchecked(slice, len) }
    }

    /// Construct self from slice of [`BitStore`] and len. Caller should
    /// guarantee the len is valid. Otherwise, undefined behavior happens
    ///
    /// # Safety
    ///
    /// The len is valid: `slice.len() * BIT_STORE_BITS <= len`
    #[inline]
    pub unsafe fn from_slice_and_len_unchecked(slice: &[BitStore], len: usize) -> Self {
        unsafe { Self::from_aligned_vec_and_len_unchecked(AlignedVec::from_slice(slice), len) }
    }

    /// Construct self from [`AlignedVec`] and len. Caller should
    /// guarantee the len is valid. Otherwise, undefined behavior happens
    ///
    /// # Safety
    ///
    /// The len is valid: `buffer.len() * BIT_STORE_BITS >= len`
    pub unsafe fn from_aligned_vec_and_len_unchecked(
        buffer: AlignedVec<BitStore>,
        len: usize,
    ) -> Self {
        #[cfg(feature = "verify")]
        assert!(buffer.len() * BIT_STORE_BITS >= len);

        let mut bitmap = Self {
            buffer,
            num_bits: len,
            count_ones: 0,
            count_zeros: 0,
        };

        let count_ones = count_ones(&bitmap);
        bitmap.count_ones = count_ones;
        bitmap.count_zeros = bitmap.num_bits - count_ones;
        bitmap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl MutateBitmapGuard<'_> {
        fn set_len(&mut self, len: usize) {
            self.0.num_bits = len;
        }
    }

    #[test]
    fn test_count_ones() {
        // Empty
        let bitmap = Bitmap::new();
        assert_eq!(bitmap.count_zeros(), 0);
        assert!(bitmap.count_ones().is_none());

        // Multiple of 64
        let mut bitmap = Bitmap::from_slice_and_len(&[0x31cd, 0x10], 128);
        assert_eq!(bitmap.count_ones(), Some(9));

        // > 64
        {
            bitmap.mutate().set_len(65);
        }
        assert_eq!(bitmap.count_ones(), Some(8));

        // < 64
        {
            bitmap.mutate().set_len(13);
        }
        assert_eq!(bitmap.count_ones(), Some(7));
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
        {
            lhs.mutate().set_len(90);
            rhs.mutate().set_len(90);
        }
        assert_eq!(lhs, rhs);

        // < 64
        {
            lhs.mutate().set_len(20);
            rhs.mutate().set_len(20);
        }
        assert_eq!(lhs, rhs);

        // Not have same length
        {
            lhs.mutate().set_len(120);
            rhs.mutate().set_len(127);
        }
        assert_ne!(lhs, rhs);

        // > 64 and the first bit store is equal
        let lhs = Bitmap::from_slice_and_len(&[0x31cd, 0x30], 120);
        let rhs = Bitmap::from_slice_and_len(&[0x31cd, 0x33], 120);
        assert_ne!(lhs, rhs);
    }

    #[test]
    fn test_mutate_bitmap() {
        let mut bitmap = Bitmap::from_slice_and_len(&[0x31cd, 0x10], 128);
        assert_eq!(bitmap.count_ones(), Some(9));
        assert_eq!(bitmap.count_zeros(), 119);

        bitmap
            .mutate()
            .clear_and_resize(150)
            .iter_mut()
            .for_each(|v| *v = 0);

        assert_eq!(bitmap.count_ones, 0);
        assert_eq!(bitmap.count_zeros, 150);
    }

    #[test]
    fn test_copy() {
        let mut bitmap = Bitmap::from_slice_and_len(&[128], 10);
        let source = Bitmap::default();
        bitmap.mutate().copy(&source, 0, 8);
        assert!(bitmap.is_empty());

        // Fast path, u64 boundary
        let source = Bitmap::from_slice_and_len(&[256, 65544], 127);
        bitmap.mutate().copy(&source, 64, 60);
        assert_eq!(bitmap.buffer.get(0).unwrap(), &65544);

        // Fast path, u8 boundary
        bitmap.mutate().copy(&source, 8, 100);
        // byte: [1, 0, 0, 0, 0, 0, 0, 8]
        assert_eq!(bitmap.buffer.get(0).unwrap(), &(1 + (1 << 59)));

        bitmap.mutate().copy(&source, 7, 110);
        assert_eq!(bitmap.buffer.get(0).unwrap(), &(2 + (1 << 60)));
        assert_eq!(bitmap.buffer.get(1).unwrap(), &(1 << 9));
    }
}
