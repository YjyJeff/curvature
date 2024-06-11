//! Implementation of the Cache line aligned [`Vec`]
//!
//! Heavily adapted from [Arrow](https://github.com/apache/arrow-rs)

pub mod alignment;
use alignment::ALIGNMENT;

use std::alloc::{alloc, dealloc, handle_alloc_error, realloc, Layout};
use std::fmt::{Debug, Display};
use std::mem::size_of;
use std::ptr::{copy_nonoverlapping, NonNull};

use crate::element::interval::DayTime;
use crate::private::Sealed;
use crate::utils::roundup_to_multiple_of_pow_of_two_base;

/// Size of the cache line in bytes
///
/// FIXME: From the simd view, 128 is better. If we change it to 128, manually simd
/// should also changed
pub const CACHE_LINE_SIZE: usize = 64;

/// Trait for types that can be allocated on the [`AlignedVec`]. This trait is
/// sealed to avoid other types implement it
pub trait AllocType:
    Sealed + Clone + Sized + Debug + Default + Display + 'static + Send + Sync
{
}

macro_rules! impl_alloc_types {
    ($($ty:ty),*) => {
        $(
            impl Sealed for $ty {}

            impl AllocType for $ty {}
        )*
    };
}

impl_alloc_types!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, f32, f64, DayTime);

/// [`AlignedVec`] is a continuous memory region that allocated from memory
/// allocator. The memory is **cache line aligned** and its **capacity in bytes**
/// is multiple of **cache line size**
///
/// [`AlignedVec`] can accelerate the performance a lot! Firstly, the memory
/// is cache line aligned, we can use aligned load instructions and **tumbling window**
/// to load the continuous data into SIMD registers. Secondly, the capacity is multiple
/// of cache line size, SIMD instructions can load uninitialized data to perform
/// computation and do not need to check the memory load is valid or not.
///
/// Note that you can not remove `#[repr(C)]` !!!! According to the [Rustonomicon],
/// `transmute` need this attribute
///
/// [Rustonomicon]: https://doc.rust-lang.org/nomicon/transmutes.html
#[repr(C)]
pub struct AlignedVec<T: AllocType> {
    /// Pointer to the start of the memory region
    pub(crate) ptr: NonNull<T>,
    /// Number of elements in the AlignedVec
    pub(crate) len: usize,
    /// Memory layout of the region
    capacity_in_bytes: usize,
}

impl<T: AllocType> AlignedVec<T> {
    /// Create a new [`AlignedVec`]
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            capacity_in_bytes: 0,
        }
    }

    /// Create a new [`AlignedVec`] with given capacity
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        let mut capacity_in_bytes = capacity * std::mem::size_of::<T>();
        // round the cap up to multiple of [`CACHE_LINE_SIZE`]
        capacity_in_bytes =
            roundup_to_multiple_of_pow_of_two_base(capacity_in_bytes, CACHE_LINE_SIZE);
        // SAFETY: [`ALIGNMENT`] is guaranteed to be power of two
        unsafe {
            if capacity_in_bytes == 0 {
                Self::new()
            } else {
                let layout = Layout::from_size_align_unchecked(capacity_in_bytes, ALIGNMENT);
                let ptr = alloc(layout);
                Self {
                    ptr: NonNull::new(ptr as _).unwrap_or_else(|| handle_alloc_error(layout)),
                    len: 0,
                    capacity_in_bytes,
                }
            }
        }
    }

    #[inline]
    fn layout(&self) -> Layout {
        unsafe { Layout::from_size_align_unchecked(self.capacity_in_bytes, ALIGNMENT) }
    }

    /// Returns the raw pointer to the aligned memory region
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr() as _
    }

    /// Get the number of elements in the [`AlignedVec`], also referred to its 'length'
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the [`AlignedVec`] contains no elements
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// View the entire vector as a slice
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: Self guarantees memory region from self.ptr to self.ptr+self.len is always valid
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// View the entire vector as a mutable slice
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: Self guarantees memory region from self.ptr to self.ptr+self.len is always valid
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Resize the [`AlignedVec`] to new_len, all of the visible length will
    /// be uninitialized(Actually, partial of the Vec remain the old value 😊). It is
    /// caller's responsibility to init the visible region.
    #[inline]
    #[must_use = "mutable slice is uninitialized, caller should init it manually"]
    pub fn clear_and_resize(&mut self, new_len: usize) -> &mut [T] {
        if new_len > self.len {
            let new_len_in_bytes = new_len * size_of::<T>();
            if new_len_in_bytes > self.capacity_in_bytes {
                self.realloc(new_len_in_bytes);
            }
        }
        self.len = new_len;

        self.as_mut_slice()
    }

    /// Clear the [`AlignedVec`], removing all elements.
    ///
    /// Note that this method has no effect on the allocated capacity
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0
    }

    /// Reserves capacity for at least additional more elements to be inserted
    #[inline]
    pub(crate) fn reserve(&mut self, additional: usize) {
        let new_len_in_bytes = (self.len + additional) * size_of::<T>();

        if new_len_in_bytes > self.capacity_in_bytes {
            // SAFETY:
            // 1. self.ptr and self.layout is pre-allocated by the allocator
            // 2. [`ALIGNMENT`] is guaranteed to be power of two
            self.realloc(new_len_in_bytes);
        }
    }

    #[cold]
    fn realloc(&mut self, new_cap_in_bytes: usize) {
        // SAFETY:
        // 1. self.ptr and self.layout is pre-allocated by the allocator
        // 2. [`ALIGNMENT`] is guaranteed to be power of two
        unsafe {
            let new_cap_in_bytes =
                roundup_to_multiple_of_pow_of_two_base(new_cap_in_bytes, CACHE_LINE_SIZE);
            // The new memory region is at least two times larger than the old region
            let new_cap_in_bytes = std::cmp::max(new_cap_in_bytes, self.capacity_in_bytes * 2);
            let new_layout = Layout::from_size_align_unchecked(new_cap_in_bytes, ALIGNMENT);
            let ptr = if self.capacity_in_bytes == 0 {
                // ptr is not allocated, according to [`Safety`](https://doc.rust-lang.org/std/alloc/trait.GlobalAlloc.html#safety-4)
                // section, we should use alloc instead of realloc
                alloc(new_layout)
            } else {
                realloc(self.ptr.as_ptr() as _, self.layout(), new_cap_in_bytes)
            };

            self.ptr = NonNull::new(ptr as _).unwrap_or_else(|| handle_alloc_error(new_layout));
            self.capacity_in_bytes = new_cap_in_bytes;
        }
    }

    /// Returns a T at the given index without bound check
    ///
    /// # Safety
    /// Caller should guarantee `index < self.len()`, otherwise, [undefined behavior] happens
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        &*self.ptr.as_ptr().add(index)
    }

    /// Returns a T at the given index without bound check
    ///
    /// # Safety
    /// Caller should guarantee `index < self.len()`, otherwise, [undefined behavior] happens
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        &mut *self.ptr.as_ptr().add(index)
    }

    /// Returns a `&[T]` start from the given index with given length without bound check
    ///
    /// # Safety
    /// Caller should guarantee `index + len < self.len()`, otherwise, [undefined behavior] happens
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    pub unsafe fn get_slice_unchecked(&self, index: usize, len: usize) -> &[T] {
        std::slice::from_raw_parts(self.ptr.as_ptr().add(index), len)
    }
}

unsafe impl<T: AllocType> Send for AlignedVec<T> {}
unsafe impl<T: AllocType> Sync for AlignedVec<T> {}

impl<T: AllocType> Drop for AlignedVec<T> {
    #[inline]
    fn drop(&mut self) {
        if self.capacity_in_bytes != 0 {
            // Not dangling pointer
            unsafe { dealloc(self.ptr.as_ptr() as _, self.layout()) };
        }
    }
}

impl<T: AllocType> Default for AlignedVec<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: AllocType> Clone for AlignedVec<T> {
    fn clone(&self) -> Self {
        if self.capacity_in_bytes == 0 {
            // Not allocated
            Self::new()
        } else {
            // SAFETY: layout is correct and the new allocated memory region never overlap with self
            unsafe {
                let ptr = NonNull::new(alloc(self.layout()) as _)
                    .unwrap_or_else(|| handle_alloc_error(self.layout()));
                copy_nonoverlapping(self.ptr.as_ptr(), ptr.as_ptr(), self.len);

                Self {
                    ptr,
                    len: self.len,
                    capacity_in_bytes: self.capacity_in_bytes,
                }
            }
        }
    }
}

impl<T: AllocType> Debug for AlignedVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AlignedVec {{ ptr: {:?}, len: {}, data: ",
            self.ptr, self.len
        )?;

        f.debug_list().entries(self.as_slice().iter()).finish()?;

        write!(f, " }}")
    }
}

/// Ergonomic helper functions
impl<T: AllocType> AlignedVec<T> {
    /// Construct Self from slice
    pub fn from_slice(slice: &[T]) -> Self {
        let mut new = Self::with_capacity(slice.len());
        new.len = slice.len();
        // SAFETY: with_capacity will allocate enough space
        unsafe { std::ptr::copy_nonoverlapping(slice.as_ptr(), new.ptr.as_ptr(), slice.len()) };
        new
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::AsPrimitive;

    fn enumerate_assign<'a, V, T>(vals: T)
    where
        T: IntoIterator<Item = &'a mut V>,
        V: 'static + Copy,
        usize: AsPrimitive<V>,
    {
        vals.into_iter().enumerate().for_each(|(i, val)| {
            *val = i.as_();
        })
    }

    #[test]
    fn test_aligned_vec() {
        // Big enough
        let mut aligned_vec = AlignedVec::from_slice(&[0, 1, 2, 3]);

        assert_eq!(aligned_vec.capacity_in_bytes, CACHE_LINE_SIZE);
        assert_eq!(aligned_vec.as_slice(), [0, 1, 2, 3]);

        // Need resize
        let new_len = CACHE_LINE_SIZE / std::mem::size_of::<i32>() + 1;
        enumerate_assign(aligned_vec.clear_and_resize(new_len));

        assert_eq!(aligned_vec.capacity_in_bytes, CACHE_LINE_SIZE * 2);
        assert_eq!(
            aligned_vec.as_slice(),
            (0..new_len as i32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_clone_aligned_vec() {
        // does not allocate memory
        let mut empty_vec = AlignedVec::<i32>::new();
        let new_empty_vec = empty_vec.clone();
        assert!(new_empty_vec.is_empty());

        enumerate_assign(empty_vec.clear_and_resize(10));

        let cloned_vec = empty_vec.clone();
        assert_eq!(cloned_vec.as_slice(), (0..10).collect::<Vec<_>>());
    }
}
