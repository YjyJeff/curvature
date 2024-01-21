//! Iterator of the Array

use super::Array;
use crate::bitmap::{Bitmap, BitmapIter};
use crate::scalar::Scalar;
use std::fmt::Debug;
use std::iter::Zip;

/// Default implementation for the iterator of the values in the array
#[derive(Debug)]
pub struct ArrayValuesIter<'a, A: Array> {
    array: &'a A,
    len: usize,
    current: usize,
}

impl<'a, A: Array> ArrayValuesIter<'a, A> {
    #[inline]
    pub(super) fn new(array: &'a A) -> Self {
        Self {
            array,
            len: array.len(),
            current: 0,
        }
    }

    #[inline]
    pub(super) fn new_with_current_and_len(array: &'a A, current: usize, len: usize) -> Self {
        Self {
            array,
            len,
            current,
        }
    }
}

impl<'a, A: Array> Iterator for ArrayValuesIter<'a, A> {
    type Item = <A::ScalarType as Scalar>::RefType<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current == self.len {
            return None;
        }
        let old = self.current;
        self.current += 1;
        // SAFETY: current < len, the index is valid
        Some(unsafe { self.array.get_value_unchecked(old) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len - self.current, Some(self.len - self.current))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let new_index = self.current + n;
        if new_index >= self.len {
            self.current = self.len;
            None
        } else {
            self.current = new_index;
            self.next()
        }
    }
}

impl<'a, A: Array> ExactSizeIterator for ArrayValuesIter<'a, A> {}

/// Iterator of the [`Array`]. Especially useful for [`Debug`]
pub enum ArrayIter<'a, A: Array> {
    /// Iterator of values, all of the
    Values(A::ValuesIter<'a>),
    /// Iterator of values and validity bitmap
    ValuesAndValidity(Zip<A::ValuesIter<'a>, BitmapIter<'a>>),
}

impl<'a, A: Array> ArrayIter<'a, A> {
    /// Create a new [`ArrayIter`]
    pub fn new(values_iter: A::ValuesIter<'a>, validity: &'a Bitmap) -> Self {
        if validity.is_empty() {
            Self::Values(values_iter)
        } else {
            Self::ValuesAndValidity(values_iter.zip(validity.iter()))
        }
    }

    /// Create a new [`ArrayIter`] with values iterator
    pub fn new_values(values_iter: A::ValuesIter<'a>) -> Self {
        Self::Values(values_iter)
    }

    /// Create a new [`ArrayIter`] with values iterator and bitmap iterator
    pub fn new_values_and_validity(
        values_iter: A::ValuesIter<'a>,
        bitmap_iter: BitmapIter<'a>,
    ) -> Self {
        Self::ValuesAndValidity(values_iter.zip(bitmap_iter))
    }
}

impl<'a, A: Array> Debug for ArrayIter<'a, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ArrayIter")
    }
}

impl<'a, A: Array> Iterator for ArrayIter<'a, A> {
    type Item = Option<<A::ScalarType as Scalar>::RefType<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Values(values) => values.next().map(Some),
            Self::ValuesAndValidity(iter) => {
                let (value, not_null) = iter.next()?;
                if not_null {
                    Some(Some(value))
                } else {
                    Some(None)
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Values(values) => values.size_hint(),
            Self::ValuesAndValidity(iter) => iter.size_hint(),
        }
    }
}

impl<'a, A: Array> ExactSizeIterator for ArrayIter<'a, A> {}
