//! Iterator of the Array

use super::Array;
use crate::bitmap::BitValIter;
use crate::scalar::Scalar;
use std::fmt::Debug;

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
pub struct ArrayIter<'a, A: Array> {
    pub(super) values_iter: A::ValuesIter<'a>,
    pub(super) validity: Option<BitValIter<'a>>,
}

impl<'a, A: Array> Debug for ArrayIter<'a, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ArrayIter")
    }
}

impl<'a, A: Array> Iterator for ArrayIter<'a, A> {
    type Item = Option<<A::ScalarType as Scalar>::RefType<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        let val = self.values_iter.next();
        val.map(|val| {
            let not_null = self.validity.as_mut().map(|v| {
                v.next()
                    .expect("Validity must have the same length with ValuesIter")
            });
            match not_null {
                Some(not_null) => {
                    if not_null {
                        Some(val)
                    } else {
                        None
                    }
                }
                None => Some(val),
            }
        })
    }
}
