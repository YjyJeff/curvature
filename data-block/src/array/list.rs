//! Elements in the array are list
//!
//! ListArray breaks assumptions
//!
//! - Array can not be partial slice of another `Array`: ith element of the ListArray is
//! an [`Array`]! We have to create a new array to retrieve the ith element, therefore
//! we have to do slicing!

use crate::aligned_vec::AlignedVec;
use crate::bitmap::Bitmap;
use crate::private::Sealed;
use crate::scalar::list::{ListScalar, ListScalarRef};
use crate::scalar::Scalar;
use crate::types::LogicalType;
use std::fmt::Debug;
use std::rc::Rc;

use super::iter::ArrayValuesIter;
use super::ping_pong::PingPongPtr;
use super::{Array, ArrayImpl};

/// [Array of lists](https://facebookincubator.github.io/velox/develop/vectors.html#flat-vectors-complex-types)
pub struct ListArray {
    /// Logical type
    logical_type: LogicalType,
    /// validity
    validity: PingPongPtr<Bitmap>,
    /// offsets[i] and offsets[i+1] represents the start and end address of the ith
    /// element in the child array
    offsets: PingPongPtr<AlignedVec<u32>>,
    /// length of each list in the array
    lengths: PingPongPtr<AlignedVec<u32>>,
    /// array that contains its elements
    elements: Rc<ArrayImpl>,
}

impl Debug for ListArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ListArray{{ len : {}, data: [] }}", self.len())
    }
}

impl Sealed for ListArray {}

impl Array for ListArray {
    type ScalarType = ListScalar;

    type ValuesIter<'a> = ArrayValuesIter<'a, Self>;

    #[inline]
    fn values_iter(&self) -> Self::ValuesIter<'_> {
        ArrayValuesIter::new(self)
    }

    #[inline]
    fn values_slice_iter(&self, offset: usize, length: usize) -> Self::ValuesIter<'_> {
        assert!(offset + length <= self.len());
        ArrayValuesIter::new_with_current_and_len(self, offset, length)
    }

    #[inline]
    fn len(&self) -> usize {
        self.offsets.len
    }

    #[inline]
    fn validity(&self) -> &Bitmap {
        &self.validity
    }

    #[inline]
    unsafe fn get_value_unchecked(
        &self,
        index: usize,
    ) -> <Self::ScalarType as Scalar>::RefType<'_> {
        ListScalarRef::new(
            &self.elements,
            *self.offsets.get_unchecked(index),
            *self.lengths.get_unchecked(index),
        )
    }

    #[inline]
    fn logical_type(&self) -> &LogicalType {
        &self.logical_type
    }
}
