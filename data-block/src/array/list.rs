//! Elements in the array are list

use snafu::ensure;

use crate::aligned_vec::AlignedVec;
use crate::array::InvalidLogicalTypeSnafu;
use crate::bitmap::Bitmap;
use crate::element::list::{ListElement, ListElementRef};
use crate::element::Element;
use crate::private::Sealed;
use crate::types::{LogicalType, PhysicalType};
use std::fmt::Debug;

use super::iter::ArrayValuesIter;
use super::ping_pong::PingPongPtr;
use super::{Array, ArrayImpl, MutateArrayExt, Result};

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
    elements: PingPongPtr<ArrayImpl>,
}

impl Debug for ListArray {
    /// FIXME: implement debug
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ListArray{{ logical_type: {:?}, len : {}, data: [] }}",
            self.logical_type,
            self.len()
        )
    }
}

impl ListArray {
    /// Create a new empty [`ListArray`]
    #[inline]
    pub fn new(logical_type: LogicalType) -> Result<Self> {
        Self::with_capacity(logical_type, 0)
    }

    /// Create a new empty [`ListArray`] without check
    ///
    /// # Safety
    ///
    /// physical type of the logical type should be `List`
    #[inline]
    pub unsafe fn new_unchecked(logical_type: LogicalType) -> Self {
        Self::with_capacity_unchecked(logical_type, 0)
    }

    /// Create a new [`ListArray`] with given capacity
    #[inline]
    pub fn with_capacity(logical_type: LogicalType, capacity: usize) -> Result<Self> {
        ensure!(
            logical_type.physical_type() == PhysicalType::List,
            InvalidLogicalTypeSnafu {
                array_name: "ListArray".to_string(),
                array_physical_type: PhysicalType::List,
                logical_type
            }
        );

        // SAFETY: we check the physical type above
        unsafe { Ok(Self::with_capacity_unchecked(logical_type, capacity)) }
    }

    /// Create a new [`ListArray`] with given capacity without check
    ///
    /// # Safety
    ///
    /// physical type of the logical type should be `List`
    #[inline]
    pub unsafe fn with_capacity_unchecked(logical_type: LogicalType, capacity: usize) -> Self {
        let child_type = logical_type
            .child(0)
            .expect("List must have one child type")
            .clone();

        Self {
            logical_type,
            validity: PingPongPtr::default(),
            offsets: PingPongPtr::new(AlignedVec::with_capacity(capacity)),
            lengths: PingPongPtr::new(AlignedVec::with_capacity(capacity)),
            elements: PingPongPtr::with_constructor(|| ArrayImpl::new(child_type.clone())),
        }
    }
}

impl Sealed for ListArray {}

impl Array for ListArray {
    type Element = ListElement;

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
    fn validity_mut(&mut self) -> &mut PingPongPtr<Bitmap> {
        &mut self.validity
    }

    #[inline]
    unsafe fn get_value_unchecked(
        &self,
        index: usize,
    ) -> <Self::Element as Element>::ElementRef<'_> {
        ListElementRef::new(
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

impl MutateArrayExt for ListArray {
    /// TBD: Could we reference the elements directly?
    #[inline]
    fn reference(&mut self, other: &Self) {
        self.offsets.reference(&other.offsets);
        self.lengths.reference(&other.lengths);
        self.elements.reference(&other.elements);
        self.validity.reference(&other.validity);
    }
}
