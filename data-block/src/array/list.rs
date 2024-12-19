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
use super::swar::SwarPtr;
use super::{Array, ArrayImpl, Result};

/// [Array of lists](https://facebookincubator.github.io/velox/develop/vectors.html#flat-vectors-complex-types)
pub struct ListArray {
    /// Logical type
    logical_type: LogicalType,
    /// validity
    pub(crate) validity: SwarPtr<Bitmap>,
    /// offsets[i] and offsets[i+1] represents the start and end address of the ith
    /// element in the child array
    offsets: SwarPtr<AlignedVec<u32>>,
    /// length of each list in the array
    lengths: SwarPtr<AlignedVec<u32>>,
    /// array that contains its elements
    elements: SwarPtr<ArrayImpl>,
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
        #[cfg(feature = "verify")]
        assert_eq!(logical_type.physical_type(), PhysicalType::List);

        let child_type = logical_type
            .child(0)
            .expect("List must have one child type")
            .clone();

        Self {
            logical_type,
            validity: SwarPtr::default(),
            offsets: SwarPtr::new(AlignedVec::with_capacity(capacity)),
            lengths: SwarPtr::new(AlignedVec::with_capacity(capacity)),
            elements: SwarPtr::with_constructor(|| ArrayImpl::new(child_type.clone())),
        }
    }
}

impl Sealed for ListArray {}

impl Array for ListArray {
    const PHYSICAL_TYPE: PhysicalType = PhysicalType::List;
    type Element = ListElement;

    type ValuesIter<'a> = ArrayValuesIter<'a, Self>;

    #[inline]
    fn values_iter(&self) -> Self::ValuesIter<'_> {
        ArrayValuesIter::new(self)
    }

    #[inline]
    fn values_slice_iter(&self, offset: usize, length: usize) -> Self::ValuesIter<'_> {
        #[cfg(feature = "verify")]
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
    unsafe fn validity_mut(&mut self) -> &mut Bitmap {
        self.validity.as_mut()
    }

    #[inline]
    unsafe fn get_value_unchecked(
        &self,
        index: usize,
    ) -> <Self::Element as Element>::ElementRef<'_> {
        #[cfg(feature = "verify")]
        assert!(index < self.len());

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

    // Mutate array

    /// FIXME: Could we reference the elements directly?
    #[inline]
    fn reference(&mut self, other: &Self) {
        self.offsets.reference(&other.offsets);
        self.lengths.reference(&other.lengths);
        self.elements.reference(&other.elements);
        self.validity.reference(&other.validity);
    }

    unsafe fn set_all_invalid(&mut self, _len: usize) {
        todo!()
    }

    unsafe fn replace_with_trusted_len_values_iterator<I>(
        &mut self,
        _len: usize,
        _trusted_len_iterator: I,
    ) where
        I: Iterator<Item = Self::Element>,
    {
        todo!()
    }

    unsafe fn replace_with_trusted_len_iterator<I>(&mut self, _len: usize, _trusted_len_iterator: I)
    where
        I: Iterator<Item = Option<Self::Element>>,
    {
        todo!()
    }

    unsafe fn replace_with_trusted_len_values_ref_iterator<'a, I>(
        &mut self,
        _len: usize,
        _trusted_len_iterator: I,
    ) where
        I: Iterator<Item = <Self::Element as Element>::ElementRef<'a>> + 'a,
    {
        todo!()
    }

    unsafe fn replace_with_trusted_len_ref_iterator<'a, I>(
        &mut self,
        _len: usize,
        _trusted_len_iterator: I,
    ) where
        I: Iterator<Item = Option<<Self::Element as Element>::ElementRef<'a>>> + 'a,
    {
        todo!()
    }

    unsafe fn clear(&mut self) {
        todo!()
    }

    unsafe fn copy(&mut self, source: &Self, start: usize, len: usize) {
        #[cfg(feature = "verify")]
        assert!(start + len <= source.len());

        todo!()
    }
}
