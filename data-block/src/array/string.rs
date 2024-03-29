//! Array of String. Array of String is different with arrow, we
//! implement it with the format specified by the [`Umbra`].
//! [`Velox`] also use this format.
//!
//! Note that this format is pretty slow if strings has common
//! prefix. For example logs may contain path as its common prefix.
//!
//! # Pointer VS Indices
//! There are some [`discussion`] here to replace the raw pointer with indices.
//! The pointer is more flexible, we do not need to copy the data into the `bytes`
//! buffer. But programmer should take care of the lifetime: reallocation, serialization
//! could invalid the address. Index is more safe, but you need to copy the data
//! into the `bytes` buffer.
//!
//! [`Umbra`]: https://db.in.tum.de/~freitag/papers/p29-neumann-cidr20.pdf
//! [`Velox`]: https://vldb.org/pvldb/vol15/p3372-pedreira.pdf
//! [`discussion`]: https://github.com/facebookincubator/velox/discussions/4362

use snafu::ensure;

use crate::aligned_vec::AlignedVec;
use crate::bitmap::Bitmap;
use crate::element::string::{StringElement, StringView, INLINE_LEN};
use crate::private::Sealed;
use crate::types::{LogicalType, PhysicalType};
use std::fmt::Debug;

use super::iter::ArrayValuesIter;
use super::swar::SwarPtr;
use super::{Array, InvalidLogicalTypeSnafu, MutateArrayExt, Result, ScalarArray};

/// [`Array`] of string
pub struct StringArray {
    logical_type: LogicalType,
    /// A continuous byte array that stores the content of the
    /// indirection String
    _bytes: SwarPtr<AlignedVec<u8>>,
    /// Views of the String. Use [`StringView`] here is safe! The pointer in the views
    /// points to the [`Self::_bytes`], therefore the 'static lifetime is fake !!! It
    /// is a hack to allow the self-referential
    pub(crate) views: SwarPtr<AlignedVec<StringView<'static>>>,
    pub(crate) validity: SwarPtr<Bitmap>,
    // An auxiliary vector to record the offset to the bytes. Reallocation
    // invalid the address of the bytes, therefore, if we store the pointer
    // to the bytes directly, we will get invalid pointer
    _calibrate_offsets: Vec<usize>,
}

impl StringArray {
    /// Create a new empty [`StringArray`]
    #[inline]
    pub fn new(logical_type: LogicalType) -> Result<Self> {
        Self::with_capacity(logical_type, 0)
    }

    /// Create a new empty [`StringArray`] without check
    ///
    /// # Safety
    ///
    /// physical type of the logical type should be `String`
    #[inline]
    pub unsafe fn new_unchecked(logical_type: LogicalType) -> Self {
        Self::with_capacity_unchecked(logical_type, 0)
    }

    /// Create a new [`StringArray`] with given capacity
    #[inline]
    pub fn with_capacity(logical_type: LogicalType, capacity: usize) -> Result<Self> {
        ensure!(
            logical_type.physical_type() == PhysicalType::String,
            InvalidLogicalTypeSnafu {
                array_name: "StringArray".to_string(),
                array_physical_type: PhysicalType::String,
                logical_type,
            }
        );
        // SAFETY: we check the physical type above
        unsafe { Ok(Self::with_capacity_unchecked(logical_type, capacity)) }
    }

    /// Create a new [`StringArray`] with given capacity without check
    ///
    /// # Safety
    ///
    /// physical type of the logical type should be `String`
    #[inline]
    pub unsafe fn with_capacity_unchecked(logical_type: LogicalType, capacity: usize) -> Self {
        Self {
            logical_type,
            _bytes: SwarPtr::default(),
            views: SwarPtr::new(AlignedVec::with_capacity(capacity)),
            validity: SwarPtr::default(),
            _calibrate_offsets: Vec::with_capacity(capacity),
        }
    }

    /// Construct [`StringArray`] from iterator of the str
    #[must_use]
    pub fn from_values_iter<V: AsRef<str>, I: IntoIterator<Item = V>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut bytes = AlignedVec::<u8>::new();
        let mut views = AlignedVec::<StringView<'static>>::with_capacity(lower);
        // An auxiliary vector to record the offset to the bytes. Reallocation
        // invalid the address of the bytes, therefore, if we store the pointer
        // to the bytes directly, we will get invalid pointer
        let mut calibrate_offsets = Vec::<usize>::with_capacity(lower);

        for val in iter {
            views.reserve(1);
            push_str(val.as_ref(), &mut bytes, &mut views, &mut calibrate_offsets);
            views.len += 1;
        }

        calibrate_pointers(bytes.ptr.as_ptr(), views.as_mut_slice(), &calibrate_offsets);

        Self {
            logical_type: LogicalType::VarChar,
            _bytes: SwarPtr::new(bytes),
            views: SwarPtr::new(views),
            validity: SwarPtr::default(),
            _calibrate_offsets: calibrate_offsets,
        }
    }
}

impl Debug for StringArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "StringArray {{ logical_type: {:?}, len: {}, data: ",
            self.logical_type,
            self.len()
        )?;
        f.debug_list().entries(self.iter()).finish()?;
        writeln!(f, "}}")
    }
}

impl Sealed for StringArray {}

impl Array for StringArray {
    type Element = StringElement;

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
        self.views.len()
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
    ) -> <Self::Element as crate::element::Element>::ElementRef<'_> {
        self.views.get_unchecked(index).shorten()
    }

    #[inline]
    fn logical_type(&self) -> &LogicalType {
        &self.logical_type
    }
}

impl MutateArrayExt for StringArray {
    #[inline]
    fn reference(&mut self, other: &Self) {
        self._bytes.reference(&other._bytes);
        self.views.reference(&other.views);
        self.validity.reference(&other.validity);
    }
}

fn push_str(
    val: &str,
    bytes: &mut AlignedVec<u8>,
    views: &mut AlignedVec<StringView<'static>>,
    calibrate_offsets: &mut Vec<usize>,
) {
    let len = val.len();
    let view = if len <= INLINE_LEN {
        calibrate_offsets.push(usize::MAX);
        StringView::new_inline(val)
    } else {
        calibrate_offsets.push(bytes.len);
        bytes.reserve(len);
        unsafe {
            // Copy the string content to bytes
            let dst = bytes.ptr.as_ptr().add(bytes.len);
            std::ptr::copy_nonoverlapping(val.as_ptr(), dst, len);
            bytes.len += len;
            StringView::new_indirect(dst, len as _)
        }
    };
    unsafe {
        *views.ptr.as_ptr().add(views.len) = view;
    }
}

#[inline]
unsafe fn assign_string_element(
    element: StringElement,
    bytes: &mut AlignedVec<u8>,
    views: &mut [StringView<'static>],
    index: usize,
    calibrate_offsets: &mut Vec<usize>,
) {
    *views.get_unchecked_mut(index) = element.view;

    if let Some(data) = element._data {
        let len = data.len();
        calibrate_offsets.push(bytes.len);
        bytes.reserve(len);
        unsafe {
            // Copy the string content to bytes
            let dst = bytes.ptr.as_ptr().add(bytes.len);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, len);
            bytes.len += len;
        }
    } else {
        calibrate_offsets.push(usize::MAX);
    }
}

fn calibrate_pointers(
    base_ptr: *mut u8,
    views: &mut [StringView<'static>],
    calibrate_offsets: &[usize],
) {
    calibrate_offsets
        .iter()
        .zip(views)
        .for_each(|(&offset, view)| unsafe {
            if offset != usize::MAX {
                view.calibrate(base_ptr.add(offset))
            }
        });
}

impl<V: AsRef<str> + Debug> FromIterator<Option<V>> for StringArray {
    #[must_use]
    fn from_iter<T: IntoIterator<Item = Option<V>>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut bytes = AlignedVec::<u8>::new();
        let mut views = AlignedVec::<StringView<'static>>::with_capacity(lower);
        let mut calibrate_offsets = Vec::<usize>::with_capacity(lower);
        let mut validity = Bitmap::new();

        for val in iter {
            views.reserve(1);
            match val {
                Some(val) => {
                    push_str(val.as_ref(), &mut bytes, &mut views, &mut calibrate_offsets);
                    validity.push(true);
                }
                None => {
                    unsafe {
                        *views.ptr.as_ptr().add(views.len) = StringView::default();
                    }
                    calibrate_offsets.push(usize::MAX);
                    validity.push(false);
                }
            }
            views.len += 1;
        }

        calibrate_pointers(bytes.ptr.as_ptr(), views.as_mut_slice(), &calibrate_offsets);

        Self {
            logical_type: LogicalType::VarChar,
            _bytes: SwarPtr::new(bytes),
            views: SwarPtr::new(views),
            validity: SwarPtr::new(validity),
            _calibrate_offsets: calibrate_offsets,
        }
    }
}

impl ScalarArray for StringArray {
    const PHYSCIAL_TYPE: PhysicalType = PhysicalType::String;

    #[inline]
    unsafe fn replace_with_trusted_len_values_iterator(
        &mut self,
        len: usize,
        trusted_len_iterator: impl Iterator<Item = Self::Element>,
    ) {
        self.validity.as_mut().clear();

        let uninitiated_views = self.views.as_mut().clear_and_resize(len);
        let uninitiated_bytes = self._bytes.as_mut();
        uninitiated_bytes.clear();
        self._calibrate_offsets.clear();

        trusted_len_iterator
            .enumerate()
            .for_each(|(index, element)| {
                assign_string_element(
                    element,
                    uninitiated_bytes,
                    uninitiated_views,
                    index,
                    &mut self._calibrate_offsets,
                )
            });

        calibrate_pointers(
            uninitiated_bytes.ptr.as_ptr(),
            uninitiated_views,
            &self._calibrate_offsets,
        )
    }

    unsafe fn replace_with_trusted_len_iterator(
        &mut self,
        len: usize,
        trusted_len_iterator: impl Iterator<Item = Option<Self::Element>>,
    ) {
        let uninitiated_validity = self.validity.as_mut();

        let uninitiated_views = self.views.as_mut().clear_and_resize(len);
        let uninitiated_bytes = self._bytes.as_mut();
        uninitiated_bytes.clear();
        self._calibrate_offsets.clear();

        uninitiated_validity.reset(
            len,
            trusted_len_iterator.enumerate().map(|(index, element)| {
                if let Some(element) = element {
                    assign_string_element(
                        element,
                        uninitiated_bytes,
                        uninitiated_views,
                        index,
                        &mut self._calibrate_offsets,
                    );
                    true
                } else {
                    false
                }
            }),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_array_from_iter() {
        let data = [
            "curvature",
            "",
            "auto",
            "vectorization",
            "auto-vectorization",
        ];
        let string_array = StringArray::from_values_iter(data.iter());

        assert_eq!(
            string_array
                .values_iter()
                .map(|v| v.as_str().to_string())
                .collect::<Vec<_>>(),
            data
        );

        let data = [Some("curvature"), Some("auto"), None, Some("vectorization")];
        let string_array = StringArray::from_iter(data.iter().copied());

        assert_eq!(
            string_array
                .iter()
                .map(|v| v.as_ref().map(|v| v.as_str().to_string()))
                .collect::<Vec<_>>(),
            data.iter()
                .map(|v| v.as_ref().map(|v| v.to_string()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_string_array_replace_with_trusted_len_values_iter() {
        let data = [
            "curvature",
            "",
            "auto",
            "vectorization",
            "auto-vectorization",
        ];

        let iter = data.iter().map(|v| StringElement::new(v.to_string()));

        let mut string_array = StringArray::new(LogicalType::VarChar).unwrap();
        unsafe { string_array.replace_with_trusted_len_values_iterator(5, iter) }

        assert_eq!(
            string_array
                .values_iter()
                .map(|v| v.as_str().to_string())
                .collect::<Vec<_>>(),
            data
        );
    }
}
