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
use super::ping_pong::PingPongPtr;
use super::{Array, InvalidLogicalTypeSnafu, MutateArrayExt, Result};

/// [`Array`] of string
pub struct StringArray {
    logical_type: LogicalType,
    /// A continuous byte array that stores the content of the
    /// indirection String
    _bytes: PingPongPtr<AlignedVec<u8>>,
    /// Views of the String. Use [`StringView`] here is safe! The pointer in the views
    /// points to the [`Self::_bytes`], therefore the 'static lifetime is fake !!! It
    /// is a hack to allow the self-referential
    pub(crate) views: PingPongPtr<AlignedVec<StringView<'static>>>,
    pub(crate) validity: PingPongPtr<Bitmap>,
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
            _bytes: PingPongPtr::default(),
            views: PingPongPtr::new(AlignedVec::with_capacity(capacity)),
            validity: PingPongPtr::default(),
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
        let mut offsets = Vec::<usize>::with_capacity(lower);

        for val in iter {
            views.reserve(1);
            push_str(val.as_ref(), &mut bytes, &mut views, &mut offsets);
            views.len += 1;
        }

        calibrate_pointers(bytes.ptr.as_ptr(), views.as_mut_slice(), &offsets);

        Self {
            logical_type: LogicalType::VarChar,
            _bytes: PingPongPtr::new(bytes),
            views: PingPongPtr::new(views),
            validity: PingPongPtr::default(),
        }
    }
}

impl Debug for StringArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StringArray {{ len: {}, data: ", self.len())?;
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
    fn validity_mut(&mut self) -> &mut PingPongPtr<Bitmap> {
        &mut self.validity
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
    offsets: &mut Vec<usize>,
) {
    let len = val.len();
    let view = if len <= INLINE_LEN {
        offsets.push(usize::MAX);
        StringView::new_inline(val)
    } else {
        offsets.push(bytes.len);
        bytes.reserve(len);
        unsafe {
            // Copy the string content to bytes
            let dst = bytes.ptr.as_ptr().add(bytes.len);
            std::ptr::copy_nonoverlapping(val.as_ptr(), dst, len);
            bytes.len += len;
            StringView::new_indirect(dst, len)
        }
    };
    unsafe {
        *views.ptr.as_ptr().add(views.len) = view;
    }
}

fn calibrate_pointers(base_ptr: *mut u8, views: &mut [StringView<'static>], offsets: &[usize]) {
    offsets
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
        // An auxiliary vector to record the offset to the bytes. Reallocation
        // invalid the address of the bytes, therefore, if we store the pointer
        // to the bytes directly, we will get invalid pointer
        let mut offsets = Vec::<usize>::with_capacity(lower);
        let mut validity = Bitmap::new();

        for val in iter {
            views.reserve(1);
            match val {
                Some(val) => {
                    push_str(val.as_ref(), &mut bytes, &mut views, &mut offsets);
                    validity.push(true);
                }
                None => {
                    unsafe {
                        *views.ptr.as_ptr().add(views.len) = StringView::default();
                    }
                    offsets.push(usize::MAX);
                    validity.push(false);
                }
            }
            views.len += 1;
        }

        calibrate_pointers(bytes.ptr.as_ptr(), views.as_mut_slice(), &offsets);

        Self {
            logical_type: LogicalType::VarChar,
            _bytes: PingPongPtr::new(bytes),
            views: PingPongPtr::new(views),
            validity: PingPongPtr::new(validity),
        }
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
}
