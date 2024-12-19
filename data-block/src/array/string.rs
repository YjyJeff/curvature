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
use super::{Array, InvalidLogicalTypeSnafu, Result};

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
        #[cfg(feature = "verify")]
        assert_eq!(logical_type.physical_type(), PhysicalType::String);

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

        calibrate_pointers(bytes.as_ptr(), views.as_mut_slice(), &calibrate_offsets);

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
    const PHYSICAL_TYPE: PhysicalType = PhysicalType::String;
    type Element = StringElement;

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
        #[cfg(feature = "verify")]
        assert!(index < self.len());

        self.views.get_unchecked(index).shorten()
    }

    #[inline]
    fn logical_type(&self) -> &LogicalType {
        &self.logical_type
    }

    // Mutate array

    #[inline]
    fn reference(&mut self, other: &Self) {
        self._bytes.reference(&other._bytes);
        self.views.reference(&other.views);
        self.validity.reference(&other.validity);
    }

    #[inline]
    unsafe fn set_all_invalid(&mut self, len: usize) {
        self.validity.as_mut().mutate().set_all_invalid(len);
        // Reset all of the string view with default
        self.views
            .as_mut()
            .clear_and_resize(len)
            .iter_mut()
            .for_each(|view| *view = StringView::default());
        self._bytes.as_mut().clear();
    }

    #[inline]
    unsafe fn replace_with_trusted_len_values_iterator<I>(
        &mut self,
        len: usize,
        trusted_len_iterator: I,
    ) where
        I: Iterator<Item = Self::Element>,
    {
        self.validity.as_mut().mutate().clear();

        let uninitiated_views = self.views.as_mut().clear_and_resize(len);
        let uninitiated_bytes = self._bytes.as_mut();
        uninitiated_bytes.clear();
        self._calibrate_offsets.clear();

        trusted_len_iterator
            .enumerate()
            .for_each(|(index, element)| {
                assign_string_view(
                    element.view,
                    uninitiated_bytes,
                    uninitiated_views,
                    index,
                    &mut self._calibrate_offsets,
                )
            });

        calibrate_pointers(
            uninitiated_bytes.as_ptr(),
            uninitiated_views,
            &self._calibrate_offsets,
        )
    }

    unsafe fn replace_with_trusted_len_iterator<I>(&mut self, len: usize, trusted_len_iterator: I)
    where
        I: Iterator<Item = Option<Self::Element>>,
    {
        let mut uninitiated_validity = self.validity.as_mut().mutate();

        let uninitiated_views = self.views.as_mut().clear_and_resize(len);
        let uninitiated_bytes = self._bytes.as_mut();
        uninitiated_bytes.clear();
        self._calibrate_offsets.clear();

        uninitiated_validity.reset(
            len,
            trusted_len_iterator.enumerate().map(|(index, element)| {
                if let Some(element) = element {
                    assign_string_view(
                        element.view,
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

    #[inline]
    unsafe fn replace_with_trusted_len_values_ref_iterator<'a, I>(
        &mut self,
        len: usize,
        trusted_len_iterator: I,
    ) where
        I: Iterator<Item = StringView<'a>>,
    {
        self.validity.as_mut().mutate().clear();

        let uninitiated_views = self.views.as_mut().clear_and_resize(len);
        let uninitiated_bytes = self._bytes.as_mut();
        uninitiated_bytes.clear();
        self._calibrate_offsets.clear();

        trusted_len_iterator.enumerate().for_each(|(index, view)| {
            assign_string_view(
                view,
                uninitiated_bytes,
                uninitiated_views,
                index,
                &mut self._calibrate_offsets,
            )
        });

        calibrate_pointers(
            uninitiated_bytes.as_ptr(),
            uninitiated_views,
            &self._calibrate_offsets,
        )
    }

    unsafe fn replace_with_trusted_len_ref_iterator<'a, I>(
        &mut self,
        len: usize,
        trusted_len_iterator: I,
    ) where
        I: Iterator<Item = Option<StringView<'a>>>,
    {
        let mut uninitiated_validity = self.validity.as_mut().mutate();

        let uninitiated_views = self.views.as_mut().clear_and_resize(len);
        let uninitiated_bytes = self._bytes.as_mut();
        uninitiated_bytes.clear();
        self._calibrate_offsets.clear();

        uninitiated_validity.reset(
            len,
            trusted_len_iterator.enumerate().map(|(index, view)| {
                if let Some(view) = view {
                    assign_string_view(
                        view,
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

    #[inline]
    unsafe fn clear(&mut self) {
        self.views.as_mut().clear();
        self._bytes.as_mut().clear();
        self.validity.as_mut().mutate().clear();
    }

    unsafe fn copy(&mut self, source: &Self, start: usize, len: usize) {
        #[cfg(feature = "verify")]
        assert!(start + len <= source.len());

        self.validity
            .as_mut()
            .mutate()
            .copy(&source.validity, start, len);

        let mut bytes_len = 0;
        let mut bytes_src = None;
        self._calibrate_offsets.clear();
        source
            .views
            .get_slice_unchecked(start, len)
            .iter()
            .for_each(|view| {
                if view.is_inlined() {
                    self._calibrate_offsets.push(usize::MAX);
                } else {
                    if bytes_src.is_none() {
                        bytes_src = Some(view.indirect_ptr());
                    }
                    self._calibrate_offsets.push(bytes_len);
                    bytes_len += view.length as usize;
                }
            });

        // Copy View
        let views_dst = self.views.as_mut().clear_and_resize(len);
        std::ptr::copy_nonoverlapping(
            source.views.as_ptr().add(start),
            views_dst.as_mut_ptr(),
            len,
        );

        // Copy bytes
        let bytes_dst = self._bytes.as_mut().clear_and_resize(bytes_len);
        if let Some(bytes_src) = bytes_src {
            std::ptr::copy_nonoverlapping(bytes_src, bytes_dst.as_mut_ptr(), bytes_len);
            calibrate_pointers(bytes_dst.as_ptr(), views_dst, &self._calibrate_offsets);
        }
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
unsafe fn assign_string_view(
    view: StringView<'_>,
    bytes: &mut AlignedVec<u8>,
    views: &mut [StringView<'static>],
    index: usize,
    calibrate_offsets: &mut Vec<usize>,
) {
    *views.get_unchecked_mut(index) = view.expand();

    if view.is_inlined() {
        // Do not need to calibrate
        calibrate_offsets.push(usize::MAX);
    } else {
        let len = view.length as usize;
        calibrate_offsets.push(bytes.len);
        bytes.reserve(len);
        unsafe {
            // Copy the string content to bytes
            let dst = bytes.ptr.as_ptr().add(bytes.len);
            std::ptr::copy_nonoverlapping(view.as_ptr(), dst, len);
            bytes.len += len;
        }
    }
}

fn calibrate_pointers(
    base_ptr: *const u8,
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
        {
            let mut mutate_validity_guard = validity.mutate();

            for val in iter {
                views.reserve(1);
                match val {
                    Some(val) => {
                        push_str(val.as_ref(), &mut bytes, &mut views, &mut calibrate_offsets);
                        mutate_validity_guard.push(true);
                    }
                    None => {
                        unsafe {
                            *views.ptr.as_ptr().add(views.len) = StringView::default();
                        }
                        calibrate_offsets.push(usize::MAX);
                        mutate_validity_guard.push(false);
                    }
                }
                views.len += 1;
            }

            calibrate_pointers(bytes.as_ptr(), views.as_mut_slice(), &calibrate_offsets);
        }

        Self {
            logical_type: LogicalType::VarChar,
            _bytes: SwarPtr::new(bytes),
            views: SwarPtr::new(views),
            validity: SwarPtr::new(validity),
            _calibrate_offsets: calibrate_offsets,
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

    #[test]
    fn test_string_array_replace_with_trusted_len_values_ref_iter() {
        let data = [
            "curvature",
            "",
            "auto",
            "vectorization",
            "auto-vectorization",
        ];

        let iter = data.iter().map(|v| StringView::from_static_str(v));

        let mut string_array = StringArray::new(LogicalType::VarChar).unwrap();
        unsafe { string_array.replace_with_trusted_len_values_ref_iterator(5, iter) }

        assert_eq!(
            string_array
                .values_iter()
                .map(|v| v.as_str().to_string())
                .collect::<Vec<_>>(),
            data
        );
    }

    #[test]
    fn test_copy() {
        let mut array = StringArray::new(LogicalType::VarChar).unwrap();
        let source = StringArray::from_values_iter([
            "auto-vectorization",
            "ClickHouse",
            "Curvature is fast",
            "auto-vectorization",
            "Yes!",
            "No",
        ]);

        unsafe {
            array.copy(&source, 1, 5);
        }

        assert_eq!(
            array.values_iter().collect::<Vec<_>>(),
            [
                StringView::from_static_str("ClickHouse"),
                StringView::from_static_str("Curvature is fast"),
                StringView::from_static_str("auto-vectorization"),
                StringView::from_static_str("Yes!"),
                StringView::from_static_str("No"),
            ]
        );
    }
}
