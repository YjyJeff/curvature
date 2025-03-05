//! Array that stores the variable-width data types

use std::fmt::Debug;

use snafu::ensure;

use crate::aligned_vec::AlignedVec;
use crate::bitmap::Bitmap;
use crate::element::Element;
use crate::private::Sealed;
use crate::types::{LogicalType, PhysicalType};

use super::iter::ArrayValuesIter;
use super::swar::SwarPtr;
use super::{Array, InvalidLogicalTypeSnafu, Result};

/// Offset in the binary array
///
/// Most of the time, we are running on 64 bit machines
pub type Offset = u64;

/// [`BinaryArray`] stores the array of `Vec<u8>`
///
/// This is different with array of String. Comparison on [`BinaryArray`] is rarely seen
pub struct BinaryArray {
    logical_type: LogicalType,
    /// A continuous byte array that stores the data of the element, to access the data
    /// in the array, you need to depend on [`Self::offsets`]
    pub(crate) bytes: SwarPtr<AlignedVec<u8>>,
    /// offsets[i] and offsets[i+1] represents the start and end address of the ith
    /// element in the array. Therefore, its length is always equal to self.len() + 1
    pub(crate) offsets: SwarPtr<AlignedVec<Offset>>,
    /// Validity map of the array
    pub(crate) validity: SwarPtr<Bitmap>,
}

impl BinaryArray {
    /// Create a new [`BinaryArray`]
    #[inline]
    pub fn new(logical_type: LogicalType) -> Result<Self> {
        Self::with_capacity(logical_type, 0)
    }

    /// Create a new [`BinaryArray`] without check
    ///
    /// # Safety
    ///
    /// physical type of the logical type should be `Binary`
    #[inline]
    pub unsafe fn new_unchecked(logical_type: LogicalType) -> Self {
        unsafe {
            #[cfg(feature = "verify")]
            assert_eq!(logical_type.physical_type(), PhysicalType::Binary);

            Self::with_capacity_unchecked(logical_type, 0)
        }
    }

    /// Create a new [`BinaryArray`] with given capacity
    #[inline]
    pub fn with_capacity(logical_type: LogicalType, capacity: usize) -> Result<Self> {
        ensure!(
            logical_type.physical_type() == PhysicalType::Binary,
            InvalidLogicalTypeSnafu {
                array_name: "BinaryArray".to_string(),
                array_physical_type: PhysicalType::Binary,
                logical_type,
            }
        );

        // SAFETY: we check the physical type above
        unsafe { Ok(Self::with_capacity_unchecked(logical_type, capacity)) }
    }

    /// Create a new [`BinaryArray`] with given capacity
    ///
    /// # Safety
    ///
    /// physical type of the logical type should be `Binary`
    #[inline]
    pub unsafe fn with_capacity_unchecked(logical_type: LogicalType, capacity: usize) -> Self {
        unsafe {
            #[cfg(feature = "verify")]
            assert_eq!(logical_type.physical_type(), PhysicalType::Binary);

            let mut offsets = AlignedVec::<Offset>::with_capacity(capacity + 1);
            *offsets.ptr.as_ptr() = 0;
            offsets.len = 1;
            Self {
                logical_type,
                bytes: SwarPtr::default(),
                offsets: SwarPtr::new(offsets),
                validity: SwarPtr::default(),
            }
        }
    }

    /// Get the number of elements in the [`BinaryArray`]
    #[inline]
    fn len(&self) -> usize {
        self.offsets.len - 1
    }

    /// Construct [`Self`] from iterator of `&[u8]`. It is pretty slow, because
    /// we need do multiple times of reserve
    #[must_use]
    pub fn from_values_iter<V: AsRef<[u8]>, I: IntoIterator<Item = V>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut bytes = AlignedVec::<u8>::new();
        let mut offsets = AlignedVec::<Offset>::with_capacity(lower + 1);
        // SAFETY: offsets is allocated successfully, index 0 is valid
        unsafe {
            *offsets.ptr.as_ptr() = 0;
            offsets.len = 1;
        }
        for val in iter.into_iter() {
            let val_bytes = val.as_ref();
            let new_len = bytes.len() + val_bytes.len();
            bytes.reserve(val_bytes.len());
            offsets.reserve(1);
            // SAFETY: we have reserve the bytes
            unsafe {
                // Copy data to bytes
                std::ptr::copy_nonoverlapping(
                    val_bytes.as_ptr(),
                    bytes.ptr.as_ptr().add(bytes.len),
                    val_bytes.len(),
                );
                bytes.len = new_len;

                *offsets.ptr.as_ptr().add(offsets.len) = new_len as Offset;
                offsets.len += 1;
            }
        }

        Self {
            logical_type: LogicalType::VarBinary,
            bytes: SwarPtr::new(bytes),
            offsets: SwarPtr::new(offsets),
            validity: SwarPtr::default(),
        }
    }
}

impl Debug for BinaryArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BinaryArray {{ logical_type: {:?} len: {}, data: ",
            self.logical_type,
            self.len()
        )?;
        f.debug_list().entries(self.iter()).finish()?;
        writeln!(f, "}}")
    }
}

impl Sealed for BinaryArray {}

impl Array for BinaryArray {
    type Element = Vec<u8>;
    const PHYSICAL_TYPE: PhysicalType = PhysicalType::Binary;

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
        self.len()
    }

    #[inline]
    fn validity(&self) -> &Bitmap {
        &self.validity
    }

    #[inline]
    unsafe fn validity_mut(&mut self) -> &mut Bitmap {
        unsafe { self.validity.as_mut() }
    }

    #[inline]
    unsafe fn get_value_unchecked(
        &self,
        index: usize,
    ) -> <Self::Element as Element>::ElementRef<'_> {
        unsafe {
            let start = *self.offsets.get_unchecked(index) as usize;
            let end = *self.offsets.get_unchecked(index + 1) as usize;
            self.bytes.get_slice_unchecked(start, end - start)
        }
    }

    #[inline]
    fn logical_type(&self) -> &LogicalType {
        &self.logical_type
    }

    // Mutate array

    #[inline]
    fn reference(&mut self, other: &Self) {
        self.bytes.reference(&other.bytes);
        self.offsets.reference(&other.offsets);
        self.validity.reference(&other.validity);
    }

    #[inline]
    unsafe fn set_all_invalid(&mut self, len: usize) {
        unsafe {
            self.validity.as_mut().mutate().set_all_invalid(len);
            self.offsets
                .as_mut()
                .clear_and_resize(len + 1)
                .iter_mut()
                .for_each(|offset| *offset = 0);
            self.bytes.as_mut().clear();
        }
    }

    unsafe fn replace_with_trusted_len_values_iterator<I>(
        &mut self,
        len: usize,
        trusted_len_iterator: I,
    ) where
        I: Iterator<Item = Self::Element>,
    {
        unsafe {
            self.validity.as_mut().mutate().clear();
            let bytes = self.bytes.as_mut();
            bytes.clear();

            let offsets = &mut self
                .offsets
                .as_mut()
                .clear_and_resize(len + 1)
                .get_unchecked_mut(1..);
            offsets
                .iter_mut()
                .zip(trusted_len_iterator)
                .for_each(|(offset, element)| {
                    copy_bytes(&element, bytes);
                    *offset = bytes.len as _;
                });
        }
    }

    unsafe fn replace_with_trusted_len_iterator<I>(&mut self, len: usize, trusted_len_iterator: I)
    where
        I: Iterator<Item = Option<Self::Element>>,
    {
        unsafe {
            let mut uninitiated_validity = self.validity.as_mut().mutate();

            let bytes = self.bytes.as_mut();
            bytes.clear();
            let offsets = &mut self
                .offsets
                .as_mut()
                .clear_and_resize(len + 1)
                .get_unchecked_mut(1..);

            uninitiated_validity.reset(
                len,
                offsets
                    .iter_mut()
                    .zip(trusted_len_iterator)
                    .map(|(offset, element)| {
                        let not_null = if let Some(element) = element {
                            copy_bytes(&element, bytes);
                            true
                        } else {
                            false
                        };
                        *offset = bytes.len as _;

                        not_null
                    }),
            );
        }
    }

    unsafe fn replace_with_trusted_len_values_ref_iterator<'a, I>(
        &mut self,
        len: usize,
        trusted_len_iterator: I,
    ) where
        I: Iterator<Item = &'a [u8]> + 'a,
    {
        unsafe {
            self.validity.as_mut().mutate().clear();
            let bytes = self.bytes.as_mut();
            bytes.clear();

            let offsets = &mut self
                .offsets
                .as_mut()
                .clear_and_resize(len + 1)
                .get_unchecked_mut(1..);
            offsets
                .iter_mut()
                .zip(trusted_len_iterator)
                .for_each(|(offset, element)| {
                    copy_bytes(element, bytes);
                    *offset = bytes.len as _;
                });
        }
    }

    unsafe fn replace_with_trusted_len_ref_iterator<'a, I>(
        &mut self,
        len: usize,
        trusted_len_iterator: I,
    ) where
        I: Iterator<Item = Option<&'a [u8]>> + 'a,
    {
        unsafe {
            let mut uninitiated_validity = self.validity.as_mut().mutate();

            let bytes = self.bytes.as_mut();
            bytes.clear();
            let offsets = &mut self
                .offsets
                .as_mut()
                .clear_and_resize(len + 1)
                .get_unchecked_mut(1..);

            uninitiated_validity.reset(
                len,
                offsets
                    .iter_mut()
                    .zip(trusted_len_iterator)
                    .map(|(offset, element)| {
                        let not_null = if let Some(element) = element {
                            copy_bytes(element, bytes);
                            true
                        } else {
                            false
                        };
                        *offset = bytes.len as _;

                        not_null
                    }),
            );
        }
    }

    #[inline]
    unsafe fn clear(&mut self) {
        unsafe {
            self.bytes.as_mut().clear();
            let _ = self.offsets.as_mut().clear_and_resize(1);
            self.validity.as_mut().mutate().clear();
        }
    }

    unsafe fn copy(&mut self, source: &Self, start: usize, len: usize) {
        unsafe {
            #[cfg(feature = "verify")]
            assert!(start + len <= source.len());

            self.validity
                .as_mut()
                .mutate()
                .copy(&source.validity, start, len);

            // Copy offset
            let dst = self.offsets.as_mut().clear_and_resize(len + 1);
            let src = source.offsets.as_ptr().add(start);
            std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), len + 1);
            let calibrate = *src;
            dst.iter_mut().for_each(|v| *v -= calibrate);

            // Copy data
            let data_len = *dst.last().unwrap_unchecked() as usize;
            let dst = self.bytes.as_mut().clear_and_resize(data_len);
            std::ptr::copy_nonoverlapping(
                source.bytes.as_ptr().add(calibrate as usize),
                dst.as_mut_ptr(),
                data_len,
            );
        }
    }
}

#[inline]
fn copy_bytes(element_ref: &[u8], bytes: &mut AlignedVec<u8>) {
    bytes.reserve(element_ref.len());
    // SAFETY: we have reserved the bytes
    unsafe {
        // Copy data to bytes
        std::ptr::copy_nonoverlapping(
            element_ref.as_ptr(),
            bytes.ptr.as_ptr().add(bytes.len),
            element_ref.len(),
        );
    }
    bytes.len += element_ref.len();
}

impl<V: AsRef<[u8]>> FromIterator<Option<V>> for BinaryArray {
    /// Construct [`Self`] from iterator of `Option<T::BytesSliceType>`. It is pretty slow, because
    /// we need do multiple times of reserve
    #[must_use]
    fn from_iter<I: IntoIterator<Item = Option<V>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut bytes = AlignedVec::<u8>::new();
        let mut offsets = AlignedVec::<Offset>::with_capacity(lower + 1);
        let mut validity = Bitmap::with_capacity(lower);
        {
            let mut mutate_validity_guard = validity.mutate();
            // SAFETY: offsets is allocated successfully, index 0 is valid
            unsafe {
                *offsets.ptr.as_ptr() = 0;
                offsets.len = 1;
            }
            for val in iter {
                offsets.reserve(1);
                match val {
                    Some(val) => {
                        let val_bytes = val.as_ref();
                        let new_len = bytes.len() + val_bytes.len();
                        bytes.reserve(val_bytes.len());
                        // SAFETY: we have reserve the bytes
                        unsafe {
                            // Copy data to bytes
                            std::ptr::copy_nonoverlapping(
                                val_bytes.as_ptr(),
                                bytes.ptr.as_ptr().add(bytes.len),
                                val_bytes.len(),
                            );
                        }
                        bytes.len = new_len;
                        mutate_validity_guard.push(true);
                    }
                    None => {
                        mutate_validity_guard.push(false);
                    }
                }

                // SAFETY: we have reserved the space
                unsafe {
                    *offsets.ptr.as_ptr().add(offsets.len) = bytes.len as Offset;
                }
                offsets.len += 1;
            }
        }

        Self {
            logical_type: LogicalType::VarBinary,
            bytes: SwarPtr::new(bytes),
            offsets: SwarPtr::new(offsets),
            validity: SwarPtr::new(validity),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replace_with_trusted_len_values_ref_iter() {
        let mut array = BinaryArray::new(LogicalType::VarBinary).unwrap();

        let data = ["arrow", "curvature", "lock", "free", "query", "wtf"];
        unsafe {
            array.replace_with_trusted_len_values_ref_iterator(
                6,
                data.iter().map(|&v| v.as_bytes()),
            );
        }

        assert_eq!(
            array.values_iter().collect::<Vec<_>>(),
            data.iter().map(|v| v.as_bytes()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_replace_with_trusted_len_ref_iter() {
        let mut array = BinaryArray::new(LogicalType::VarBinary).unwrap();

        let data = [
            Some("arrow"),
            None,
            Some("lock"),
            Some("free"),
            Some("query"),
            None,
        ];
        unsafe {
            array.replace_with_trusted_len_ref_iterator(
                6,
                data.iter().map(|&v| v.map(|v| v.as_bytes())),
            );
        }

        assert_eq!(
            array.iter().collect::<Vec<_>>(),
            data.iter()
                .map(|v| v.map(|v| v.as_bytes()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_copy() {
        let mut array = BinaryArray::new(LogicalType::VarBinary).unwrap();

        let source = BinaryArray::from_values_iter([
            "arrow",
            "quack",
            "lock",
            "curvature",
            "Learn a lot from DuckDB and ClickHouse",
            "Yes!",
        ]);

        unsafe {
            array.copy(&source, 2, 3);
        }

        let gt = unsafe {
            (2..5)
                .map(|index| source.get_value_unchecked(index))
                .collect::<Vec<_>>()
        };

        assert_eq!(array.values_iter().collect::<Vec<_>>(), gt);
    }
}
