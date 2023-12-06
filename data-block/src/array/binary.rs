//! Array that stores the variable-width data types

use std::fmt::Debug;

use snafu::ensure;

use crate::aligned_vec::AlignedVec;
use crate::bitmap::Bitmap;
use crate::private::Sealed;
use crate::scalar::Scalar;
use crate::types::{LogicalType, PhysicalType};

use super::iter::ArrayValuesIter;
use super::ping_pong::PingPongPtr;
use super::{Array, ArrayExt, InvalidLogicalTypeSnafu, Result};

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
    pub(crate) bytes: PingPongPtr<AlignedVec<u8>>,
    /// offsets[i] and offsets[i+1] represents the start and end address of the ith
    /// element in the array. Therefore, its length is always equal to self.len() + 1
    pub(crate) offsets: PingPongPtr<AlignedVec<Offset>>,
    /// Validity map of the array
    pub(crate) validity: PingPongPtr<Bitmap>,
}

impl BinaryArray {
    /// Create a new [`BinaryArray`]
    #[inline]
    pub fn new(logical_type: LogicalType) -> Result<Self> {
        Self::with_capacity(logical_type, 0)
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
        let mut offsets = AlignedVec::<Offset>::with_capacity(capacity + 1);
        // SAFETY: offsets is allocated successfully, index 0 is valid
        unsafe {
            *offsets.ptr.as_ptr().add(1) = 0;
            offsets.len = 1;
        }
        Ok(Self {
            logical_type,
            bytes: PingPongPtr::default(),
            offsets: PingPongPtr::new(offsets),
            validity: PingPongPtr::default(),
        })
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
            bytes: PingPongPtr::new(bytes),
            offsets: PingPongPtr::new(offsets),
            validity: PingPongPtr::default(),
        }
    }
}

impl Debug for BinaryArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BinaryArray {{ len: {}, data: ", self.len())?;
        f.debug_list().entries(self.iter()).finish()?;
        writeln!(f, "}}")
    }
}

impl Sealed for BinaryArray {}

impl Array for BinaryArray {
    type ScalarType = Vec<u8>;

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
        self.len()
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
        let start = *self.offsets.get_unchecked(index) as usize;
        let end = *self.offsets.get_unchecked(index + 1) as usize;
        self.bytes.get_slice_unchecked(start, end - start)
    }

    #[inline]
    fn logical_type(&self) -> &LogicalType {
        &self.logical_type
    }
}

impl ArrayExt for BinaryArray {
    #[inline]
    fn reference(&mut self, other: &Self) {
        self.bytes.reference(&other.bytes);
        self.offsets.reference(&other.offsets);
        self.validity.reference(&other.validity);
    }
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
                        bytes.len = new_len;
                        validity.push(true);
                    }
                }
                None => {
                    validity.push(false);
                }
            }

            // SAFETY: we have reserved the space
            unsafe {
                *offsets.ptr.as_ptr().add(offsets.len) = bytes.len as Offset;
            }
            offsets.len += 1;
        }

        Self {
            logical_type: LogicalType::VarBinary,
            bytes: PingPongPtr::new(bytes),
            offsets: PingPongPtr::new(offsets),
            validity: PingPongPtr::new(validity),
        }
    }
}
