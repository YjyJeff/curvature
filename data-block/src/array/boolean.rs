//! BooleanArray

use std::fmt::Debug;

use snafu::ensure;

use super::swar::SwarPtr;
use super::{Array, InvalidLogicalTypeSnafu, Result};
use crate::bitmap::{Bitmap, BitmapIter};
use crate::private::Sealed;
use crate::types::{LogicalType, PhysicalType};

/// Boolean Array
pub struct BooleanArray {
    logical_type: LogicalType,
    pub(crate) data: SwarPtr<Bitmap>,
    pub(crate) validity: SwarPtr<Bitmap>,
}

impl BooleanArray {
    /// Create a new [`BooleanArray`]
    #[inline]
    pub fn try_new(logical_type: LogicalType) -> Result<Self> {
        Self::with_capacity(logical_type, 0)
    }

    /// Create a new [`BooleanArray`] without check
    ///
    /// # Safety
    ///
    /// physical type of the logical type should be `Boolean`
    #[inline]
    pub unsafe fn new_unchecked(logical_type: LogicalType) -> Self {
        Self::with_capacity_unchecked(logical_type, 0)
    }

    /// Create a new [`BooleanArray`] with given capacity
    #[inline]
    pub fn with_capacity(logical_type: LogicalType, capacity: usize) -> Result<Self> {
        ensure!(
            logical_type.physical_type() == PhysicalType::Boolean,
            InvalidLogicalTypeSnafu {
                array_name: "BooleanArray".to_string(),
                array_physical_type: PhysicalType::Boolean,
                logical_type
            }
        );
        // SAFETY: we check the physical type above
        unsafe { Ok(Self::with_capacity_unchecked(logical_type, capacity)) }
    }

    /// Create a new [`BooleanArray`] with given capacity without check
    ///
    /// # Safety
    ///
    /// physical type of the logical type should be `Boolean`
    #[inline]
    pub unsafe fn with_capacity_unchecked(logical_type: LogicalType, capacity: usize) -> Self {
        #[cfg(feature = "verify")]
        assert_eq!(logical_type.physical_type(), PhysicalType::Boolean);

        Self {
            logical_type,
            data: SwarPtr::new(Bitmap::with_capacity(capacity)),
            validity: SwarPtr::new(Bitmap::new()),
        }
    }

    /// Get the mutable data
    ///
    /// # Safety
    ///
    /// You must enforce Rust’s aliasing rules. In particular, while this reference exists,
    /// the Bitmap must not get accessed (read or written) through any other array.
    #[inline]
    pub unsafe fn data_mut(&mut self) -> &mut Bitmap {
        self.data.as_mut()
    }

    /// Get underling bitmap data
    #[inline]
    pub fn data(&self) -> &Bitmap {
        &self.data
    }
}

impl Sealed for BooleanArray {}

impl Array for BooleanArray {
    const PHYSICAL_TYPE: PhysicalType = PhysicalType::Boolean;
    type Element = bool;
    type ValuesIter<'a> = BitmapIter<'a>;

    #[inline]
    fn values_iter(&self) -> Self::ValuesIter<'_> {
        self.data.iter()
    }

    #[inline]
    fn values_slice_iter(&self, offset: usize, length: usize) -> Self::ValuesIter<'_> {
        BitmapIter::new_with_offset_and_len(&self.data, offset, length)
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
    fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    unsafe fn get_value_unchecked(&self, index: usize) -> bool {
        #[cfg(feature = "verify")]
        assert!(index < self.data.len());

        self.data.get_unchecked(index)
    }

    #[inline]
    fn logical_type(&self) -> &LogicalType {
        &self.logical_type
    }

    #[inline]
    fn reference(&mut self, other: &Self) {
        self.data.reference(&other.data);
        self.validity.reference(&other.validity);
    }

    #[inline]
    unsafe fn set_all_invalid(&mut self, len: usize) {
        self.validity.as_mut().mutate().set_all_invalid(len);
        self.data.as_mut().mutate().set_all_invalid(len);
    }

    unsafe fn replace_with_trusted_len_values_iterator<I>(
        &mut self,
        len: usize,
        trusted_len_iterator: I,
    ) where
        I: Iterator<Item = bool>,
    {
        self.validity.as_mut().mutate().clear();
        self.data.as_mut().mutate().reset(len, trusted_len_iterator);
    }

    unsafe fn replace_with_trusted_len_iterator<I>(&mut self, len: usize, trusted_len_iterator: I)
    where
        I: Iterator<Item = Option<bool>>,
    {
        let mut uninitiated = self.data.as_mut().mutate();
        let _ = uninitiated.clear_and_resize(len);
        let mut uninitiated_validity = self.validity.as_mut().mutate();

        uninitiated_validity.reset(
            len,
            trusted_len_iterator.enumerate().map(|(i, val)| {
                if let Some(val) = val {
                    uninitiated.set_unchecked(i, val);
                    true
                } else {
                    false
                }
            }),
        );
    }

    unsafe fn replace_with_trusted_len_values_ref_iterator<'a, I>(
        &mut self,
        len: usize,
        trusted_len_iterator: I,
    ) where
        I: Iterator<Item = bool> + 'a,
    {
        self.replace_with_trusted_len_values_iterator(len, trusted_len_iterator);
    }

    unsafe fn replace_with_trusted_len_ref_iterator<'a, I>(
        &mut self,
        len: usize,
        trusted_len_iterator: I,
    ) where
        I: Iterator<Item = Option<bool>> + 'a,
    {
        self.replace_with_trusted_len_iterator(len, trusted_len_iterator);
    }

    unsafe fn clear(&mut self) {
        self.data.as_mut().mutate().clear();
        self.validity.as_mut().mutate().clear();
    }

    unsafe fn copy(&mut self, source: &Self, start: usize, len: usize) {
        #[cfg(feature = "verify")]
        assert!(start + len <= source.len());

        self.validity
            .as_mut()
            .mutate()
            .copy(&source.validity, start, len);
        self.data.as_mut().mutate().copy(&source.data, start, len);
    }
}

impl Debug for BooleanArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BooleanArray {{ logical_type: {:?}, len: {}, data: ",
            self.logical_type,
            self.len()
        )?;
        f.debug_list().entries(self.iter()).finish()?;
        write!(f, "}}")
    }
}

impl Default for BooleanArray {
    #[inline]
    fn default() -> Self {
        Self {
            logical_type: LogicalType::Boolean,
            data: SwarPtr::default(),
            validity: SwarPtr::default(),
        }
    }
}

impl FromIterator<Option<bool>> for BooleanArray {
    fn from_iter<T: IntoIterator<Item = Option<bool>>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut data = Bitmap::with_capacity(lower);
        let mut validity = Bitmap::with_capacity(lower);
        {
            let mut mutate_data_guard = data.mutate();
            let mut mutate_validity_guard = validity.mutate();
            for val in iter {
                match val {
                    Some(val) => {
                        mutate_data_guard.push(val);
                        mutate_validity_guard.push(true);
                    }
                    None => {
                        mutate_data_guard.push(false);
                        mutate_validity_guard.push(false);
                    }
                }
            }
        }

        Self {
            logical_type: LogicalType::Boolean,
            data: SwarPtr::new(data),
            validity: SwarPtr::new(validity),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl BooleanArray {
        /// Create a new BooleanArray with Bitmap
        pub fn new_with_data(data: Bitmap) -> Self {
            Self {
                logical_type: LogicalType::Boolean,
                data: SwarPtr::new(data),
                validity: SwarPtr::default(),
            }
        }
    }

    #[test]
    fn test_boolean_array_from_iter() {
        let data = [
            Some(true),
            Some(false),
            Some(true),
            Some(true),
            None,
            Some(false),
            Some(true),
            Some(true),
            Some(false),
            Some(false),
            None,
            Some(false),
            None,
            Some(false),
            Some(false),
            Some(false),
            Some(true),
        ];

        let array = BooleanArray::from_iter(data.iter().copied());
        assert_eq!(array.iter().collect::<Vec<_>>(), data)
    }
}
