//! BooleanArray

use snafu::ensure;

use super::ping_pong::PingPongPtr;
use super::{Array, ArrayExt, InvalidLogicalTypeSnafu, Result};
use crate::bitmap::{Bitmap, BitmapIter};
use crate::private::Sealed;
use crate::types::{LogicalType, PhysicalType};

/// Boolean Array
#[derive(Debug)]
pub struct BooleanArray {
    logical_type: LogicalType,
    pub(crate) data: PingPongPtr<Bitmap>,
    pub(crate) validity: PingPongPtr<Bitmap>,
}

impl BooleanArray {
    /// Create a new [`BooleanArray`]
    #[inline]
    pub fn new(logical_type: LogicalType) -> Result<Self> {
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
        Self {
            logical_type,
            data: PingPongPtr::new(Bitmap::with_capacity(capacity)),
            validity: PingPongPtr::new(Bitmap::new()),
        }
    }
}

impl Sealed for BooleanArray {}

impl Array for BooleanArray {
    type ScalarType = bool;
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
    fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    unsafe fn get_value_unchecked(&self, index: usize) -> bool {
        self.data.get_unchecked(index)
    }

    #[inline]
    fn logical_type(&self) -> &LogicalType {
        &self.logical_type
    }
}

impl ArrayExt for BooleanArray {
    #[inline]
    fn reference(&mut self, other: &Self) {
        self.data.reference(&other.data);
        self.validity.reference(&other.validity);
    }
}

impl Default for BooleanArray {
    #[inline]
    fn default() -> Self {
        Self {
            logical_type: LogicalType::Boolean,
            data: PingPongPtr::default(),
            validity: PingPongPtr::default(),
        }
    }
}

impl FromIterator<Option<bool>> for BooleanArray {
    fn from_iter<T: IntoIterator<Item = Option<bool>>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut data = Bitmap::with_capacity(lower);
        let mut validity = Bitmap::with_capacity(lower);
        for val in iter {
            match val {
                Some(val) => {
                    data.push(val);
                    validity.push(true);
                }
                None => {
                    data.push(false);
                    validity.push(false);
                }
            }
        }

        Self {
            logical_type: LogicalType::Boolean,
            data: PingPongPtr::new(data),
            validity: PingPongPtr::new(validity),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl BooleanArray {
        pub fn new_with_data(data: Bitmap) -> Self {
            Self {
                logical_type: LogicalType::Boolean,
                data: PingPongPtr::new(data),
                validity: PingPongPtr::default(),
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
