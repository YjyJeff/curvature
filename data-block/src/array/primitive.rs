//! [`PrimitiveArray`] that stores fixed byte-width data

use super::ping_pong::PingPongPtr;
use super::{Array, ArrayExt, InvalidLogicalTypeSnafu, Result};
use snafu::ensure;
use std::fmt::Debug;
use std::iter::Copied;
use std::slice::Iter;

use crate::aligned_vec::{AlignedVec, AllocType};
use crate::bitmap::Bitmap;
use crate::macros::for_all_primitive_types;
use crate::private::Sealed;
use crate::scalar::interval::DayTime;
use crate::scalar::Scalar;
use crate::types::LogicalType;

/// Trait for types that can be placed on the [`PrimitiveArray`]
pub trait PrimitiveType: AllocType + Scalar + Copy {
    /// Default logical type of this primitive type
    const LOGICAL_TYPE: LogicalType;
}

macro_rules! impl_primitive_type {
    ($({$pt_variant:ident, $primitive_scalar_ty:ty, $alias:ident, $lt:ident}),*) => {
        $(
            impl PrimitiveType for $primitive_scalar_ty {
                const LOGICAL_TYPE: LogicalType = LogicalType::$lt;
            }
        )*
    };
}

for_all_primitive_types!(impl_primitive_type);

/// [`PrimitiveArray`] that stores fixed byte-width data, such as `i32` or `f64`
pub struct PrimitiveArray<T: PrimitiveType> {
    logical_type: LogicalType,
    pub(crate) data: PingPongPtr<AlignedVec<T>>,
    pub(crate) validity: PingPongPtr<Bitmap>,
}

impl<T: PrimitiveType> PrimitiveArray<T> {
    /// Create a new empty [`PrimitiveArray`]
    #[inline]
    pub fn new(logical_type: LogicalType) -> Result<Self> {
        Self::with_capacity(logical_type, 0)
    }

    /// Create a new empty [`PrimitiveArray`] without check
    ///
    /// # Safety
    ///
    /// physical type of the logical type should be `T::PHYSICAL_TYPE`
    #[inline]
    pub unsafe fn new_unchecked(logical_type: LogicalType) -> Self {
        Self::with_capacity_unchecked(logical_type, 0)
    }

    /// Create a new empty [`PrimitiveArray`] with given capacity
    #[inline]
    pub fn with_capacity(logical_type: LogicalType, capacity: usize) -> Result<Self> {
        ensure!(
            logical_type.physical_type() == T::PHYSICAL_TYPE,
            InvalidLogicalTypeSnafu {
                array_name: format!("{}Array", T::NAME),
                array_physical_type: T::PHYSICAL_TYPE,
                logical_type,
            }
        );
        // SAFETY: we check the physical type above
        unsafe { Ok(Self::with_capacity_unchecked(logical_type, capacity)) }
    }

    /// Create a new empty [`PrimitiveArray`] with given capacity without check
    ///
    /// # Safety
    ///
    /// physical type of the logical type should be `T::PHYSICAL_TYPE`
    #[inline]
    pub unsafe fn with_capacity_unchecked(logical_type: LogicalType, capacity: usize) -> Self {
        PrimitiveArray {
            logical_type,
            data: PingPongPtr::new(AlignedVec::with_capacity(capacity)),
            validity: PingPongPtr::default(),
        }
    }

    /// Get the values of the array
    #[inline]
    pub fn values(&self) -> &[T] {
        self.data.as_slice()
    }

    /// Construct [`Self`] from iterator of values
    pub fn from_values_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (low, _) = iter.size_hint();
        let mut data = AlignedVec::<T>::with_capacity(low);
        for val in iter {
            data.reserve(1);
            unsafe {
                *data.ptr.as_ptr().add(data.len) = val;
                data.len += 1;
            }
        }

        Self {
            logical_type: T::LOGICAL_TYPE,
            data: PingPongPtr::new(data),
            validity: PingPongPtr::default(),
        }
    }
}

macro_rules! alias {
    ($({$_:ident, $ty:ty, $alias:ident, $lt:ident}),*) => {
        $(
            #[doc = concat!("A [`PrimitiveArray`] of [`", stringify!($ty), "`]")]
            pub type $alias = PrimitiveArray<$ty>;
        )*
    };
}

for_all_primitive_types!(alias);

impl<T: PrimitiveType> Debug for PrimitiveArray<T>
where
    Self: Array,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}Array {{ len: {}, data: ", T::NAME, self.len())?;
        f.debug_list().entries(self.iter()).finish()?;
        writeln!(f, "}}")
    }
}

impl<T: PrimitiveType> Sealed for PrimitiveArray<T> {}

impl<T> Array for PrimitiveArray<T>
where
    T: for<'a> PrimitiveType<RefType<'a> = T>,
{
    type ScalarType = T;

    type ValuesIter<'a> = Copied<Iter<'a, T>>;

    #[inline]
    fn values_iter(&self) -> Self::ValuesIter<'_> {
        self.data.as_slice().iter().copied()
    }

    #[inline]
    fn values_slice_iter(&self, offset: usize, length: usize) -> Self::ValuesIter<'_> {
        self.data.as_slice()[offset..offset + length]
            .iter()
            .copied()
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
    unsafe fn get_value_unchecked(&self, index: usize) -> T {
        *self.data.get_unchecked(index)
    }

    #[inline]
    fn logical_type(&self) -> &LogicalType {
        &self.logical_type
    }
}

impl<T> ArrayExt for PrimitiveArray<T>
where
    T: PrimitiveType,
    PrimitiveArray<T>: Array,
{
    #[inline]
    fn reference(&mut self, other: &Self) {
        self.data.reference(&other.data);
        self.validity.reference(&other.validity);
    }
}

impl<T: PrimitiveType> FromIterator<Option<T>> for PrimitiveArray<T> {
    fn from_iter<I: IntoIterator<Item = Option<T>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut data = AlignedVec::<T>::with_capacity(lower);
        let mut validity = Bitmap::with_capacity(lower);
        for val in iter {
            data.reserve(1);
            unsafe {
                if let Some(val) = val {
                    *data.ptr.as_ptr().add(data.len) = val;
                    validity.push(true);
                } else {
                    validity.push(false);
                }
                data.len += 1;
            }
        }

        Self {
            logical_type: T::LOGICAL_TYPE,
            data: PingPongPtr::new(data),
            validity: PingPongPtr::new(validity),
        }
    }
}

impl<T: PrimitiveType> Default for PrimitiveArray<T> {
    #[inline]
    fn default() -> Self {
        Self {
            logical_type: T::LOGICAL_TYPE,
            data: PingPongPtr::default(),
            validity: PingPongPtr::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_array_from_iter() {
        let data = [1.5, 4.6, 7.7];
        let f64_array = Float64Array::from_values_iter(data);
        assert_eq!(f64_array.data.as_slice(), data);

        let data = [Some(1.5), Some(4.6), None, Some(7.7)];
        let f64_array = Float64Array::from_iter(data);
        assert_eq!(f64_array.iter().collect::<Vec<_>>(), data);
    }
}
