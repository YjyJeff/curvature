//! [`PrimitiveArray`] that stores fixed byte-width data

use super::swar::SwarPtr;
use super::{Array, InvalidLogicalTypeSnafu, MutateArrayExt, Result, ScalarArray};
use snafu::ensure;
use std::fmt::Debug;
use std::iter::Copied;
use std::slice::Iter;

use crate::aligned_vec::{AlignedVec, AllocType};
use crate::bitmap::Bitmap;
use crate::element::interval::DayTime;
use crate::element::Element;
use crate::macros::for_all_primitive_types;
use crate::private::Sealed;
use crate::types::{LogicalType, PhysicalType};

/// Trait for types that can be placed on the [`PrimitiveArray`]
pub trait PrimitiveType: AllocType + for<'a> Element<ElementRef<'a> = Self> + Copy {
    /// Default logical type of this primitive type
    const LOGICAL_TYPE: LogicalType;
}

macro_rules! impl_primitive_type {
    ($({$pt_variant:ident, $primitive_element_ty:ty, $alias:ident, $lt:ident}),*) => {
        $(
            impl PrimitiveType for $primitive_element_ty {
                const LOGICAL_TYPE: LogicalType = LogicalType::$lt;
            }
        )*
    };
}

for_all_primitive_types!(impl_primitive_type);

/// [`PrimitiveArray`] that stores fixed byte-width data, such as `i32` or `f64`
pub struct PrimitiveArray<T: PrimitiveType> {
    logical_type: LogicalType,
    pub(crate) data: SwarPtr<AlignedVec<T>>,
    pub(crate) validity: SwarPtr<Bitmap>,
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
        Self {
            logical_type,
            data: SwarPtr::new(AlignedVec::with_capacity(capacity)),
            validity: SwarPtr::default(),
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
            data: SwarPtr::new(data),
            validity: SwarPtr::default(),
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
        write!(
            f,
            "{}Array {{ logical_type: {:?}, len: {}, data: ",
            T::NAME,
            self.logical_type,
            self.len()
        )?;
        f.debug_list().entries(self.iter()).finish()?;
        write!(f, "}}")
    }
}

impl<T: PrimitiveType> Sealed for PrimitiveArray<T> {}

impl<T> Array for PrimitiveArray<T>
where
    T: for<'a> PrimitiveType<ElementRef<'a> = T>,
{
    type Element = T;

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
    unsafe fn validity_mut(&mut self) -> &mut Bitmap {
        self.validity.as_mut()
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

impl<T> MutateArrayExt for PrimitiveArray<T>
where
    T: for<'a> PrimitiveType<ElementRef<'a> = T>,
    PrimitiveArray<T>: Array<Element = T>,
{
    #[inline]
    fn reference(&mut self, other: &Self) {
        self.data.reference(&other.data);
        self.validity.reference(&other.validity);
    }
}

impl<T> ScalarArray for PrimitiveArray<T>
where
    T: for<'a> PrimitiveType<ElementRef<'a> = T>,
{
    const PHYSCIAL_TYPE: PhysicalType = T::PHYSICAL_TYPE;

    #[inline]
    unsafe fn replace_with_trusted_len_values_iterator(
        &mut self,
        len: usize,
        trusted_len_iterator: impl Iterator<Item = T>,
    ) {
        self.validity.as_mut().clear();

        let uninitiated = self.data.as_mut();
        uninitiated
            .clear_and_resize(len)
            .iter_mut()
            .zip(trusted_len_iterator)
            .for_each(|(uninitiated, item)| {
                *uninitiated = item;
            });
    }

    #[inline]
    unsafe fn replace_with_trusted_len_iterator(
        &mut self,
        len: usize,
        trusted_len_iterator: impl Iterator<Item = Option<T>>,
    ) {
        let uninitiated_vec = self.data.as_mut();
        let uninitiated_slice = uninitiated_vec.clear_and_resize(len);
        let uninitiated_validity = self.validity.as_mut();

        uninitiated_validity.reset(
            len,
            trusted_len_iterator.enumerate().map(|(i, val)| {
                if let Some(val) = val {
                    *uninitiated_slice.get_unchecked_mut(i) = val;
                    true
                } else {
                    *uninitiated_slice.get_unchecked_mut(i) = T::default();
                    false
                }
            }),
        );
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
                    // Init the memory with default, such that all of the bytes in self.data is initialized!
                    // Otherwise, read the slice from self.data may cause undefined behavior
                    *data.ptr.as_ptr().add(data.len) = T::default();
                    validity.push(false);
                }
                data.len += 1;
            }
        }

        Self {
            logical_type: T::LOGICAL_TYPE,
            data: SwarPtr::new(data),
            validity: SwarPtr::new(validity),
        }
    }
}

impl<T: PrimitiveType> Default for PrimitiveArray<T> {
    #[inline]
    fn default() -> Self {
        Self {
            logical_type: T::LOGICAL_TYPE,
            data: SwarPtr::default(),
            validity: SwarPtr::default(),
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
