//! [`Array`] is the memory format of the columnar storage.
//!
//! Heavily adapted from [`type-exercise-in-rust`](https://github.com/skyzh/type-exercise-in-rust)

pub mod binary;
pub mod boolean;
pub mod iter;
pub mod list;
pub mod primitive;
pub mod string;
pub mod swar;
pub mod utils;

use self::iter::ArrayIter;
use crate::bitmap::{Bitmap, BitmapIter};
use crate::element::{Element, ElementImplRef};
use crate::private::Sealed;
use crate::types::{LogicalType, PhysicalType};
pub use binary::*;
pub use boolean::BooleanArray;
pub use list::ListArray;
pub use primitive::*;
use snafu::Snafu;
use std::fmt::Debug;
use std::iter::FusedIterator;
pub use string::StringArray;

#[allow(missing_docs)]
#[derive(Debug, Snafu)]
pub enum ArrayError {
    #[snafu(display(
        "Invalid logical type `{:?}({})` passed to creating a new array `{}` that has `{}`",
        logical_type,
        logical_type.physical_type(),
        array_name,
        array_physical_type
        ))]
    InvalidLogicalType {
        /// In ideal, array_name should be &'static str. However, we can not create a
        /// &'static str for `PrimitiveArray<T>` currently. An ugly [solution] can be
        /// found here. Keep an eye on this [issue]
        ///
        /// [solution]: https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=51b8c210bc6f51afffa682be99e0476b
        /// [issue]: https://github.com/rodrimati1992/const_format_crates/issues/48
        array_name: String,
        array_physical_type: PhysicalType,
        logical_type: LogicalType,
    },
    #[snafu(display(
        "Can not reference `{array}` to `{target}`, reference requires two arrays have same type"
    ))]
    Reference { array: String, target: String },
    #[snafu(display(
        "Can not filter `{other}` to `{array}`, filter requires two arrays have same type"
    ))]
    Filter { array: String, other: String },
    #[snafu(display("Can not convert `ArrayImpl::{array}` array into `{target}` array"))]
    Convert {
        array: &'static str,
        target: &'static str,
    },
}

type Result<T> = std::result::Result<T, ArrayError>;

/// A trait over all arrays
///
/// # Hack
///
/// In our current design, we can not distinguish the `ConstantArray` and `FlatArray` that
/// has length 1. We will treat it as `ConstantArray` 😂. The advantage of this design is
/// that they have same memory representation, we do not need to allocate/deallocate memory
/// when mutual transformation is performed. The disadvantage is that we can not distinguish
/// them in the compiler stage, which means that we may call the incorrect computation function
/// and the compiler says it is good.... So, when you need to perform computation, remember
/// to check the length of the array
pub trait Array: Sealed + Debug + 'static + Sized {
    /// Physical type of the scalar array
    const PHYSICAL_TYPE: PhysicalType;

    /// Element of the [`Array`]
    type Element: Element;

    /// Iterator of the values in the [`Array`], it returns the reference to the element
    type ValuesIter<'a>: Iterator<Item = <Self::Element as Element>::ElementRef<'a>>
        + FusedIterator
        + ExactSizeIterator;

    /// Get the iterator of values in the [`Array`]
    fn values_iter(&self) -> Self::ValuesIter<'_>;

    /// Get the iterator of values in the array from offset with given length
    ///
    /// offset + length <= self.len()
    fn values_slice_iter(&self, offset: usize, length: usize) -> Self::ValuesIter<'_>;

    /// Get the validity. **If the validity is not empty, the length must equal to
    /// [`Self::len`]**
    fn validity(&self) -> &Bitmap;

    /// Get the mutable validity.
    ///
    /// # Safety
    ///
    /// You must enforce Rust’s aliasing rules. In particular, while this reference exists,
    /// the Bitmap must not get accessed (read or written) through any other array.
    unsafe fn validity_mut(&mut self) -> &mut Bitmap;

    /// Get the number of elements in the [`Array`]
    fn len(&self) -> usize;

    /// Returns `true` if the [`Array`] contains no elements
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a reference to the element at the given index without bound check
    ///
    /// # Safety
    /// Caller should guarantee `index < self.len()`, otherwise, [undefined behavior] happens
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    unsafe fn get_value_unchecked(
        &self,
        index: usize,
    ) -> <Self::Element as Element>::ElementRef<'_>;

    /// Returns a reference to the element at the given index. It will panic if the index
    /// out of bounds
    fn get(&self, index: usize) -> Option<<Self::Element as Element>::ElementRef<'_>> {
        assert!(index < self.len());
        unsafe { self.get_unchecked(index) }
    }

    /// Returns a reference to the element at the given index without bound check
    ///
    /// # Safety
    /// Caller should guarantee `index < self.len()`, otherwise, [undefined behavior] happens
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    unsafe fn get_unchecked(
        &self,
        index: usize,
    ) -> Option<<Self::Element as Element>::ElementRef<'_>> {
        let validity = self.validity();
        if validity.is_empty() || validity.get_unchecked(index) {
            Some(self.get_value_unchecked(index))
        } else {
            None
        }
    }

    /// Get iterator of the array
    #[inline]
    fn iter(&self) -> ArrayIter<'_, Self> {
        ArrayIter::new(self.values_iter(), self.validity())
    }

    /// Get the iterator of elements in the array from offset with given length
    ///
    /// offset + length <= self.len()
    fn slice_iter(&self, offset: usize, length: usize) -> ArrayIter<'_, Self> {
        let bitmap = self.validity();
        let values_iter = self.values_slice_iter(offset, length);
        if bitmap.all_valid() {
            ArrayIter::new_values(values_iter)
        } else {
            ArrayIter::new_values_and_validity(
                values_iter,
                BitmapIter::new_with_offset_and_len(bitmap, offset, length),
            )
        }
    }

    /// Get the logical type of the array. This function enforces each array implementation
    /// should have a [`LogicalType`] as its field
    fn logical_type(&self) -> &LogicalType;

    // Mutate array

    /// Reference self to other array
    fn reference(&mut self, other: &Self);

    /// Resize the array to `len` and set all the elements be `NULL`
    ///
    /// # Note
    ///
    /// We do not provide the default implementation based on the `replace` methods because
    /// it is not optimal
    ///
    /// # Safety
    ///
    /// - Satisfy the mutate condition
    unsafe fn set_all_invalid(&mut self, len: usize);

    /// Replace the array with the trusted_len values iterator that has `len` items
    ///
    /// # Safety
    ///
    /// - The `trusted_len_iterator` must has `len` items
    ///
    /// - Satisfy the mutate condition
    unsafe fn replace_with_trusted_len_values_iterator<I>(
        &mut self,
        len: usize,
        trusted_len_iterator: I,
    ) where
        I: Iterator<Item = Self::Element>;

    /// Replace the array with the trusted_len iterator that has `len` items
    ///
    /// # Safety
    ///
    /// - The `trusted_len_iterator` must has `len` items
    ///
    /// - Satisfy the mutate condition
    unsafe fn replace_with_trusted_len_iterator<I>(&mut self, len: usize, trusted_len_iterator: I)
    where
        I: Iterator<Item = Option<Self::Element>>;

    // FIXME: Ugly !!!! Duplicated code.... Using trait to combine them!

    /// Replace the array with the trusted_len values iterator that has `len` items
    ///
    /// # Safety
    ///
    /// - The `trusted_len_iterator` must has `len` items
    ///
    /// - Satisfy the mutate condition
    unsafe fn replace_with_trusted_len_values_ref_iterator<'a, I>(
        &mut self,
        len: usize,
        trusted_len_iterator: I,
    ) where
        I: Iterator<Item = <Self::Element as Element>::ElementRef<'a>> + 'a;

    /// Replace the array with the trusted_len iterator that has `len` items
    ///
    /// # Safety
    ///
    /// - The `trusted_len_iterator` must has `len` items
    ///
    /// - Satisfy the mutate condition
    unsafe fn replace_with_trusted_len_ref_iterator<'a, I>(
        &mut self,
        len: usize,
        trusted_len_iterator: I,
    ) where
        I: Iterator<Item = Option<<Self::Element as Element>::ElementRef<'a>>> + 'a;

    /// Filter out the elements that are not selected from source, copy the remained
    /// elements into self
    ///
    /// # Safety
    ///
    /// - If the `selection` is not empty, `source` and `selection` should have same length.
    ///   Otherwise, undefined behavior happens
    ///
    /// - `selection` should not be referenced by any array
    unsafe fn filter(&mut self, selection: &Bitmap, source: &Self) {
        #[cfg(feature = "verify")]
        assert!(selection.is_empty() || selection.len() == source.len());

        if selection.all_valid() || source.len() == 1 {
            // All of the elements are selected or source is a Constant Array
            self.reference(source)
        } else {
            crate::compute::filter::filter(selection, source, self)
        }
    }

    /// Clear the array
    ///
    /// # Safety
    ///
    /// Satisfy the mutate condition
    unsafe fn clear(&mut self);

    /// Copy `source[start..start+len]` into self
    ///
    /// # Safety
    ///
    /// - `start + len <= source.len()`
    ///
    /// - Satisfy the mutate condition
    unsafe fn copy(&mut self, source: &Self, start: usize, len: usize);
}

macro_rules! array_impl {
    ($({$variant:ident, $element_ty:ty, $array_ty:ident}),+) => {
        /// Implementations of the [`Array`], enum dispatch
        #[derive(Debug)]
        pub enum ArrayImpl {
            $(
                #[doc = concat!("Array of `", stringify!($element_ty), "`")]
                $variant($array_ty)
            ),+
        }

        impl ArrayImpl{
            /// Create a new empty [`ArrayImpl`] based on the [`LogicalType`]
            ///
            /// Type inference is based on the [`LogicalType`],
            /// caller can use this function to create the [`ArrayImpl`] that match
            /// the required inference result
            pub fn new(logical_type: LogicalType) -> Self {
                let physical_type = logical_type.physical_type();
                macro_rules! create_array_with_physical_type {
                    () => {
                        match physical_type {
                            $(
                                PhysicalType::$variant => ArrayImpl::$variant($array_ty::new_unchecked(logical_type)),
                            )+
                        }
                    }
                }

                unsafe {
                    create_array_with_physical_type!()
                }
            }

            /// Get the number of elements in the Array
            pub fn len(&self) -> usize {
                match self{
                    $(
                        Self::$variant(array) => array.len(),
                    )+
                }
            }

            /// Returns `true` if the Array contains no elements
            pub fn is_empty(&self) -> bool {
                match self{
                    $(
                        Self::$variant(array) => array.is_empty(),
                    )+
                }
            }

            /// Get ident of the array
            pub fn ident(&self) -> &'static str{
                match self{
                    $(
                        Self::$variant(_) => stringify!($variant),
                    )+
                }
            }

            /// Return the [`LogicalType`] of the array
            pub fn logical_type(&self) -> &LogicalType{
                match self{
                    $(
                        Self::$variant(array) => array.logical_type(),
                    )+
                }
            }

            /// Get the validity bitmap
            pub fn validity(&self) -> &Bitmap{
                match self{
                    $(
                        Self::$variant(array) => array.validity(),
                    )+
                }
            }

            /// Get element ref. It will panic if the index out of bounds
            pub fn get(&self, index: usize) -> Option<ElementImplRef<'_>> {
                match self {
                    $(
                        Self::$variant(array) => array.get(index).map(ElementImplRef::$variant),
                    )+
                }
            }

            /// Get element ref without bound check
            ///
            /// # Safety
            ///
            /// `index < self.len()`
            pub unsafe fn get_unchecked(&self, index: usize) -> Option<ElementImplRef<'_>> {
                match self {
                    $(
                        Self::$variant(array) => array.get_unchecked(index).map(ElementImplRef::$variant),
                    )+
                }
            }

            /// Get the value of element ref without bound check
            ///
            /// # Safety
            ///
            /// `index < self.len()`
            pub unsafe fn get_value_unchecked(&self, index: usize) -> ElementImplRef<'_> {
                match self {
                    $(
                        Self::$variant(array) => ElementImplRef::$variant(array.get_value_unchecked(index)),
                    )+
                }
            }

            /// Debug the array slice
            ///
            /// `start + len <= self.len()`
            pub(crate) fn debug_array_slice(
                &self,
                f: &mut std::fmt::Formatter<'_>,
                offset: usize,
                len: usize
            ) -> std::fmt::Result {
                match self{
                    $(
                        Self::$variant(array) => {
                            write!(f, "{} {{ len: {}, data: ", stringify!($array_ty), len)?;
                            f.debug_list().entries(array.slice_iter(offset, len)).finish()?;
                            writeln!(f, "}}")
                        }
                    )+
                }
            }

            /// Reference to other array
            ///
            /// If self and other do not have same physical type, return error
            pub fn reference(&mut self, other: &Self) -> Result<()> {
                macro_rules! reference {
                    () => {
                        match (self, other) {
                            $(
                                (ArrayImpl::$variant(lhs), ArrayImpl::$variant(rhs)) => {
                                    lhs.reference(rhs);
                                }
                            )+
                            (lhs, rhs) => return ReferenceSnafu {
                                array: self::utils::physical_array_name(lhs),
                                target: self::utils::physical_array_name(rhs),
                            }.fail()
                        }
                    };
                }

                reference!();
                Ok(())
            }

            /// Filter out elements that are not selected from source, copy the remained
            /// elements into self
            ///
            /// If self and other do not have same physical type, return error
            ///
            /// # Safety
            ///
            /// - If the `selection` is not empty, `source` and `selection` should have same length.
            ///   Otherwise, undefined behavior happens
            ///
            /// - `selection` should not be referenced by any array
            pub unsafe fn filter(&mut self, selection: &Bitmap, source: &Self) -> Result<()> {
                macro_rules! filter {
                    () => {
                        match (self, source) {
                            $(
                                (ArrayImpl::$variant(lhs), ArrayImpl::$variant(rhs)) => {
                                    unsafe { lhs.filter(selection, rhs) };
                                }
                            )+
                            (lhs, rhs) => return FilterSnafu {
                                array: self::utils::physical_array_name(lhs),
                                other: self::utils::physical_array_name(rhs),
                            }.fail()
                        }
                    };
                }

                filter!();
                Ok(())
            }

            /// Copy the `source[start..start+len]` into self
            ///
            /// # Safety
            ///
            /// - `start + len <= source.len()`
            ///
            /// - Satisfy the mutate condition
            pub unsafe fn copy(&mut self, source: &Self, start: usize, len: usize) -> Result<()>{
                macro_rules! copy {
                    () => {
                        match (self, source) {
                            $(
                                (ArrayImpl::$variant(lhs), ArrayImpl::$variant(rhs)) => {
                                    unsafe { lhs.copy(rhs, start, len) };
                                }
                            )+
                            (lhs, rhs) => return FilterSnafu {
                                array: self::utils::physical_array_name(lhs),
                                other: self::utils::physical_array_name(rhs),
                            }.fail()
                        }
                    };
                }

                copy!();
                Ok(())
            }

            /// Clear the array
            ///
            /// # Safety
            ///
            /// Satisfy the mutate condition
            pub unsafe fn clear(&mut self) {
                match self {
                    $(
                        Self::$variant(array) => array.clear(),
                    )+
                }
            }
        }

        $(
            impl<'a> TryFrom<&'a ArrayImpl> for &'a $array_ty {
                type Error = ArrayError;

                fn try_from(array: &'a ArrayImpl) -> Result<&'a $array_ty>{
                    if let ArrayImpl::$variant(array) = array{
                        Ok(array)
                    }else{
                        ConvertSnafu{
                            array: array.ident(),
                            target: stringify!($array_ty),
                        }.fail()
                    }
                }
            }

            impl<'a> TryFrom<&'a mut ArrayImpl> for &'a mut $array_ty {
                type Error = ArrayError;

                fn try_from(array: &'a mut ArrayImpl) -> Result<&'a mut $array_ty>{
                    if let ArrayImpl::$variant(array) = array{
                        Ok(array)
                    }else{
                        ConvertSnafu{
                            array: array.ident(),
                            target: stringify!($array_ty),
                        }.fail()
                    }
                }
            }
        )+
    };
}

crate::macros::for_all_variants!(array_impl);
