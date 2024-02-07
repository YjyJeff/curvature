//! [`Array`] is the memory format of the columnar storage.
//!
//! Heavily adapted from [`type-exercise-in-rust`](https://github.com/skyzh/type-exercise-in-rust)

pub mod binary;
pub mod boolean;
pub mod constant;
pub mod iter;
pub mod list;
pub mod ping_pong;
pub mod primitive;
pub mod string;
pub mod utils;

use self::iter::ArrayIter;
use self::ping_pong::PingPongPtr;
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
    #[snafu(display("Can not convert `ArrayImpl::{array}` array into `{target}` array"))]
    Convert {
        array: &'static str,
        target: &'static str,
    },
}

type Result<T> = std::result::Result<T, ArrayError>;

/// A trait over all arrays
pub trait Array: Sealed + Debug + 'static + Sized {
    /// Element of the [`Array`]
    type Element: Element;

    /// Iterator of the values in the [`Array`], it returns the reference to the element
    type ValuesIter<'a>: Iterator<Item = <Self::Element as Element>::ElementRef<'a>>;

    /// Get the iterator of values in the [`Array`]
    fn values_iter(&self) -> Self::ValuesIter<'_>;

    /// Get the iterator of values in the array from offset with given length
    ///
    /// offset + length <= self.len()
    fn values_slice_iter(&self, offset: usize, length: usize) -> Self::ValuesIter<'_>;

    /// Get the validity. If the validity is not empty, the length must equal to
    /// [`Self::len`]
    fn validity(&self) -> &Bitmap;

    /// Get the mutable validity.
    fn validity_mut(&mut self) -> &mut PingPongPtr<Bitmap>;

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
        if bitmap.is_empty() {
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
}

/// Extension for mutate array
pub trait MutateArrayExt: Array {
    /// Reference self to other array
    fn reference(&mut self, other: &Self);
}

/// Trait for arrays that element is scalar type
pub trait ScalarArray: Array {
    /// Replace the array with the trusted_len values iterator that has `len` items
    ///
    /// # Safety
    ///
    /// - The `trusted_len_iterator` must has `len` items
    ///
    /// - Satisfy the mutate condition
    unsafe fn replace_with_trusted_len_values_iterator(
        &mut self,
        len: usize,
        trusted_len_iterator: impl Iterator<Item = Self::Element>,
    );

    /// Replace the array with the trusted_len iterator that has `len` items
    ///
    /// # Safety
    ///
    /// - The `trusted_len_iterator` must has `len` items
    ///
    /// - Satisfy the mutate condition
    unsafe fn replace_with_trusted_len_iterator(
        &mut self,
        len: usize,
        trusted_len_iterator: impl Iterator<Item = Option<Self::Element>>,
    );
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
