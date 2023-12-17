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

use self::iter::ArrayIter;
use crate::bitmap::Bitmap;
use crate::private::Sealed;
use crate::scalar::Scalar;
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
#[snafu(display(
    "Invalid logical type `{:?}({})` passed to creating a new array `{}` that has `{}`",
    logical_type,
    logical_type.physical_type(),
    array_name,
    array_physical_type
))]
pub struct InvalidLogicalTypeError {
    /// In ideal, array_name should be &'static str. However, we can not create a
    /// &'static str for `PrimitiveArray<T>` currently. An ugly [solution] can be
    /// found here. Keep an eye on this [issue]
    ///
    /// [solution]: https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=51b8c210bc6f51afffa682be99e0476b
    /// [issue]: https://github.com/rodrimati1992/const_format_crates/issues/48
    array_name: String,
    array_physical_type: PhysicalType,
    logical_type: LogicalType,
}

type Result<T> = std::result::Result<T, InvalidLogicalTypeError>;

/// A trait over all arrays
pub trait Array: Sealed + Debug + 'static + Sized {
    /// Scalar type of the [`Array`]
    type ScalarType: Scalar;

    /// Iterator of the values in the [`Array`], it returns the reference to the scalar
    type ValuesIter<'a>: Iterator<Item = <Self::ScalarType as Scalar>::RefType<'a>>;

    /// Get the iterator of values in the [`Array`]
    fn values_iter(&self) -> Self::ValuesIter<'_>;

    /// Get the iterator of values in the array from offset with given length
    ///
    /// offset + length <= self.len()
    fn values_slice_iter(&self, offset: usize, length: usize) -> Self::ValuesIter<'_>;

    /// Get the validity slice. If the validity is not empty, the length must equal to
    /// [`Self::len`]
    fn validity(&self) -> &Bitmap;

    /// Get the number of elements in the [`Array`]
    fn len(&self) -> usize;

    /// Returns `true` if the [`Array`] contains no elements
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a reference to the scalar at the given index without bound check
    ///
    /// # Safety
    /// Caller should guarantee `index < self.len()`, otherwise, [undefined behavior] happens
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    unsafe fn get_value_unchecked(&self, index: usize)
        -> <Self::ScalarType as Scalar>::RefType<'_>;

    /// Get iterator of the array
    #[inline]
    fn iter(&self) -> ArrayIter<'_, Self> {
        let validity = self.validity();
        let validity = if validity.is_empty() {
            None
        } else {
            Some(validity.iter())
        };
        ArrayIter {
            values_iter: self.values_iter(),
            validity,
        }
    }

    /// Get the iterator of elements in the array from offset with given length
    ///
    /// offset + length <= self.len()
    fn slice_iter(&self, offset: usize, length: usize) -> ArrayIter<'_, Self> {
        let validity = self.validity();
        let validity = if validity.is_empty() {
            None
        } else {
            Some(validity.iter())
        };
        ArrayIter {
            values_iter: self.values_slice_iter(offset, length),
            validity,
        }
    }

    /// Get the logical type of the array. This function enforces each array implementation
    /// should have a [`LogicalType`] as its field
    fn logical_type(&self) -> &LogicalType;
}

/// Extension for array
pub trait ArrayExt: Array {
    /// Reference self to other array
    fn reference(&mut self, other: &Self);
}

macro_rules! array_impl {
    ($({$variant:ident, $scalar_ty:ty, $array_ty:ident}),+) => {
        /// Implementations of the [`Array`], enum dispatch
        #[derive(Debug)]
        pub enum ArrayImpl {
            $(
                #[doc = concat!("Array of `", stringify!($scalar_ty), "`")]
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

            /// Return the [`LogicalType`] of the array
            pub fn logical_type(&self) -> &LogicalType{
                match self{
                    $(
                        Self::$variant(array) => array.logical_type(),
                    )+
                }
            }

            /// Debug the array slice
            ///
            /// start + len <= self.len()
            pub fn debug_array_slice(
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
        }
    };
}

crate::macros::for_all_variants!(array_impl);
