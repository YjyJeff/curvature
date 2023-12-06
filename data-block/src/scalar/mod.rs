//! This module contains the scalar value of the `Array` and reference of the `Scalar`

pub mod interval;
pub mod list;
pub mod string;

use self::interval::DayTime;
use self::list::ListScalar;
use self::string::StringScalar;
use std::fmt::Debug;

use crate::macros::for_all_primitive_types;
use crate::private::Sealed;
use crate::types::PhysicalType;

/// Scalar represents the owned single value in an `Array`
pub trait Scalar: Sealed + Debug + 'static {
    /// Unique name of the Scalar,
    const NAME: &'static str;

    /// Physical type of this Scalar
    const PHYSICAL_TYPE: PhysicalType;

    /// Reference to the [`Scalar`]
    type RefType<'a>: ScalarRef<'a, OwnedType = Self>;

    /// Get the reference of the [`Scalar`]
    fn as_ref(&self) -> Self::RefType<'_>;
}

/// ScalarRef represents a reference to the [`Scalar`]
pub trait ScalarRef<'a>: Sealed + Debug + Clone + Copy + 'a {
    /// Owned type of the reference
    type OwnedType: Scalar<RefType<'a> = Self>;

    /// Convert the reference to owned scalar
    fn to_owned(self) -> Self::OwnedType;
}

macro_rules! impl_scalar_for_primitive_types {
    ($({$variant:ident, $primitive_scalar_ty:ty, $_:ident, $__:ident}),*) => {
        $(
            #[doc = concat!(
                "Implement [`Scalar`] for primitive type [`", stringify!($primitive_scalar_ty), "`]. ",
                "Note that primitive types are both [`Scalar`] and [`ScalarRef`] as they have little cost for copy.")]
            impl Scalar for $primitive_scalar_ty {

                const NAME: &'static str = stringify!($variant);

                const PHYSICAL_TYPE: PhysicalType = PhysicalType::$variant;

                type RefType<'a> = $primitive_scalar_ty;

                #[inline]
                fn as_ref(&self) -> Self::RefType<'_> {
                    *self
                }

            }

            #[doc = concat!(
                "Implement [`ScalarRef`] for primitive type [`", stringify!($primitive_scalar_ty), "`]. ",
                "Note that primitive types are both [`Scalar`] and [`ScalarRef`] as they have little cost for copy.")]
            impl<'a> ScalarRef<'a> for $primitive_scalar_ty {
                type OwnedType = $primitive_scalar_ty;

                #[inline]
                fn to_owned(self) -> Self::OwnedType {
                    self
                }
            }
        )*
    };
}

for_all_primitive_types!(impl_scalar_for_primitive_types);

impl Sealed for bool {}

impl Scalar for bool {
    const NAME: &'static str = "Boolean";
    const PHYSICAL_TYPE: PhysicalType = PhysicalType::Boolean;

    type RefType<'a> = bool;

    #[inline]
    fn as_ref(&self) -> Self::RefType<'_> {
        *self
    }
}

impl<'a> ScalarRef<'a> for bool {
    type OwnedType = bool;

    fn to_owned(self) -> Self::OwnedType {
        self
    }
}

impl Sealed for Vec<u8> {}

impl Scalar for Vec<u8> {
    const NAME: &'static str = "Binary";
    const PHYSICAL_TYPE: PhysicalType = PhysicalType::Binary;

    type RefType<'a> = &'a [u8];

    #[inline]
    fn as_ref(&self) -> Self::RefType<'_> {
        self.as_slice()
    }
}

impl<'a> Sealed for &'a [u8] {}

impl<'a> ScalarRef<'a> for &'a [u8] {
    type OwnedType = Vec<u8>;

    #[inline]
    fn to_owned(self) -> Self::OwnedType {
        self.to_vec()
    }
}

macro_rules! scalar_impl {
    ($({$variant:ident, $scalar_ty:ty, $_:ident}),+) => {
        /// Implementations of the [`Scalar`], enum dispatch
        #[derive(Debug)]
        pub enum ScalarImpl {
            $(
                #[doc = concat!("`", stringify!($scalar_ty), "`")]
                $variant($scalar_ty)
            ),+
        }
    };
}

crate::macros::for_all_variants!(scalar_impl);
