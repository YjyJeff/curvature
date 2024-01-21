//! This module contains the element value of the `Array` and reference of the `Element`

pub mod interval;
pub mod list;
pub mod string;

use self::interval::DayTime;
use self::list::ListElement;
use self::string::StringElement;
use std::fmt::Debug;

use crate::macros::for_all_primitive_types;
use crate::private::Sealed;
use crate::types::PhysicalType;

/// Element represents the owned single value in an `Array`
pub trait Element: Sealed + Debug + 'static {
    /// Unique name of the Element,
    const NAME: &'static str;

    /// Physical type of this Element
    const PHYSICAL_TYPE: PhysicalType;

    /// Reference to the [`Element`]
    type ElementRef<'a>: ElementRef<'a, OwnedType = Self>;

    /// Get the reference of the [`Element`]
    fn as_ref(&self) -> Self::ElementRef<'_>;
}

/// ElementRef represents a reference to the [`Element`]
pub trait ElementRef<'a>: Sealed + Debug + Clone + Copy + 'a {
    /// Owned type of the reference
    type OwnedType: Element<ElementRef<'a> = Self>;

    /// Convert the reference to owned Element
    fn to_owned(self) -> Self::OwnedType;
}

macro_rules! impl_element_for_primitive_types {
    ($({$variant:ident, $primitive_element_ty:ty, $_:ident, $__:ident}),*) => {
        $(
            #[doc = concat!(
                "Implement [`Element`] for primitive type [`", stringify!($primitive_element_ty), "`]. ",
                "Note that primitive types are both [`Element`] and [`ElementRef`] as they have little cost for copy.")]
            impl Element for $primitive_element_ty {

                const NAME: &'static str = stringify!($variant);

                const PHYSICAL_TYPE: PhysicalType = PhysicalType::$variant;

                type ElementRef<'a> = $primitive_element_ty;

                #[inline]
                fn as_ref(&self) -> Self::ElementRef<'_> {
                    *self
                }

            }

            #[doc = concat!(
                "Implement [`ElementRef`] for primitive type [`", stringify!($primitive_element_ty), "`]. ",
                "Note that primitive types are both [`Element`] and [`ElementRef`] as they have little cost for copy.")]
            impl<'a> ElementRef<'a> for $primitive_element_ty {
                type OwnedType = $primitive_element_ty;

                #[inline]
                fn to_owned(self) -> Self::OwnedType {
                    self
                }
            }
        )*
    };
}

for_all_primitive_types!(impl_element_for_primitive_types);

impl Sealed for bool {}

impl Element for bool {
    const NAME: &'static str = "Boolean";
    const PHYSICAL_TYPE: PhysicalType = PhysicalType::Boolean;

    type ElementRef<'a> = bool;

    #[inline]
    fn as_ref(&self) -> Self::ElementRef<'_> {
        *self
    }
}

impl<'a> ElementRef<'a> for bool {
    type OwnedType = bool;

    fn to_owned(self) -> Self::OwnedType {
        self
    }
}

impl Sealed for Vec<u8> {}

impl Element for Vec<u8> {
    const NAME: &'static str = "Binary";
    const PHYSICAL_TYPE: PhysicalType = PhysicalType::Binary;

    type ElementRef<'a> = &'a [u8];

    #[inline]
    fn as_ref(&self) -> Self::ElementRef<'_> {
        self.as_slice()
    }
}

impl<'a> Sealed for &'a [u8] {}

impl<'a> ElementRef<'a> for &'a [u8] {
    type OwnedType = Vec<u8>;

    #[inline]
    fn to_owned(self) -> Self::OwnedType {
        self.to_vec()
    }
}

macro_rules! element_impl {
    ($({$variant:ident, $element_ty:ty, $_:ident}),+) => {
        /// Implementations of the [`Element`], enum dispatch
        #[derive(Debug)]
        pub enum ElementImpl {
            $(
                #[doc = concat!("`", stringify!($element_ty), "`")]
                $variant($element_ty)
            ),+
        }
    };
}

crate::macros::for_all_variants!(element_impl);
