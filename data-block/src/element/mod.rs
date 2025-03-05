//! This module contains the element value of the `Array` and reference of the `Element`

pub mod interval;
pub mod list;
pub mod string;

use self::interval::Interval;
use self::list::{ListElement, ListElementRef};
use self::string::{StringElement, StringView};
use std::fmt::{Debug, Display};

use crate::for_all_variants;
use crate::macros::for_all_primitive_types;
use crate::private::Sealed;
use crate::types::{PhysicalType, PrimitiveType};

/// Element represents the owned single value in an `Array`
pub trait Element: Sealed + Debug + 'static {
    /// Unique name of the Element
    const NAME: &'static str;

    /// Physical type of this Element
    const PHYSICAL_TYPE: PhysicalType;

    /// Reference to the [`Element`]
    type ElementRef<'a>: ElementRef<'a, OwnedType = Self>;

    /// Get the reference of the [`Element`]
    fn as_ref(&self) -> Self::ElementRef<'_>;

    /// Replace self with the element ref, used in the aggregation to avoid
    /// repeated memory allocation
    fn replace_with(&mut self, element_ref: Self::ElementRef<'_>);

    /// Convert long lifetime to short lifetime. Manually covariance
    #[allow(single_use_lifetimes)]
    fn upcast_gat<'short, 'long: 'short>(long: Self::ElementRef<'long>)
    -> Self::ElementRef<'short>;
}

/// ElementRef represents a reference to the [`Element`]
pub trait ElementRef<'a>: Sealed + Debug + Clone + Copy + 'a {
    /// Owned type of the reference
    type OwnedType: Element<ElementRef<'a> = Self>;

    /// Convert the reference to owned Element
    fn to_owned(self) -> Self::OwnedType;
}

/// Extension of the ElementRef, use to serialize and deserialize ElementRef
pub trait ElementRefSerdeExt<'a>: ElementRef<'a> {
    /// Serialize self into the buf. The serialized bytes should not be ambiguous,
    /// which means that we can deserialize the bytes to get `Self` and
    /// we can compare the equivalency of the bytes to get the equivalency of the
    /// element. Implementation should serialize self into the end of the buf, which
    /// means that we should extend the buf in stead of write self to the start of
    /// the buf
    fn serialize(self, buf: &mut Vec<u8>);

    /// Deserialize the element ref from buf with the start address. The lifetime
    /// of the deserialized ElementRef is the lifetime of the buf. Implementation
    /// should also advance the ptr such that we can deserialize other element
    ///
    /// # Safety
    ///
    /// Caller should guarantee the buf has adequate space to store `Self`.
    /// Otherwise, undefined behavior happens
    unsafe fn deserialize(ptr: &'a mut *const u8) -> Self;
}

macro_rules! impl_element_for_primitive_types {
    ($({$variant:ident, $primitive_element_ty:ty, $_:ident, $__:ident, $___:path}),*) => {
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

                #[inline]
                fn replace_with(&mut self, element_ref: Self::ElementRef<'_>){
                    *self = element_ref;
                }

                #[allow(single_use_lifetimes)]
                #[inline]
                fn upcast_gat<'short, 'long: 'short>(
                    long: Self::ElementRef<'long>,
                ) -> Self::ElementRef<'short> {
                    long
                }

            }

            #[doc = concat!(
                "Implement [`ElementRef`] for primitive type [`", stringify!($primitive_element_ty), "`]. ",
                "Note that primitive types are both [`Element`] and [`ElementRef`] as they have little cost for copy.")]
            impl ElementRef<'_> for $primitive_element_ty {
                type OwnedType = $primitive_element_ty;

                #[inline]
                fn to_owned(self) -> Self::OwnedType {
                    self
                }

            }

            impl ElementRefSerdeExt<'_> for $primitive_element_ty {
                #[inline]
                #[allow(clippy::size_of_in_element_count)]
                fn serialize(self, buf: &mut Vec<u8>){
                    let normalized = <$primitive_element_ty>::NORMALIZE_FUNC(self);
                    unsafe{
                        buf.extend_from_slice(std::slice::from_raw_parts(
                            &normalized as *const _ as *const u8,
                            std::mem::size_of::<$primitive_element_ty>()
                        ))
                    }
                }

                #[inline]
                unsafe fn deserialize(ptr: &mut *const u8) -> Self { unsafe {
                    let v = std::ptr::read_unaligned(*ptr as *const _);
                    *ptr = ptr.add(std::mem::size_of::<$primitive_element_ty>());
                    v
                }}
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

    #[inline]
    fn replace_with(&mut self, element_ref: Self::ElementRef<'_>) {
        *self = element_ref;
    }

    #[allow(single_use_lifetimes)]
    #[inline]
    fn upcast_gat<'short, 'long: 'short>(
        long: Self::ElementRef<'long>,
    ) -> Self::ElementRef<'short> {
        long
    }
}

impl ElementRef<'_> for bool {
    type OwnedType = bool;

    #[inline]
    fn to_owned(self) -> Self::OwnedType {
        self
    }
}

impl ElementRefSerdeExt<'_> for bool {
    #[inline]
    fn serialize(self, buf: &mut Vec<u8>) {
        buf.push(self as u8)
    }

    #[inline]
    unsafe fn deserialize(ptr: &mut *const u8) -> Self {
        unsafe {
            let v = **ptr != 0;
            *ptr = ptr.add(1);
            v
        }
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

    #[inline]
    fn replace_with(&mut self, element_ref: Self::ElementRef<'_>) {
        self.clear();
        self.extend_from_slice(element_ref)
    }

    #[allow(single_use_lifetimes)]
    #[inline]
    fn upcast_gat<'short, 'long: 'short>(
        long: Self::ElementRef<'long>,
    ) -> Self::ElementRef<'short> {
        long
    }
}

impl Sealed for &'_ [u8] {}

impl<'a> ElementRef<'a> for &'a [u8] {
    type OwnedType = Vec<u8>;

    #[inline]
    fn to_owned(self) -> Self::OwnedType {
        self.to_vec()
    }
}

impl<'a> ElementRefSerdeExt<'a> for &'a [u8] {
    #[inline]
    fn serialize(self, buf: &mut Vec<u8>) {
        // Write length before data
        let len = self.len() as u32;
        len.serialize(buf);

        // write data
        buf.extend_from_slice(self)
    }

    #[inline]
    unsafe fn deserialize(ptr: &'a mut *const u8) -> &'a [u8] {
        unsafe {
            let length = std::ptr::read_unaligned(*ptr as *const u32);
            let data_ptr = ptr.add(std::mem::size_of::<u32>());
            let v = std::slice::from_raw_parts(data_ptr, length as usize);
            *ptr = ptr.add(length as usize + std::mem::size_of::<u32>());
            v
        }
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

        impl ElementImpl {
            /// Get physical type of the element
            pub fn physical_type(&self) -> PhysicalType {
                match self {
                    $(
                        Self::$variant(_) => <$element_ty>::PHYSICAL_TYPE,
                    )+
                }
            }
        }
    };
}

crate::macros::for_all_variants!(element_impl);

/// Reference to the [`ElementImpl`]
///
/// FIXME:
/// - include it in macro
/// - lost logical info, can not display it
#[derive(Debug, PartialEq)]
pub enum ElementImplRef<'a> {
    /// Int8
    Int8(i8),
    /// Uint8
    UInt8(u8),
    /// Int16
    Int16(i16),
    /// Uint16
    UInt16(u16),
    /// Int32
    Int32(i32),
    /// UInt32
    UInt32(u32),
    /// Int64
    Int64(i64),
    /// Uint64
    UInt64(u64),
    /// Float32
    Float32(f32),
    /// Float64
    Float64(f64),
    /// Int128
    Int128(i128),
    /// Interval
    Interval(Interval),
    /// StringView
    String(StringView<'a>),
    /// Binary blob
    Binary(&'a [u8]),
    /// Boolean
    Boolean(bool),
    /// List
    List(ListElementRef<'a>),
}

macro_rules! impl_element_impl_ref {
    ($({$variant:ident, $_:ty, $__:ident}),+) => {
        impl Display for ElementImplRef<'_> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $(
                        // FIXME: Display instead of Debug
                        Self::$variant(element) => write!(f, "{:?}", element),
                    )+
                }
            }
        }
    };
}

for_all_variants!(impl_element_impl_ref);
