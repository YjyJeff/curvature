//! ListElement

use std::fmt::Debug;

use super::{Element, ElementRef};
use crate::array::ArrayImpl;
use crate::private::Sealed;
use crate::types::{LogicalType, PhysicalType};

/// List as a element
#[derive(Debug)]
pub struct ListElement(ArrayImpl);

impl ListElement {
    /// Get the child type of the list element
    pub fn child_type(&self) -> &LogicalType {
        self.0.logical_type()
    }
}

impl Sealed for ListElement {}

impl Element for ListElement {
    const NAME: &'static str = "List";

    const PHYSICAL_TYPE: PhysicalType = PhysicalType::List;

    type ElementRef<'a> = ListElementRef<'a>;

    #[inline]
    fn as_ref(&self) -> Self::ElementRef<'_> {
        ListElementRef {
            array: &self.0,
            offset: 0,
            len: self.0.len() as u32,
        }
    }

    #[inline]
    fn replace_with(&mut self, _element_ref: Self::ElementRef<'_>) {
        todo!()
    }

    #[allow(single_use_lifetimes)]
    #[inline]
    fn upcast_gat<'short, 'long: 'short>(
        long: Self::ElementRef<'long>,
    ) -> Self::ElementRef<'short> {
        long
    }
}

/// Reference to the ListElement
#[derive(Clone, Copy)]
pub struct ListElementRef<'a> {
    array: &'a ArrayImpl,
    offset: u32,
    len: u32,
}

impl<'a> ListElementRef<'a> {
    /// Create a new ListElementRef
    #[inline]
    pub fn new(array: &'a ArrayImpl, offset: u32, len: u32) -> Self {
        Self { array, offset, len }
    }
}

impl Sealed for ListElementRef<'_> {}

impl<'a> ElementRef<'a> for ListElementRef<'a> {
    type OwnedType = ListElement;

    fn to_owned(self) -> Self::OwnedType {
        todo!()
    }
}

impl Debug for ListElementRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.array
            .debug_array_slice(f, self.offset as usize, self.len as usize)
    }
}

#[cfg(test)]
mod tests {
    use crate::array::Float32Array;

    use super::*;

    #[allow(clippy::approx_constant)]
    #[test]
    fn test_debug_list() {
        let array = ArrayImpl::Float32(Float32Array::from_iter([
            Some(3.14),
            Some(9.99),
            None,
            Some(2.74),
        ]));

        let list_ref = ListElementRef::new(&array, 0, 3);
        let expect = expect_test::expect![[r#"
            Float32Array { len: 3, data: [
                Some(
                    3.14,
                ),
                Some(
                    9.99,
                ),
                None,
            ]}

        "#]];
        expect.assert_debug_eq(&list_ref);
    }
}
