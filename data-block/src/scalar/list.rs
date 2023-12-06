//! ListScalar

use std::fmt::Debug;

use super::{Scalar, ScalarRef};
use crate::array::ArrayImpl;
use crate::private::Sealed;
use crate::types::PhysicalType;

/// List as a scalar
#[derive(Debug)]
pub struct ListScalar(ArrayImpl);

impl Sealed for ListScalar {}

impl Scalar for ListScalar {
    const NAME: &'static str = "List";

    const PHYSICAL_TYPE: PhysicalType = PhysicalType::List;

    type RefType<'a> = ListScalarRef<'a>;

    #[inline]
    fn as_ref(&self) -> Self::RefType<'_> {
        ListScalarRef {
            array: &self.0,
            offset: 0,
            len: self.0.len() as u32,
        }
    }
}

/// Reference to the ListScalar
#[derive(Clone, Copy)]
pub struct ListScalarRef<'a> {
    array: &'a ArrayImpl,
    offset: u32,
    len: u32,
}

impl<'a> ListScalarRef<'a> {
    /// Create a new ListScalarRef
    #[inline]
    pub fn new(array: &'a ArrayImpl, offset: u32, len: u32) -> Self {
        Self { array, offset, len }
    }
}

impl<'a> Sealed for ListScalarRef<'a> {}

impl<'a> ScalarRef<'a> for ListScalarRef<'a> {
    type OwnedType = ListScalar;

    #[inline]
    fn to_owned(self) -> Self::OwnedType {
        todo!()
    }
}

impl<'a> Debug for ListScalarRef<'a> {
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

        let list_ref = ListScalarRef::new(&array, 0, 3);
        assert_eq!(
            format!("{:?}", list_ref),
            "Float32Array { len: 3, data: [Some(3.14), Some(9.99), None]}\n"
        )
    }
}
