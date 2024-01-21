//! [`ConstantArray`] contains a single constant. Although its length is always `1`,
//! get any index from the [`ConstantArray`] is valid!

use std::fmt::Debug;
// use std::iter::{once, Once};

use crate::bitmap::Bitmap;
use crate::element::Element;
use crate::private::Sealed;

// use super::Array;

/// An array contains a single constant Element
pub struct ConstantArray<T> {
    data: T,
    validity: Bitmap,
}

impl<T: Element> Debug for ConstantArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}ConstantArray{{ ", T::NAME)?;
        if unsafe { self.validity.get_unchecked(0) } {
            write!(f, "Some({:?}) }}", self.data)
        } else {
            write!(f, "None }}")
        }
    }
}

impl<T: Element> Sealed for ConstantArray<T> {}

// impl<T: Element> Array for ConstantArray<T> {
//     type Element = T;

//     type ValuesIter<'a> = Once<T::ElementRef<'a>>;

//     #[inline]
//     fn values_iter(&self) -> Self::ValuesIter<'_> {
//         once(self.data.as_ref())
//     }

//     #[inline]
//     fn validity(&self) -> &Bitmap {
//         &self.validity
//     }

//     #[inline]
//     fn len(&self) -> usize {
//         1
//     }

//     /// Not matter what index is passed, we always return the ElementRef
//     #[inline]
//     unsafe fn get_value_unchecked(
//         &self,
//         _index: usize,
//     ) -> <Self::Element as Element>::ElementRef<'_> {
//         self.data.as_ref()
//     }
// }
