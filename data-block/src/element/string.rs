//! Implementation of the String that meets the [`Umbra`] format, the string is read only
//!
//! [`Umbra`]: https://db.in.tum.de/~freitag/papers/p29-neumann-cidr20.pdf

use crate::aligned_vec::AllocType;
use crate::private::Sealed;
use crate::types::PhysicalType;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::slice::from_raw_parts;
use std::str::from_utf8_unchecked;

use super::{Element, ElementRef, ElementRefSerdeExt};
use libc::memcmp;

/// Length of the prefix
pub(crate) const PREFIX_LEN: usize = 4;
/// Inline length
pub(crate) const INLINE_LEN: usize = 12;

/// Implementation of the String that meets the [`Umbra`] format, the string is read only
///
/// This struct is totally unsafe!!! User should take care of the lifetime: **lifetime
/// of the pointed string should outlive this struct**
///
/// [`Umbra`]: https://db.in.tum.de/~freitag/papers/p29-neumann-cidr20.pdf
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct StringView<'a> {
    /// Length of the string
    ///
    /// If `length <= INLINE_LEN`, content must be inlined. Otherwise, content
    /// must be indirect
    pub length: u32,
    content: StringViewContent<'a>,
}

/// If `length >= PERFIX_LEN`, the first `PREFIX_LEN` bytes of content is always prefix
///
/// If `length <= INLINE_LEN`, remaining bytes should be padded with 0
#[repr(C)]
#[derive(Clone, Copy)]
union StringViewContent<'a> {
    /// Inlined string content
    inlined: [u8; INLINE_LEN],
    /// Indirection to the string
    indirect: PrefixAndPointer<'a>,
}

#[repr(packed)]
#[derive(Clone, Copy)]
struct PrefixAndPointer<'a> {
    _prefix: [u8; PREFIX_LEN],
    pointer: *const u8,
    /// Semantic store &'a u8
    _phantom: PhantomData<&'a u8>,
}

unsafe impl Send for PrefixAndPointer<'_> {}
unsafe impl Sync for PrefixAndPointer<'_> {}

impl<'a> StringView<'a> {
    /// Create a new inlined string
    ///
    /// Caller should guarantee `length <= INLINE_LEN`
    #[inline]
    pub(crate) fn new_inline(val: &'a str) -> StringView<'static> {
        #[cfg(feature = "verify")]
        assert!(val.len() <= INLINE_LEN);

        let mut inlined = [0; INLINE_LEN];
        // Compiler will optimize it to memcpy
        val.as_bytes()
            .iter()
            .zip(inlined.iter_mut())
            .for_each(|(&v, dst)| {
                *dst = v;
            });
        StringView {
            length: val.len() as u32,
            content: StringViewContent { inlined },
        }
    }

    /// # Safety
    ///
    /// The returned [`StringView`] with 'static lifetime is fake !!! Caller should
    /// guarantee it should only used during the input ptr is valid
    #[inline]
    pub(crate) unsafe fn new_indirect(ptr: *const u8, length: u32) -> StringView<'static> {
        #[cfg(feature = "verify")]
        assert!(length > INLINE_LEN as u32);

        // SAFETY:
        // length > INLINE_LEN, get bytes 0..PREFIX_LEN is valid
        unsafe {
            // Compiler will optimize it to load instruction
            let prefix = from_raw_parts(ptr, PREFIX_LEN)
                .get_unchecked(0..PREFIX_LEN)
                .try_into()
                .unwrap_unchecked();
            StringView {
                length,
                content: StringViewContent {
                    indirect: PrefixAndPointer {
                        _prefix: prefix,
                        pointer: ptr,
                        _phantom: PhantomData,
                    },
                },
            }
        }
    }

    /// Create a new StringView from &'static str
    #[inline]
    pub fn from_static_str(str: &'static str) -> StringView<'static> {
        if str.len() <= INLINE_LEN {
            Self::new_inline(str)
        } else {
            // Safety: str is &'static,  the returned pointer &'static ptr is not fake
            unsafe { Self::new_indirect(str.as_ptr(), str.len() as u32) }
        }
    }

    /// Check string is inlined or not
    #[inline]
    pub fn is_inlined(&self) -> bool {
        self.length <= INLINE_LEN as u32
    }

    /// Shorten the lifetime of the StringView to the lifetime of the borrow.
    ///
    /// This is pretty useful! When self has a fake lifetime 'static, we can short
    /// the lifetime to the real lifetime and let the compiler check the memory safety!
    #[inline]
    pub(crate) fn shorten(&self) -> StringView<'_> {
        // Just copy self !!!
        *self
    }

    /// Expand the lifetime to static
    ///
    /// # Safety
    ///
    /// [`StringView`] should be inlined
    ///
    /// # Allow
    ///
    /// The clippy hint is wrong!
    #[allow(clippy::unnecessary_cast)]
    pub(crate) unsafe fn expand(&self) -> StringView<'static> {
        unsafe { *((self as *const _) as *const StringView<'static>) }
    }

    /// Calibrate the old pointer to new pointer. Reallocation, serialization would invalid
    /// the old pointer, we need to calibrate it with new pointer
    ///
    /// # Safety
    ///
    /// Caller should guarantee the pointer is valid until the StringView is dropped
    #[inline]
    pub(crate) unsafe fn calibrate(&mut self, pointer: *const u8) {
        self.content.indirect.pointer = pointer;
    }

    /// Convert self to str
    #[inline]
    pub fn as_str(&self) -> &str {
        unsafe { from_utf8_unchecked(from_raw_parts(self.as_ptr(), self.length as usize)) }
    }

    #[inline]
    pub(crate) unsafe fn as_ptr(&self) -> *const u8 {
        if self.is_inlined() {
            unsafe { self.inlined_ptr() }
        } else {
            unsafe { self.indirect_ptr() }
        }
    }

    /// Inlined ptr that contains the prefix
    #[inline]
    pub(crate) unsafe fn inlined_ptr(&self) -> *const u8 {
        unsafe { self.content.inlined.as_ptr() }
    }

    #[inline]
    pub(crate) unsafe fn indirect_ptr(&self) -> *const u8 {
        unsafe { self.content.indirect.pointer }
    }

    /// Read the prefix as u32
    #[inline]
    pub(crate) fn prefix_as_u32(&self) -> u32 {
        unsafe { *(self.content.inlined.as_ptr() as *const u32) }
    }

    /// Read the length and prefix as u64
    #[inline]
    pub(crate) fn size_and_prefix_as_u64(&self) -> u64 {
        unsafe { *(self as *const _ as *const u64) }
    }

    /// Read the inlined data without prefix as u64
    #[inline]
    pub(crate) fn inlined_without_prefix_as_u64(&self) -> u64 {
        unsafe { *(self as *const _ as *const u64).add(1) }
    }
}

impl Default for StringView<'_> {
    #[inline]
    fn default() -> Self {
        Self {
            length: 0,
            content: StringViewContent {
                inlined: [0; INLINE_LEN],
            },
        }
    }
}

impl Debug for StringView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl Display for StringView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl<'a> PartialEq<StringView<'a>> for StringView<'_> {
    fn eq(&self, other: &StringView<'a>) -> bool {
        if self.size_and_prefix_as_u64() != other.size_and_prefix_as_u64() {
            return false;
        }

        if self.is_inlined() {
            return self.length <= PREFIX_LEN as u32
                || self.inlined_without_prefix_as_u64() == other.inlined_without_prefix_as_u64();
        }
        // Compare remaining
        unsafe {
            // Both of them are indirection
            memcmp(
                self.indirect_ptr().add(PREFIX_LEN) as _,
                other.indirect_ptr().add(PREFIX_LEN) as _,
                self.length as usize - PREFIX_LEN,
            ) == 0
        }
    }
}

impl Eq for StringView<'_> {}

impl<'a> PartialOrd<StringView<'a>> for StringView<'_> {
    #[inline]
    fn partial_cmp(&self, other: &StringView<'a>) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for StringView<'_> {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare the ordering of the prefix
        match self
            .prefix_as_u32()
            .to_be()
            .cmp(&other.prefix_as_u32().to_be())
        {
            std::cmp::Ordering::Equal => (),
            ordering => return ordering,
        }

        // From now on, prefix is equal
        let cmp_len = std::cmp::min(self.length, other.length) as i32 - PREFIX_LEN as i32;
        if cmp_len <= 0 {
            // One ends within the prefix
            return self.length.cmp(&other.length);
        }

        let cmp_len = cmp_len as usize;

        if self.is_inlined() && other.is_inlined() {
            // Compiler will optimize it to compare the u64 in the MSB form
            // Both of them are inlined
            let cmp = unsafe {
                memcmp(
                    self.inlined_ptr().add(PREFIX_LEN) as _,
                    other.inlined_ptr().add(PREFIX_LEN) as _,
                    INLINE_LEN - PREFIX_LEN,
                )
            };
            match cmp.cmp(&0) {
                std::cmp::Ordering::Equal => return self.length.cmp(&other.length),
                ordering => return ordering,
            }
        }

        let cmp = unsafe {
            memcmp(
                self.as_ptr().add(PREFIX_LEN) as _,
                other.as_ptr().add(PREFIX_LEN) as _,
                cmp_len,
            )
        };
        match cmp.cmp(&0) {
            std::cmp::Ordering::Equal => self.length.cmp(&other.length),
            ordering => ordering,
        }
    }
}

impl Sealed for StringView<'_> {}
impl AllocType for StringView<'static> {}

/// Scala of the String with the StringView, it owns the StringData
pub struct StringElement {
    /// The 'static lifetime is fake !!! If the view is indirect, it will points to
    /// [`Self::_data`]
    pub(crate) view: StringView<'static>,
    pub(crate) _data: Option<String>,
}

impl StringElement {
    /// Create a new StringElement from the String
    pub fn new(string: String) -> Self {
        if string.len() <= INLINE_LEN {
            Self {
                view: StringView::new_inline(&string),
                _data: None,
            }
        } else {
            Self {
                view: unsafe { StringView::new_indirect(string.as_ptr(), string.len() as _) },
                _data: Some(string),
            }
        }
    }

    /// Get view
    #[inline]
    pub fn view(&self) -> StringView<'_> {
        self.view.shorten()
    }

    /// Convert self to str
    #[inline]
    pub fn as_str(&self) -> &str {
        self.view.as_str()
    }
}

impl Debug for StringElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.view)
    }
}
impl Sealed for StringElement {}

impl Element for StringElement {
    const NAME: &'static str = "String";
    const PHYSICAL_TYPE: PhysicalType = PhysicalType::String;

    type ElementRef<'a> = StringView<'a>;

    #[inline]
    fn as_ref(&self) -> Self::ElementRef<'_> {
        self.view.shorten()
    }

    #[inline]
    fn replace_with(&mut self, element_ref: Self::ElementRef<'_>) {
        if element_ref.is_inlined() {
            // We do not need to modify the _data, just modify the view
            self.view = unsafe { element_ref.expand() };
        } else {
            let ptr = if let Some(data) = &mut self._data {
                // Reuse the memory allocated by the _data
                data.clear();
                data.push_str(element_ref.as_str());
                data.as_ptr()
            } else {
                let data = element_ref.as_str().to_string();
                let ptr = data.as_ptr();
                self._data = Some(data);
                ptr
            };
            // Update view in self
            self.view = unsafe { StringView::new_indirect(ptr, element_ref.length) };
        }
    }

    #[allow(single_use_lifetimes)]
    #[inline]
    fn upcast_gat<'short, 'long: 'short>(
        long: Self::ElementRef<'long>,
    ) -> Self::ElementRef<'short> {
        long
    }
}

impl<'a> ElementRef<'a> for StringView<'a> {
    type OwnedType = StringElement;
    #[inline]
    fn to_owned(self) -> Self::OwnedType {
        if self.is_inlined() {
            StringElement {
                view: StringView {
                    length: self.length,
                    content: StringViewContent {
                        inlined: unsafe { self.content.inlined },
                    },
                },
                _data: None,
            }
        } else {
            unsafe {
                let data =
                    from_utf8_unchecked(from_raw_parts(self.indirect_ptr(), self.length as usize))
                        .to_string();
                let view = StringView {
                    length: self.length,
                    content: StringViewContent {
                        indirect: PrefixAndPointer {
                            _prefix: self.content.indirect._prefix,
                            pointer: &*data.as_ptr(),
                            _phantom: PhantomData,
                        },
                    },
                };
                StringElement {
                    view,
                    _data: Some(data),
                }
            }
        }
    }
}

impl<'a> ElementRefSerdeExt<'a> for StringView<'a> {
    #[inline]
    fn serialize(self, buf: &mut Vec<u8>) {
        // Write length before data
        self.length.serialize(buf);
        // Write data
        buf.extend_from_slice(self.as_str().as_bytes())
    }

    #[inline]
    unsafe fn deserialize(ptr: &'a mut *const u8) -> StringView<'a> {
        unsafe {
            let length = std::ptr::read_unaligned(*ptr as *const u32);
            let data_ptr = ptr.add(std::mem::size_of::<u32>());
            let v = if length <= INLINE_LEN as u32 {
                let mut inlined = [0; INLINE_LEN];
                std::ptr::copy_nonoverlapping(data_ptr, inlined.as_mut_ptr(), length as usize);
                Self {
                    length,
                    content: StringViewContent { inlined },
                }
            } else {
                Self::new_indirect(data_ptr, length)
            };
            *ptr = ptr.add(std::mem::size_of::<u32>() + length as usize);
            v
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn test_memory_layout() {
        assert_eq!(size_of::<PrefixAndPointer<'_>>(), INLINE_LEN);
        assert_eq!(size_of::<StringViewContent<'_>>(), INLINE_LEN);
        assert_eq!(size_of::<StringView<'_>>(), 16);
        assert_eq!(align_of::<StringView<'_>>(), 16);
    }

    #[test]
    fn test_replace_string_element() {
        let s0 = "wtf";
        let s1 = "01234567891234";
        let s2 = "StringView is pretty awesome";
        let mut string_element = StringElement::new(String::new());

        let view_0 = StringView::from_static_str(s0);
        string_element.replace_with(view_0);
        assert_eq!(string_element.as_str(), s0);

        // allocate memory
        string_element.replace_with(StringView::from_static_str(s1));
        assert_eq!(string_element.as_str(), s1);
        assert!(string_element._data.is_some());

        // re-allocate memory
        string_element.replace_with(StringView::from_static_str(s2));
        assert_eq!(string_element.as_str(), s2);
        assert!(string_element._data.is_some());
    }

    #[test]
    fn test_string_view_eq() {
        fn assert_eq(lhs: &'static str, rhs: &'static str) {
            let lhs = StringView::from_static_str(lhs);
            let rhs = StringView::from_static_str(rhs);
            assert_eq!(lhs, rhs)
        }
        // Inlined
        assert_eq("abc", "abc");
        // Not inlined
        assert_eq("StringView comparison", "StringView comparison");
    }

    #[test]
    fn test_string_view_ne() {
        fn assert_ne(lhs: &'static str, rhs: &'static str) {
            let lhs = StringView::from_static_str(lhs);
            let rhs = StringView::from_static_str(rhs);
            assert_ne!(lhs, rhs)
        }

        // Same prefix, different length
        assert_ne("http://bb", "http://bbb");

        // Different prefix, same length
        assert_ne("tcp", "udp");

        // Same prefix, same size, inlined
        assert_ne("http://bb", "http://aa");

        // Same prefix, same size, not inlined
        assert_ne("Curvature is fast", "Curvature is slow");
    }

    #[test]
    fn test_string_view_cmp() {
        // Order is determined by prefix
        let lhs = StringView::from_static_str("hello");
        let rhs = StringView::from_static_str("https");
        assert!(lhs < rhs);

        // Prefix is equal, one ends within prefix
        let lhs = StringView::from_static_str("https");
        let rhs = StringView::from_static_str("http");
        assert!(lhs > rhs);

        let lhs = StringView::from_static_str("h\0\0\0ihi");
        let rhs = StringView::from_static_str("h");
        assert!(lhs > rhs);

        // Prefix is equal and both of them are inlined
        let lhs = StringView::from_static_str("haha, oops");
        let rhs = StringView::from_static_str("haha, oop");
        assert!(lhs >= rhs);

        let lhs = StringView::from_static_str("haha, oops\0\0");
        let rhs = StringView::from_static_str("haha, oops");
        assert!(lhs >= rhs);

        // Prefix is equal, one of them are inlined
        let lhs = StringView::from_static_str("Curvature is fast");
        let rhs = StringView::from_static_str("Curvature");
        assert!(rhs < lhs);

        // Prefix is equal, both of tem are not inlined
        let lhs = StringView::from_static_str("Curvature is fast");
        let rhs = StringView::from_static_str("Curvature is slow");
        assert!(rhs >= lhs);
    }
}
