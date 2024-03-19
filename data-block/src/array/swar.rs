//! Shared writable and readable pointer

use std::fmt::Debug;
use std::ops::Deref;
use std::ptr::NonNull;

/// Shared writable and readable pointer designed for query execution
pub struct SwarPtr<T> {
    /// Reference to other data or not
    reference: Option<NonNull<T>>,
    /// Owned data
    owned: NonNull<T>,
}

impl<T: Default> Default for SwarPtr<T> {
    #[inline]
    fn default() -> Self {
        Self {
            reference: None,
            owned: unsafe { NonNull::new_unchecked(Box::into_raw(Box::default())) },
        }
    }
}

impl<T: Debug> Debug for SwarPtr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SwarPtr {{")?;
        match self.reference {
            Some(reference) => {
                write!(f, "Reference: {:?} }}", unsafe { reference.as_ref() })
            }
            None => {
                write!(f, "Owned: {:?} }}", unsafe { self.owned.as_ref() })
            }
        }
    }
}

impl<T: Default> SwarPtr<T> {
    /// Create a new SwarPtr with value
    pub fn new(inner: T) -> Self {
        let owned = Box::new(inner);
        Self {
            reference: None,
            owned: unsafe { NonNull::new_unchecked(Box::into_raw(owned)) },
        }
    }
}

impl<T> Deref for SwarPtr<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe {
            match self.reference {
                Some(reference) => reference.as_ref(),
                None => self.owned.as_ref(),
            }
        }
    }
}

impl<T> SwarPtr<T> {
    /// Create a new SwarPtr with constructor
    pub fn with_constructor<F: Fn() -> T>(constructor: F) -> Self {
        let owned = Box::new(constructor());
        Self {
            reference: None,
            owned: unsafe { NonNull::new_unchecked(Box::into_raw(owned)) },
        }
    }

    /// Reference self to other SwarPtr
    #[inline]
    pub fn reference(&mut self, other: &Self) {
        let refrence = match other.reference {
            Some(reference) => reference,
            None => other.owned,
        };
        self.reference = Some(refrence);
    }

    /// Mutably borrows the wrapped value.
    ///
    /// # Safety
    /// You must enforce Rust’s aliasing rules. In particular, while this reference exists,
    /// the memory the pointer points to must not get accessed (read or written) through
    /// any other pointer.
    #[inline]
    pub unsafe fn as_mut(&mut self) -> &mut T {
        self.reference = None;
        self.owned.as_mut()
    }
}

/// Drop the owned memory
impl<T> Drop for SwarPtr<T> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            drop(Box::from_raw(self.owned.as_ptr()));
        }
    }
}
