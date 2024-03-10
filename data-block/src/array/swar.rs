//! Shared writable and readable pointer

use std::cell::{Ref, RefCell, RefMut};
use std::fmt::Debug;
use std::rc::Rc;

/// Shared writable and readable pointer designed for query execution
pub struct SwarPtr<T> {
    /// Reference to other data or not
    reference: Option<Rc<RefCell<T>>>,
    /// Owned data
    owned: Rc<RefCell<T>>,
}

impl<T: Default> Default for SwarPtr<T> {
    #[inline]
    fn default() -> Self {
        Self {
            reference: None,
            owned: Rc::new(RefCell::new(T::default())),
        }
    }
}

impl<T: Debug> Debug for SwarPtr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SwarPtr {{")?;
        match &self.reference {
            Some(reference) => {
                write!(f, "Reference: {:?} }}", reference)
            }
            None => {
                write!(f, "Owned: {:?} }}", self.owned)
            }
        }
    }
}

impl<T: Default> SwarPtr<T> {
    /// Create a new SwarPtr with value
    pub fn new(inner: T) -> Self {
        Self {
            reference: None,
            owned: Rc::new(RefCell::new(inner)),
        }
    }
}

impl<T> SwarPtr<T> {
    /// Create a new SwarPtr with constructor
    pub fn with_constructor<F: Fn() -> T>(constructor: F) -> Self {
        Self {
            reference: None,
            owned: Rc::new(RefCell::new(constructor())),
        }
    }

    /// Reference self to other SwarPtr
    #[inline]
    pub fn reference(&mut self, other: &Self) {
        let refrence = match &other.reference {
            Some(reference) => Rc::clone(reference),
            None => Rc::clone(&other.owned),
        };
        self.reference = Some(refrence);
    }

    /// Mutably borrows the wrapped value.
    ///
    /// # Panics
    ///
    /// Panics if the owed data is currently borrowed, in the  query execution,
    /// it should never happens! All of the data is written once, then read
    /// multiple times
    #[inline]
    pub fn borrow_mut(&mut self) -> RefMut<'_, T> {
        self.reference = None;
        self.owned.borrow_mut()
    }

    /// Immutably borrows the wrapped value.
    ///
    /// # Panics
    ///
    /// Panics if the value is currently mutably borrowed.
    #[inline]
    pub fn borrow(&self) -> Ref<'_, T> {
        match &self.reference {
            Some(reference) => reference.borrow(),
            None => self.owned.borrow(),
        }
    }
}
