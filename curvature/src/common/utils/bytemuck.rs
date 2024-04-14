//! Simple constrained bytemuck

/// A trait which indicates that a type is a `#[repr(transparent)]` wrapper around the
/// Inner value.
///
/// # Safety
///
/// See [bytemuck](https://docs.rs/bytemuck/latest/bytemuck/trait.TransparentWrapper.html#safety)
/// for details
pub unsafe trait TransparentWrapper<Inner: Sized>: Sized {
    /// Convert a mutable reference to the wrapper type into a mutable reference to the inner type.
    #[inline]
    fn peel_mut(&mut self) -> &mut Inner {
        unsafe { std::mem::transmute(self) }
    }
}
