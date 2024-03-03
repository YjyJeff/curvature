//! Visitor pattern

pub mod display;

use std::ops::ControlFlow;

/// Trait for types that can be visited
pub trait Visit {
    /// Node to visit
    type Node: ?Sized;
    /// Accept visitor to visit self
    fn accept<V: Visitor<Self>>(&self, visitor: &mut V) -> ControlFlow<V::Break>;
}

/// Trait for types that used for visiting types that implemented [`Visit`]
pub trait Visitor<V: Visit + ?Sized> {
    /// Why break
    type Break;

    /// Before visiting children of V
    fn pre_visit(&mut self, v: &V::Node) -> ControlFlow<Self::Break, ()>;

    /// After visiting children of V
    fn post_visit(&mut self, v: &V::Node) -> ControlFlow<Self::Break, ()>;
}

/// A trait used to convert [`Result`] to [`ControlFlow`]
pub trait VisitResultExt<T, E>: Sized {
    /// Convert result to control flow
    fn to_control_flow(self) -> ControlFlow<E, T>;

    /// Convert control flow to Result
    fn from_control_flow(cf: ControlFlow<E, T>) -> Self;
}

impl<T, E> VisitResultExt<T, E> for Result<T, E> {
    #[inline]
    fn to_control_flow(self) -> ControlFlow<E, T> {
        self.map_or_else(ControlFlow::Break, ControlFlow::Continue)
    }

    #[inline]
    fn from_control_flow(cf: ControlFlow<E, T>) -> Self {
        match cf {
            ControlFlow::Continue(c) => Self::Ok(c),
            ControlFlow::Break(b) => Self::Err(b),
        }
    }
}
