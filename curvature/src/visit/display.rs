//! Display

use super::{Visit, VisitResultExt, Visitor};
use std::fmt::{Debug, Display, Error, Formatter, Result};
use std::ops::ControlFlow;

/// Indent visitor for displaying
struct IndentDisplayVisitor<'a, 'b> {
    f: &'a mut Formatter<'b>,
    /// Current indent
    indent: usize,
}

impl<'a, 'b, V, N> Visitor<V> for IndentDisplayVisitor<'a, 'b>
where
    V: Visit<Node = N> + ?Sized,
    N: Display + ?Sized,
{
    type Break = Error;

    fn pre_visit(&mut self, v: &N) -> ControlFlow<Self::Break, ()> {
        write!(self.f, "{:indent$}", "", indent = self.indent * 2).to_control_flow()?;
        write!(self.f, "{}", v).to_control_flow()?;
        writeln!(self.f).to_control_flow()?;
        self.indent += 1;
        ControlFlow::Continue(())
    }

    fn post_visit(&mut self, _v: &N) -> ControlFlow<Self::Break, ()> {
        ControlFlow::Continue(())
    }
}

/// Wrapper for types that need to be displayed with indent
pub struct IndentDisplayWrapper<'a, T: ?Sized>(&'a T);

impl<'a, T, N> IndentDisplayWrapper<'a, T>
where
    T: Visit<Node = N> + ?Sized,
    N: Display + ?Sized,
{
    /// Create a new [`IndentDisplayWrapper`] with the type that implement [`Visit`]
    /// and its `Node` implements [`Display`]. The [`IndentDisplayWrapper`] implements
    /// the [`Display`] trait, display this wrapper will pretty print the inner type
    /// with indent
    pub fn new(v: &'a T) -> Self {
        Self(v)
    }
}

impl<'a, T> Debug for IndentDisplayWrapper<'a, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "IndentDisplayWrapper({})", std::any::type_name::<T>())
    }
}

impl<'a, T, N> Display for IndentDisplayWrapper<'a, T>
where
    T: Visit<Node = N> + ?Sized,
    N: Display + ?Sized,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let mut visitor = IndentDisplayVisitor { f, indent: 0 };
        Result::from_control_flow(self.0.accept(&mut visitor))
    }
}
