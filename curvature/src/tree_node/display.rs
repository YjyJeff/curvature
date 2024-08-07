//! Display

use super::{TreeNode, TreeNodeRecursion, TreeNodeVisitor};
use std::fmt::{Debug, Display, Error, Formatter};

/// Indent visitor for displaying
struct IndentDisplayVisitor<'a, 'b> {
    f: &'a mut Formatter<'b>,
    /// Current indent
    indent: usize,
}

impl<Node> TreeNodeVisitor<Node> for IndentDisplayVisitor<'_, '_>
where
    Node: TreeNode + Display + ?Sized,
{
    type Error = Error;

    fn f_down(&mut self, node: &Node) -> Result<TreeNodeRecursion, Error> {
        write!(self.f, "{:indent$}", "", indent = self.indent * 2)?;
        write!(self.f, "{}", node)?;
        writeln!(self.f)?;
        self.indent += 1;
        Ok(TreeNodeRecursion::Continue)
    }

    fn f_up(&mut self, _node: &Node) -> Result<TreeNodeRecursion, Error> {
        self.indent -= 1;
        Ok(TreeNodeRecursion::Continue)
    }
}

/// Wrapper for types that need to be displayed with indent
pub struct IndentDisplayWrapper<'a, T: ?Sized>(&'a T);

impl<'a, T> IndentDisplayWrapper<'a, T>
where
    T: TreeNode + Display + ?Sized,
{
    /// Create a new [`IndentDisplayWrapper`] with the type that implement [`TreeNode`]
    /// and its `Node` implements [`Display`]. The [`IndentDisplayWrapper`] implements
    /// the [`Display`] trait, display this wrapper will pretty print the inner type
    /// with indent
    pub fn new(v: &'a T) -> Self {
        Self(v)
    }
}

impl<T> Debug for IndentDisplayWrapper<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "IndentDisplayWrapper({})", std::any::type_name::<T>())
    }
}

impl<T> Display for IndentDisplayWrapper<'_, T>
where
    T: TreeNode + Display + ?Sized,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut visitor = IndentDisplayVisitor { f, indent: 0 };
        self.0.visit(&mut visitor).map(|_vr| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockNode {
        name: &'static str,
        children: Vec<MockNode>,
    }

    impl TreeNode for MockNode {
        fn apply_children<E, F: FnMut(&Self) -> Result<TreeNodeRecursion, E>>(
            &self,
            mut f: F,
        ) -> Result<TreeNodeRecursion, E> {
            for child in &self.children {
                match f(child)? {
                    TreeNodeRecursion::Continue | TreeNodeRecursion::Jump => (),
                    TreeNodeRecursion::Stop => return Ok(TreeNodeRecursion::Jump),
                }
            }

            Ok(TreeNodeRecursion::Continue)
        }
    }

    impl Display for MockNode {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.name)
        }
    }

    #[test]
    fn test_display_mock_node() {
        let node = MockNode {
            name: "a",
            children: vec![
                MockNode {
                    name: "b",
                    children: vec![],
                },
                MockNode {
                    name: "c",
                    children: vec![MockNode {
                        name: "d",
                        children: vec![],
                    }],
                },
            ],
        };

        let expect = expect_test::expect![[r#"
            a
              b
              c
                d
        "#]];
        expect.assert_eq(&IndentDisplayWrapper::new(&node).to_string());
    }
}
