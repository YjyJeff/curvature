//! TreeNode and its visitor pattern copied from Datafusion
//!
//! See [design doc](https://synnada.notion.site/TreeNode-Design-Proposal-bceac27d18504a2085145550e267c4c1) for more details
pub mod display;

/// This macro is used to control continuation behaviors during tree traversals
/// based on the specified direction. Depending on `$DIRECTION` and the value of
/// the given expression (`$EXPR`), which should be a variant of [`TreeNodeRecursion`],
/// the macro results in the following behavior:
///
/// - If the expression returns [`TreeNodeRecursion::Continue`], normal execution
///   continues.
/// - If it returns [`TreeNodeRecursion::Stop`], recursion halts and propagates
///   [`TreeNodeRecursion::Stop`].
/// - If it returns [`TreeNodeRecursion::Jump`], the continuation behavior depends
///   on the traversal direction:
///   - For `UP` direction, the function returns with [`TreeNodeRecursion::Jump`],
///     bypassing further bottom-up closures until the next top-down closure.
///   - For `DOWN` direction, the function returns with [`TreeNodeRecursion::Continue`],
///     skipping further exploration.
///   - If no direction is specified, `Jump` is treated like `Continue`.
#[macro_export]
macro_rules! handle_visit_recursion {
    // Internal helper macro for handling the `Jump` case based on the direction:
    (@handle_jump UP) => {
        return Ok(TreeNodeRecursion::Jump)
    };
    (@handle_jump DOWN) => {
        return Ok(TreeNodeRecursion::Continue)
    };
    (@handle_jump) => {
        {} // Treat `Jump` like `Continue`, do nothing and continue execution.
    };

    // Main macro logic with variables to handle directionality.
    ($EXPR:expr $(, $DIRECTION:ident)?) => {
        match $EXPR {
            TreeNodeRecursion::Continue => {}
            TreeNodeRecursion::Jump => handle_visit_recursion!(@handle_jump $($DIRECTION)?),
            TreeNodeRecursion::Stop => return Ok(TreeNodeRecursion::Stop),
        }
    };
}

pub use handle_visit_recursion;

/// Trait for visitable node in the tree.
pub trait TreeNode {
    /// Accept visitor to visit self
    fn visit<V: Visitor<Self>>(&self, visitor: &mut V) -> Result<TreeNodeRecursion, V::Error> {
        match visitor.pre_visit(self)? {
            TreeNodeRecursion::Continue => {
                handle_visit_recursion!(
                    self.visit_children::<V, _>(&mut |node| node.visit(visitor))?,
                    UP
                );
                visitor.post_visit(self)
            }
            TreeNodeRecursion::Jump => visitor.post_visit(self),
            TreeNodeRecursion::Stop => Ok(TreeNodeRecursion::Stop),
        }
    }

    /// Visit children of the tree node
    fn visit_children<V, F>(&self, f: &mut F) -> Result<TreeNodeRecursion, V::Error>
    where
        V: Visitor<Self>,
        F: FnMut(&Self) -> Result<TreeNodeRecursion, V::Error>;
}

/// Trait for types that used for visiting types that implemented [`TreeNode`]
pub trait Visitor<V: TreeNode + ?Sized> {
    /// Error that could happen in the visit. If error is returned, stop visiting
    type Error;

    /// Before visiting children of V
    fn pre_visit(&mut self, v: &V) -> Result<TreeNodeRecursion, Self::Error>;

    /// After visiting children of V
    fn post_visit(&mut self, v: &V) -> Result<TreeNodeRecursion, Self::Error>;
}

/// Controls how [`TreeNode`] recursions should proceed.
#[must_use = "TreeNodeRecursion controls the visit recursion"]
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum TreeNodeRecursion {
    /// Continue recursion with the next node.
    Continue,
    /// In top-down traversals, skip recursing into children but continue with
    /// the next node, which actually means pruning of the subtree.
    ///
    /// In bottom-up traversals, bypass calling bottom-up closures till the next
    /// leaf node.
    ///
    /// In combined traversals, if it is the `pre_visit` phase, execution
    /// "jumps" to the next `post_visit` phase by shortcutting its children.
    /// If it is the `post_visit` phase, execution "jumps" to the next `pre_visit`
    /// phase by shortcutting its parent nodes until the first parent node
    /// having unvisited children path.
    Jump,
    /// Stop recursion.
    Stop,
}
