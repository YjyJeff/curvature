//! TreeNode and its visitor pattern copied from Datafusion
//!
//! See [design doc](https://synnada.notion.site/TreeNode-Design-Proposal-bceac27d18504a2085145550e267c4c1) for more details
pub mod display;

/// TODO: Implement transform
///
/// API for inspecting and rewriting tree data structures.
///
/// The `TreeNode` API is used to express algorithms separately from traversing
/// the structure of `TreeNode`s, avoiding substantial code duplication.
///
/// # Overview
/// There are three categories of TreeNode APIs:
///
/// 1. "Inspecting" APIs to traverse a tree of `&TreeNodes`:
/// [`apply`], [`visit`], [`exists`].
///
/// 2. "Transforming" APIs that traverse and consume a tree of `TreeNode`s
/// producing possibly changed `TreeNode`s: [`transform`], [`transform_up`],
/// [`transform_down`], [`transform_down_up`], and [`rewrite`].
///
/// 3. Internal APIs used to implement the `TreeNode` API: [`apply_children`],
/// and [`map_children`].
///
/// | Traversal Order | Inspecting | Transforming |
/// | --- | --- | --- |
/// | top-down | [`apply`], [`exists`] | [`transform_down`]|
/// | bottom-up | | [`transform`] , [`transform_up`]|
/// | combined with separate `f_down` and `f_up` closures | | [`transform_down_up`] |
/// | combined with `f_down()` and `f_up()` in an object | [`visit`]  | [`rewrite`] |
///
/// **Note**:while there is currently no in-place mutation API that uses `&mut
/// TreeNode`, the transforming APIs are efficient and optimized to avoid
/// cloning.
///
/// [`apply`]: Self::apply
/// [`visit`]: Self::visit
/// [`exists`]: Self::exists
/// [`transform`]: Self::transform
/// [`transform_up`]: Self::transform_up
/// [`transform_down`]: Self::transform_down
/// [`transform_down_up`]: Self::transform_down_up
/// [`rewrite`]: Self::rewrite
/// [`apply_children`]: Self::apply_children
/// [`map_children`]: Self::map_children
///
/// # Terminology
/// The following terms are used in this trait
///
/// * `f_down`: Invoked before any children of the current node are visited.
/// * `f_up`: Invoked after all children of the current node are visited.
/// * `f`: closure that is applied to the current node.
/// * `map_*`: applies a transformation to rewrite owned nodes
/// * `apply_*`:  invokes a function on borrowed nodes
/// * `transform_`: applies a transformation to rewrite owned nodes
pub trait TreeNode {
    /// Visit the tree node with a [`TreeNodeVisitor`], performing a
    /// depth-first walk of the node and its children.
    ///
    /// [`TreeNodeVisitor::f_down()`] is called in top-down order (before
    /// children are visited), [`TreeNodeVisitor::f_up()`] is called in
    /// bottom-up order (after children are visited).
    ///
    /// # Return Value
    /// Specifies how the tree walk ended. See [`TreeNodeRecursion`] for details.
    ///
    /// # Example
    /// Consider the following tree structure:
    /// ```text
    /// ParentNode
    ///    left: ChildNode1
    ///    right: ChildNode2
    /// ```
    ///
    /// Here, the nodes would be visited using the following order:
    /// ```text
    /// TreeNodeVisitor::f_down(ParentNode)
    /// TreeNodeVisitor::f_down(ChildNode1)
    /// TreeNodeVisitor::f_up(ChildNode1)
    /// TreeNodeVisitor::f_down(ChildNode2)
    /// TreeNodeVisitor::f_up(ChildNode2)
    /// TreeNodeVisitor::f_up(ParentNode)
    /// ```
    fn visit<V: TreeNodeVisitor<Self>>(
        &self,
        visitor: &mut V,
    ) -> Result<TreeNodeRecursion, V::Error> {
        match visitor.f_down(self)? {
            TreeNodeRecursion::Continue => match self.apply_children(|c| c.visit(visitor))? {
                TreeNodeRecursion::Continue => visitor.f_up(self),
                // If children return `Jump`, it must in the `post_visit` phase, because if
                // `pre_visit` returns `Jump`, we will call the `f_up` function to perform
                // `post_visit` immediately
                tnr => Ok(tnr),
            },
            // Jump in the `pre_visit` phase, execute the `post_visit` phase
            TreeNodeRecursion::Jump => visitor.f_up(self),
            TreeNodeRecursion::Stop => Ok(TreeNodeRecursion::Stop),
        }
    }
    /// Applies `f` to the node then each of its children, recursively (a
    /// top-down, pre-order traversal).
    ///
    /// The return [`TreeNodeRecursion`] controls the recursion and can cause
    /// an early return.
    ///
    /// # See Also
    /// * [`Self::visit`] for both top-down and bottom up traversal.
    fn apply<E, F: FnMut(&Self) -> Result<TreeNodeRecursion, E>>(
        &self,
        mut f: F,
    ) -> Result<TreeNodeRecursion, E> {
        fn apply_impl<N: TreeNode + ?Sized, E, F: FnMut(&N) -> Result<TreeNodeRecursion, E>>(
            node: &N,
            f: &mut F,
        ) -> Result<TreeNodeRecursion, E> {
            match f(node)? {
                TreeNodeRecursion::Continue => node.apply_children(|c| apply_impl(c, f)),
                // In the context of top-down traversal, `Jump` means skip recursing its children.
                // Therefore, we do not call `apply_children` here. Return continue to ensure its
                // siblings are visited.
                TreeNodeRecursion::Jump => Ok(TreeNodeRecursion::Continue),
                TreeNodeRecursion::Stop => Ok(TreeNodeRecursion::Stop),
            }
        }

        apply_impl(self, &mut f)
    }

    /// Returns true if `f` returns true for any node in the tree.
    ///
    /// Stops recursion as soon as a matching node is found
    fn exists<E, F: FnMut(&Self) -> Result<bool, E>>(&self, mut f: F) -> Result<bool, E> {
        let mut found = false;
        self.apply(|n| {
            Ok(if f(n)? {
                found = true;
                TreeNodeRecursion::Stop
            } else {
                TreeNodeRecursion::Continue
            })
        })
        .map(|_| found)
    }

    /// Low-level API used to implement other APIs.
    ///
    /// If you want to implement the [`TreeNode`] trait for your own type, you
    /// should implement this method
    ///
    /// Users should use one of the higher level APIs described on [`Self`].
    ///
    /// Description: Apply `f` to inspect node's children (but not the node
    /// itself).
    fn apply_children<E, F: FnMut(&Self) -> Result<TreeNodeRecursion, E>>(
        &self,
        f: F,
    ) -> Result<TreeNodeRecursion, E>;

    /// Recursively rewrite the node's children and then the node using `f`
    /// (a bottom-up post-order traversal).
    fn transform(&self) {
        todo!()
    }
    /// Recursively rewrite the node using `f` in a bottom-up (post-order)
    /// fashion.
    ///
    /// `f` is applied to the node's  children first, and then to the node itself.
    fn transform_up(&self) {
        todo!()
    }
    /// Recursively rewrite the tree using `f` in a top-down (pre-order)
    /// fashion.
    fn transform_down(&self) {
        todo!()
    }

    /// TODO
    fn transform_down_up(&self) {
        todo!()
    }

    /// TODO
    fn rewrite(&self) {
        todo!()
    }

    /// TODO
    fn map_children(&self) {
        todo!()
    }
}
/// A [Visitor](https://en.wikipedia.org/wiki/Visitor_pattern) for recursively
/// inspecting [`TreeNode`]s via [`TreeNode::visit`].
///
/// See [`TreeNode`] for more details on available APIs
///
/// When passed to [`TreeNode::visit`], [`TreeNodeVisitor::f_down`] and
/// [`TreeNodeVisitor::f_up`] are invoked recursively on the tree.
/// See [`TreeNodeRecursion`] for more details on controlling the traversal.
///
/// # Return Value
/// The returns value of `f_up` and `f_down` specifies how the tree walk should
/// proceed. See [`TreeNodeRecursion`] for details. If an [`Err`] is returned,
/// the recursion stops immediately.
pub trait TreeNodeVisitor<Node: TreeNode + ?Sized>: Sized {
    /// Error that could happen in the visit. If error is returned, stop visiting
    type Error;

    /// Invoked while traversing down the tree, before any children are visited.
    /// Default implementation continues the recursion.
    fn f_down(&mut self, _node: &Node) -> Result<TreeNodeRecursion, Self::Error>;

    /// Invoked while traversing up the tree after children are visited. Default
    /// implementation continues the recursion.
    fn f_up(&mut self, _node: &Node) -> Result<TreeNodeRecursion, Self::Error>;
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
