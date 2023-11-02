use crate::node_allocator::NodeAllocator;
use atomic_float::AtomicF32;
use game::Game;
use parking_lot::RwLock;
use std::{
    ops::Deref,
    ptr::NonNull,
    sync::atomic::{AtomicU32, Ordering},
};

/// Node in the MCTS tree.
pub struct Node<G>
where
    G: Game,
{
    /// Parent of this node.
    pub parent: Option<NodePtr<G>>,
    /// Action that led to this node.
    pub action: Option<usize>,
    /// Children of this node.
    pub children: RwLock<Vec<NodePtr<G>>>,
    /// Prior probability of selecting this node.
    pub p_s_a: AtomicF32,
    /// Number of times edge s->a was visited.
    pub n_s_a: AtomicU32,
    /// Total action value of edge s->a.
    pub w_s_a: AtomicF32,
    /// Maximum number of children.
    pub maximum_children: usize,
    /// Prior probabilities of all actions.
    pub p_s: RwLock<Vec<f32>>,
    /// Game state.
    pub game: G,
    /// Reward of this node if it is a terminal node. `None` if it is not a terminal node.
    pub z: Option<f32>,
}

impl<G> Node<G>
where
    G: Game,
{
    /// Create a new node.
    pub fn new(
        parent: Option<NodePtr<G>>,
        action: Option<usize>,
        p_s_a: f32,
        p_s: Vec<f32>,
        game: G,
        z: Option<f32>,
    ) -> Self {
        Self {
            parent,
            action,
            children: RwLock::new(Vec::new()),
            p_s_a: AtomicF32::new(p_s_a),
            n_s_a: AtomicU32::new(0),
            w_s_a: AtomicF32::new(0.0),
            maximum_children: game.possible_action_count(),
            p_s: RwLock::new(p_s),
            game,
            z,
        }
    }

    /// Select a leaf node by using the given selector function.
    pub fn select_leaf(&self, selector: impl Fn(&Self, &[NodePtr<G>]) -> usize) -> &Self {
        let mut node = self;
        let mut children = self.children.read();

        loop {
            if children.len() == 0 {
                return node;
            }

            // if we have not reached the max number of children, return this node, since it is a leaf
            if children.len() != node.maximum_children {
                return node;
            }

            let index = selector(node, &children);

            node = unsafe { children[index].ptr.as_ref() };
            let child_children = node.children.read();
            children = child_children;
        }
    }

    /// Expand the given action.
    pub fn expand(
        &self,
        action: usize,
        p_s: Vec<f32>,
        game: G,
        z: Option<f32>,
        allocator: &mut NodeAllocator<Self>,
    ) -> Option<NodePtr<G>> {
        let mut children = self.children.write();

        if children.iter().any(|child| child.action == Some(action)) {
            return None;
        }

        let node = Self::new(
            Some(NodePtr::new(self.into())),
            Some(action),
            self.p_s.read()[action],
            p_s,
            game,
            z,
        );
        let node_ptr = allocator.allocate(node);
        let child = NodePtr::new(node_ptr);
        children.push(child.clone());

        Some(child)
    }

    /// Backpropagate the given action value.
    pub fn propagate(&self, mut w_s_a: f32) {
        let mut node = self;

        loop {
            node.n_s_a.fetch_add(1, Ordering::Relaxed);
            node.w_s_a.fetch_add(w_s_a, Ordering::Relaxed);

            w_s_a = -w_s_a;

            if let Some(parent) = &node.parent {
                node = &*parent;
            } else {
                break;
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// Pointer to a node in the MCTS tree.
pub struct NodePtr<G>
where
    G: Game,
{
    pub ptr: NonNull<Node<G>>,
}

impl<G> NodePtr<G>
where
    G: Game,
{
    /// Create a new node pointer.
    pub fn new(ptr: NonNull<Node<G>>) -> Self {
        Self { ptr }
    }
}

impl<G> Clone for NodePtr<G>
where
    G: Game,
{
    fn clone(&self) -> Self {
        Self { ptr: self.ptr }
    }
}

impl<G> Copy for NodePtr<G> where G: Game {}

impl<G> Deref for NodePtr<G>
where
    G: Game,
{
    type Target = Node<G>;

    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref() }
    }
}

unsafe impl<G> Send for Node<G> where G: Game {}
unsafe impl<G> Send for NodePtr<G> where G: Game {}
