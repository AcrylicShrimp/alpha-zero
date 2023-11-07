use crate::node_allocator::NodeAllocator;
use atomic_float::AtomicF32;
use game::Game;
use parking_lot::RwLock;
use std::{
    ops::Deref,
    ptr::NonNull,
    sync::atomic::{AtomicU32, Ordering},
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NodeOrAction<G>
where
    G: Game,
{
    Node(NodePtr<G>),
    Action(usize),
}

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

    /// Select a leaf node and highest scored action using the given selector function.
    pub fn select_leaf(&self, score_fn: impl Fn(&Self, NodeOrAction<G>) -> f32) -> (&Self, usize) {
        let mut node = self;
        let mut children = self.children.read();

        loop {
            let action = (0..G::POSSIBLE_ACTION_COUNT)
                .into_iter()
                .filter_map(|action| {
                    if !node.game.is_action_available(action) {
                        return None;
                    }

                    let child = children.iter().find(|child| child.action == Some(action));
                    let score = match child {
                        Some(child) => score_fn(node, NodeOrAction::Node(child.clone())),
                        None => score_fn(node, NodeOrAction::Action(action)),
                    };

                    Some((action, score))
                })
                .max_by(|(_, score1), (_, score2)| f32::total_cmp(score1, score2))
                .unwrap()
                .0;

            let child_index = children
                .iter()
                .position(|child| child.action == Some(action));

            match child_index {
                Some(child_index) => {
                    node = unsafe { children[child_index].ptr.as_ref() };
                    let child_children = node.children.read();
                    children = child_children;
                }
                None => {
                    return (node, action);
                }
            }
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
