pub mod node;
pub mod node_allocator;

use crate::{
    node::{Node, NodePtr},
    node_allocator::NodeAllocator,
};
use game::Game;
use node::NodeOrAction;
use parking_lot::Mutex;
use std::ptr::NonNull;

/// Monte Carlo Tree Search. This is a generic implementation of the MCTS algorithm.
/// It is thread-safe and can be used in a multi-threaded environment.
pub struct MCTS<G>
where
    G: Game,
{
    root: NodePtr<G>,
    allocator: Mutex<NodeAllocator<Node<G>>>,
}

impl<G> MCTS<G>
where
    G: Game,
{
    /// Create a new MCTS instance.
    pub fn new(p_s: Vec<f32>, game: G, z: Option<f32>) -> Self {
        let mut allocator = NodeAllocator::new();
        let root = allocator.allocate(Node::new(None, None, 1f32, p_s, game, z));

        Self {
            root: NodePtr::new(root),
            allocator: Mutex::new(allocator),
        }
    }

    /// Return the root node.
    pub fn root(&self) -> &Node<G> {
        &*self.root
    }

    /// Select a leaf node and highest scored action using the given selector function.
    pub fn select_leaf(
        &self,
        score_fn: impl Fn(&Node<G>, NodeOrAction<G>) -> f32,
    ) -> (&Node<G>, usize) {
        self.root().select_leaf(score_fn)
    }

    /// Expand the given action at the given node.
    pub fn expand(
        &self,
        node: &Node<G>,
        action: usize,
        p_s: Vec<f32>,
        game: G,
        z: Option<f32>,
    ) -> Option<NodePtr<G>> {
        node.expand(action, p_s, game, z, &mut self.allocator.lock())
    }

    /// Transition to the given child node. In other words, make the given child node the new root and drop all other branches.
    /// This is useful if you want to reuse the tree for the next move.
    pub fn transition(&mut self, children_index: usize) {
        let allocator = &mut self.allocator.lock();

        let mut new_root = {
            let root = self.root();
            let root_children = root.children.read();

            for index in 0..root_children.len() {
                if index == children_index {
                    continue;
                }

                dealloc_node(root_children[index].ptr, allocator);
            }

            root_children[children_index]
        };

        {
            let new_root = unsafe { new_root.ptr.as_mut() };
            new_root.parent = None;
        }

        allocator.deallocate(self.root.ptr);
        self.root = new_root;
    }
}

fn dealloc_node<G>(mut ptr: NonNull<Node<G>>, allocator: &mut NodeAllocator<Node<G>>)
where
    G: Game,
{
    {
        let node = unsafe { ptr.as_mut() };
        let children = node.children.read();
        for child in children.iter() {
            dealloc_node(child.ptr, allocator);
        }
    }
    allocator.deallocate(ptr);
}

unsafe impl<G> Send for MCTS<G> where G: Game {}
unsafe impl<G> Sync for MCTS<G> where G: Game {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition() {
        let game = game::games::TicTacToe::new();
        let p_s = vec![
            1f32 / game.possible_action_count() as f32;
            game::games::TicTacToe::POSSIBLE_ACTION_COUNT
        ];
        let mut mcts = MCTS::new(p_s, game, None);

        let child = {
            let (node, action) = mcts.select_leaf(|_, _| 0f32);
            let game = node.game.clone();
            let p_s = vec![
                1f32 / game.possible_action_count() as f32;
                game::games::TicTacToe::POSSIBLE_ACTION_COUNT
            ];
            let child = mcts
                .expand(node, action, p_s, mcts.root().game.clone(), None)
                .unwrap();

            child
        };

        mcts.transition(0);

        assert_eq!(
            mcts.root() as *const Node<game::games::TicTacToe>,
            child.ptr.as_ptr()
        );
        assert_eq!(mcts.root().children.read().len(), 0);
    }
}
