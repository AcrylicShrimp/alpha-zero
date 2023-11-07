use crate::agents::AlphaZeroAgent;
use game::{Game, Status, Turn};
use log::trace;
use mcts::node::{Node, NodeOrAction, NodePtr};
use nn::NN;
use rand::thread_rng;
use rand_distr::{Dirichlet, Distribution};
use rayon::{
    prelude::{IntoParallelRefIterator, ParallelIterator},
    ThreadPool, ThreadPoolBuildError, ThreadPoolBuilder,
};
use std::{num::NonZeroUsize, sync::atomic::Ordering};
use tch::Device;

#[derive(Debug, Clone, PartialEq)]
/// The configuration for the MCTS executor.
pub struct MCTSExecutorConfig {
    /// The number of threads to use.
    pub num_threads: Option<NonZeroUsize>,
}

impl Default for MCTSExecutorConfig {
    fn default() -> Self {
        Self { num_threads: None }
    }
}

/// A parallel MCTS executor. It will use UCB1 to select the best child node.
pub struct ParallelMCTSExecutor {
    thread_pool: ThreadPool,
}

impl ParallelMCTSExecutor {
    /// Creates a new parallel MCTS executor.
    pub fn new(config: MCTSExecutorConfig) -> Result<Self, ThreadPoolBuildError> {
        let mut thread_pool_builder = ThreadPoolBuilder::new();

        if let Some(num_threads) = config.num_threads {
            thread_pool_builder = thread_pool_builder.num_threads(num_threads.get());
        }

        let thread_pool = thread_pool_builder.build()?;

        Ok(Self { thread_pool })
    }

    pub fn execute<G>(
        &self,
        count: usize,
        batch_size: usize,
        c_puct: f32,
        alpha: f32,
        epsilon: f32,
        nn: &NN<G>,
        agents: &[AlphaZeroAgent<G>],
    ) where
        G: Game,
    {
        self.thread_pool.install(|| {
            let mut processed = 0;

            loop {
                if count <= processed {
                    break;
                }

                let requests: Vec<_> = agents
                    .par_iter()
                    .flat_map(|agent| {
                        generate_requests(agent, batch_size, c_puct, processed == 0, alpha, epsilon)
                    })
                    .collect();

                processed += batch_size;

                if requests.is_empty() {
                    continue;
                }

                trace!("requests.len()={}", requests.len());

                // we're inferencing to expand the tree, so it's important to pass `false` as the `train` argument
                // this will ensure that the batch norm layers don't update their running statistics
                let (v, pi) = nn.forward(
                    false,
                    requests.len(),
                    requests.iter().map(|request| &request.node.game),
                );

                let v = Vec::<f32>::try_from(v.to(Device::Cpu).view([-1])).unwrap();
                let mut pi = Vec::<f32>::try_from(pi.to(Device::Cpu).view([-1])).unwrap();

                for (batch_index, request) in requests.iter().enumerate() {
                    let mut w_s_a = v[batch_index];

                    if request.parent_turn != request.node.game.turn() {
                        // negate the value if the parent turn is different from the node's turn
                        w_s_a = -w_s_a;
                    }

                    let node = &*request.node;
                    let p_s = &mut pi[batch_index * G::POSSIBLE_ACTION_COUNT
                        ..(batch_index + 1) * G::POSSIBLE_ACTION_COUNT];

                    // filter out illegal actions and normalize the pi
                    for action in 0..G::POSSIBLE_ACTION_COUNT {
                        if !node.game.is_action_available(action) {
                            p_s[action] = 0f32;
                        }
                    }

                    let sum = p_s.iter().sum::<f32>();

                    if f32::EPSILON <= sum {
                        let sum_inv = sum.recip();

                        for p_s_a in p_s.iter_mut() {
                            *p_s_a *= sum_inv;
                        }
                    }

                    // update the pre-expanded child node's prior probability
                    node.p_s.write().copy_from_slice(p_s);

                    // update the post-expanded child node's prior probability
                    // this is necessary because the child node's prior probability is holding dummy values
                    for child in node.children.read().iter() {
                        let child = &*child;
                        let action = child.action.unwrap();
                        let p_s_a = p_s[action];
                        child.p_s_a.store(p_s_a, Ordering::Relaxed);
                    }

                    // perform backup from the expanded child node
                    node.propagate(w_s_a);
                }
            }
        });
    }
}

/// A request for the neural network evaluation.
struct NNEvalRequest<G>
where
    G: Game,
{
    pub parent_turn: Turn,
    pub node: NodePtr<G>,
}

/// Generate the requests for the neural network evaluation.
fn generate_requests<G>(
    agent: &AlphaZeroAgent<G>,
    batch_size: usize,
    c_puct: f32,
    apply_diritchlet_noise: bool,
    alpha: f32,
    epsilon: f32,
) -> Vec<NNEvalRequest<G>>
where
    G: Game,
{
    let mut rng = thread_rng();

    if apply_diritchlet_noise {
        let noise_dist = Dirichlet::new(&vec![alpha; G::POSSIBLE_ACTION_COUNT]).unwrap();
        let noise = noise_dist.sample(&mut rng);

        let mut p_s = agent.mcts().root().p_s.write();

        for (action, p_s_a) in p_s.iter_mut().enumerate() {
            *p_s_a = (1.0 - epsilon) * *p_s_a + epsilon * noise[action];
        }

        let sum = p_s.iter().sum::<f32>();
        let sum_inv = sum.recip();

        for p_s_a in p_s.iter_mut() {
            *p_s_a *= sum_inv;
        }

        // update the post-expanded child node's prior probability
        for child in agent.mcts().root().children.read().iter() {
            let action = child.action.unwrap();
            let p_s_a = p_s[action];
            child.p_s_a.store(p_s_a, Ordering::Relaxed);
        }
    }

    let mut requests = Vec::with_capacity(batch_size);

    for _ in 0..batch_size {
        let (node, action) = agent.mcts().select_leaf(|parent, child| match child {
            NodeOrAction::Node(child) => {
                let parent_n_s_a = parent.n_s_a.load(Ordering::Relaxed);
                compute_node_puct(parent_n_s_a, &child, c_puct)
            }
            NodeOrAction::Action(action) => {
                let parent_n_s_a = parent.n_s_a.load(Ordering::Relaxed);
                compute_action_puct(parent, parent_n_s_a, action, c_puct)
            }
        });

        if let Some(z) = node.z {
            trace!("the game is over; propagating z={}", z);
            node.propagate(z);
            continue;
        }

        let mut game = node.game.clone();
        let turn = game.turn();
        let status = game.take_action(action).unwrap();
        let z = match status {
            Status::Ongoing => None,
            status @ (Status::Player1Won | Status::Player2Won) => {
                Some(if status.is_turn(turn) { 1f32 } else { -1f32 })
            }
            Status::Draw => Some(0f32),
        };

        // pre-compute the prior probability
        // note that this is not the final prior probability; it's dummy values
        let mut p_s = vec![0f32; G::POSSIBLE_ACTION_COUNT];

        for action in 0..G::POSSIBLE_ACTION_COUNT {
            if game.is_action_available(action) {
                p_s[action] = 1f32;
            }
        }

        let sum = p_s.iter().sum::<f32>();

        if f32::EPSILON < sum {
            let sum_inv = sum.recip();

            for p_s_a in p_s.iter_mut() {
                *p_s_a *= sum_inv;
            }
        }

        let child = match agent.mcts().expand(node, action, p_s, game, z) {
            Some(child) => child,
            None => continue,
        };

        match z {
            Some(z) => {
                child.propagate(z);
            }
            None => {
                requests.push(NNEvalRequest {
                    parent_turn: node.game.turn(),
                    node: child,
                });
            }
        }
    }

    requests
}

/// Compute the PUCT value for the given node.
fn compute_node_puct<G>(parent_n_s_a: u32, node: &Node<G>, c_puct: f32) -> f32
where
    G: Game,
{
    let n_s_a = node.n_s_a.load(Ordering::Relaxed) as f32;
    let w_s_a = node.w_s_a.load(Ordering::Relaxed);
    let q_s_a = w_s_a / (n_s_a + 0.0001f32);
    let p_s_a = node.p_s_a.load(Ordering::Relaxed);
    let bias = c_puct * p_s_a * (parent_n_s_a as f32).sqrt() / (1.0 + n_s_a);
    q_s_a + bias
}

/// Compute the PUCT value for the given action.
fn compute_action_puct<G>(parent: &Node<G>, parent_n_s_a: u32, action: usize, c_puct: f32) -> f32
where
    G: Game,
{
    let p_s_a = parent.p_s.read()[action];
    c_puct * p_s_a * (parent_n_s_a as f32).sqrt()
}
