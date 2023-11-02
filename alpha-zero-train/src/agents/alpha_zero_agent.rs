use game::{Game, Status};
use log::warn;
use mcts::MCTS;
use nn::NN;
use rand::{distributions::WeightedIndex, prelude::Distribution};
use std::sync::atomic::Ordering;
use tch::Device;

use crate::{
    agent::Agent,
    encode::{encode_nn_input, EncodingPerspective},
};

// TODO: split this into a separate crate

#[derive(Debug, Clone, Copy, PartialEq)]
/// The action sampling mode.
pub enum ActionSamplingMode {
    /// Select the action with the highest visit count.
    Best,
    /// Select the action using Boltzmann distribution with the given temperature.
    Boltzmann(f32),
}

/// An agent that uses the AlphaZero algorithm.
pub struct AlphaZeroAgent<G>
where
    G: Game,
{
    mcts: MCTS<G>,
}

impl<G> AlphaZeroAgent<G>
where
    G: Game,
{
    /// Creates a new agent.
    pub fn new() -> Self {
        let game = G::new();

        let mut sum = 0f32;
        let mut p_s = vec![0f32; G::POSSIBLE_ACTION_COUNT];

        for action in 0..G::POSSIBLE_ACTION_COUNT {
            if game.is_action_available(action) {
                sum += 1f32;
                p_s[action] = 1f32;
            }
        }

        if f32::EPSILON < sum {
            let sum_inv = sum.recip();

            for action in 0..G::POSSIBLE_ACTION_COUNT {
                p_s[action] *= sum_inv;
            }
        } else {
            warn!("no action available at the beginning of the game; using uniform distribution");
            let inv_count = (G::POSSIBLE_ACTION_COUNT as f32).recip();
            p_s.fill(inv_count);
        }

        Self {
            // assume that the game is not over at the beginning
            mcts: MCTS::new(p_s, game, None),
        }
    }

    /// Returns the MCTS tree.
    pub fn mcts(&self) -> &MCTS<G> {
        &self.mcts
    }

    /// Computes the MCTS policy.
    /// Returns `None` if:
    /// - The tree is empty.
    /// - All children in the tree have 0 visits (never explored).
    ///
    /// To ensure that the policy is not empty, run MCTS for a few iterations first.
    pub fn compute_mcts_policy(&self) -> Option<Vec<f32>> {
        let root = self.mcts.root();

        let mut sum = 0f32;
        let mut policy = vec![0f32; G::POSSIBLE_ACTION_COUNT];

        {
            let children = root.children.read();

            if children.is_empty() {
                return None;
            }

            for child in children.iter() {
                // it is safe to call `unwrap` here because all children have action
                let action = child.action.unwrap();
                let n_s_a = child.n_s_a.load(Ordering::Relaxed) as f32;
                sum += n_s_a;
                policy[action] = n_s_a;
            }
        }

        if sum <= f32::EPSILON {
            return None;
        }

        let sum_inv = sum.recip();

        for action in 0..G::POSSIBLE_ACTION_COUNT {
            policy[action] *= sum_inv;
        }

        Some(policy)
    }

    /// Samples a single action from the computed policy and returns the action and the policy.
    /// Returns `None` if the policy is empty.
    /// Note that the policy returned by this function is not affected by the temperature;
    /// the temperature is only used to sample the action.
    pub fn sample_action_from_policy(&self, mode: ActionSamplingMode) -> Option<(usize, Vec<f32>)> {
        let policy = self.compute_mcts_policy()?;

        match mode {
            ActionSamplingMode::Best => policy
                .iter()
                .cloned()
                .enumerate()
                .max_by(|&(_, a), &(_, b)| f32::total_cmp(&a, &b))
                .map(|(action, _)| (action, policy)),
            ActionSamplingMode::Boltzmann(temperature) => {
                let temperature_inv = temperature.recip();

                let mut sum = 0f32;
                let mut heated_policy = vec![0f32; G::POSSIBLE_ACTION_COUNT];

                for action in 0..G::POSSIBLE_ACTION_COUNT {
                    let prob = policy[action];
                    let heated = (prob * temperature_inv).exp();
                    sum += heated;
                    heated_policy[action] = heated;
                }

                let sum_inv = sum.recip();

                if sum_inv <= f32::EPSILON {
                    return None;
                }

                for action in 0..G::POSSIBLE_ACTION_COUNT {
                    heated_policy[action] *= sum_inv;
                }

                let dist = WeightedIndex::new(&heated_policy).unwrap();
                let action = dist.sample(&mut rand::thread_rng());

                Some((action, policy))
            }
        }
    }

    /// Ensures that the action exists in the MCTS tree.
    pub fn ensure_action_exists(&mut self, device: Device, nn: &NN<G>, action: usize) {
        let mut game = self.mcts.root().game.clone();
        let turn = game.turn();
        let status = if let Some(status) = game.take_action(action) {
            status
        } else {
            return;
        };

        let input = encode_nn_input(
            device,
            1,
            EncodingPerspective::Player,
            std::iter::once(&game),
        );
        let p_s = nn.forward_pi(&input).to(Device::Cpu);
        let mut p_s = Vec::<f32>::try_from(p_s.view([-1])).unwrap();

        for action in 0..G::POSSIBLE_ACTION_COUNT {
            if !game.is_action_available(action) {
                p_s[action] = 0f32;
            }
        }

        let sum = p_s.iter().sum::<f32>();

        if f32::EPSILON < sum {
            let sum_inv = sum.recip();

            for action in 0..G::POSSIBLE_ACTION_COUNT {
                p_s[action] *= sum_inv;
            }
        }

        let z = match status {
            Status::Ongoing => None,
            status @ (Status::Player1Won | Status::Player2Won) => {
                Some(if status.is_turn(turn) { 1f32 } else { -1f32 })
            }
            Status::Draw => Some(0f32),
        };

        self.mcts.expand(self.mcts.root(), action, p_s, game, z);
    }
}

impl<G> Agent<G> for AlphaZeroAgent<G>
where
    G: Game,
{
    fn select_action(&mut self) -> Option<usize> {
        self.sample_action_from_policy(ActionSamplingMode::Best)
            .map(|(action, _)| action)
    }

    fn take_action(&mut self, action: usize) -> Option<Status> {
        if self.mcts.root().z.is_some() {
            return None;
        }

        let children_index = {
            let children = self.mcts.root().children.read();

            children
                .iter()
                .enumerate()
                .find(|(_, &child)| child.action == Some(action))
                .map(|(index, _)| index)?
        };

        let mut game = self.mcts.root().game.clone();
        let status = game.take_action(action)?;

        self.mcts.transition(children_index);

        Some(status)
    }
}
