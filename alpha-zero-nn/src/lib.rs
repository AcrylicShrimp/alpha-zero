pub mod model;
pub mod nn_optimizer;

use game::Game;
use model::{Model, ModelConfig};
use std::{
    fmt::Debug,
    io::{Read, Seek},
};
use tch::{nn::VarStore, Device, Kind, Reduction, TchError, Tensor};

/// Configuration for the neural network.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NNConfig {
    /// The device to use.
    pub device: Device,
    /// The number of residual blocks.
    pub residual_blocks: usize,
    /// The number of channels in the residual blocks.
    pub residual_block_channels: usize,
    /// The number of channels in the first fully connected layer.
    pub fc0_channels: usize,
    /// The number of channels in the second fully connected layer.
    pub fc1_channels: usize,
}

/// A neural network.
pub struct NN<G>
where
    G: Game,
{
    config: NNConfig,
    master: Model<G>,
    vs_master: VarStore,
}

impl<G> NN<G>
where
    G: Game,
{
    pub fn new(config: NNConfig) -> Self {
        let vs_master = VarStore::new(config.device);

        let master = Model::<G>::new(
            &vs_master.root(),
            ModelConfig {
                kind: Kind::Float,
                residual_blocks: config.residual_blocks,
                residual_block_channels: config.residual_block_channels,
                fc0_channels: config.fc0_channels,
                fc1_channels: config.fc1_channels,
            },
        );

        Self {
            config,
            master,
            vs_master,
        }
    }

    /// Returns the configuration.
    pub fn config(&self) -> &NNConfig {
        &self.config
    }

    /// Returns the master model.
    pub fn master(&self) -> &Model<G> {
        &self.master
    }

    /// Returns the master variable store.
    pub fn vs_master(&self) -> &VarStore {
        &self.vs_master
    }

    /// Loads the weights from the given stream.
    /// Note that the weights are loaded into the master model and then copied to the cloned model.
    pub fn load_weights(&mut self, weights: impl Read + Seek) -> Result<(), TchError> {
        self.vs_master.load_from_stream(weights)?;

        Ok(())
    }

    /// Computes the value and policy for the given batch.
    pub fn forward<'g>(
        &self,
        train: bool,
        batch_size: usize,
        game_iter: impl Iterator<Item = &'g G>,
    ) -> (Tensor, Tensor)
    where
        G: 'g,
    {
        let input = self.encode_input(batch_size, game_iter);

        self.master.forward(&input, train)
    }

    /// Computes the policy for the given batch.
    pub fn forward_pi<'g>(
        &self,
        train: bool,
        batch_size: usize,
        game_iter: impl Iterator<Item = &'g G>,
    ) -> Tensor
    where
        G: 'g,
    {
        let input = self.encode_input(batch_size, game_iter);

        self.master.forward_pi(&input, train)
    }

    /// Computes the value and policy loss for the given batch.
    pub fn loss<'g>(
        &self,
        train: bool,
        batch_size: usize,
        game_iter: impl Iterator<Item = &'g G>,
        z_iter: impl Iterator<Item = f32>,
        policy_iter: impl Iterator<Item = &'g [f32]>,
    ) -> (Tensor, Tensor)
    where
        G: 'g,
    {
        let (v, pi) = self.forward(train, batch_size, game_iter);
        let (z_target, policy_target) = self.encode_targets(batch_size, z_iter, policy_iter);

        let v = v.to_kind(Kind::Float);
        let pi = pi.to_kind(Kind::Float);

        let z_target = z_target.view([-1, 1]).to_kind(Kind::Float);
        let policy_target = policy_target
            .view([-1, G::POSSIBLE_ACTION_COUNT as i64])
            .to_kind(Kind::Float);

        // let v_loss = v.mse_loss(&z_target, Reduction::Mean);
        let v_loss = (v - z_target)
            .pow(&Tensor::from_slice(&[2f32]).to(self.config.device))
            .mean(None);
        let pi_loss =
            pi.cross_entropy_loss::<&Tensor>(&policy_target, None, Reduction::Mean, -100, 0.0);

        (v_loss, pi_loss)
    }

    /// Encodes the input for the given batch.
    fn encode_input<'g>(&self, batch_size: usize, game_iter: impl Iterator<Item = &'g G>) -> Tensor
    where
        G: 'g,
    {
        let mut input = vec![0f32; batch_size * G::BOARD_SHAPE.size()];

        for (index, game) in game_iter.enumerate() {
            let board = game.board_state();
            let dst =
                &mut input[index * G::BOARD_SHAPE.size()..(index + 1) * G::BOARD_SHAPE.size()];

            for (index, value) in board.into_iter().enumerate() {
                dst[index] = if value { 1.0 } else { 0.0 };
            }
        }

        Tensor::from_slice(&input).to(self.config.device)
    }

    /// Encodes the targets for the given batch.
    fn encode_targets<'g>(
        &self,
        batch_size: usize,
        z_iter: impl Iterator<Item = f32>,
        policy_iter: impl Iterator<Item = &'g [f32]>,
    ) -> (Tensor, Tensor)
    where
        G: 'g,
    {
        let mut z_target = vec![0f32; batch_size];
        let mut policy_target = vec![0f32; batch_size * G::POSSIBLE_ACTION_COUNT];

        for (index, (policy, z)) in (policy_iter.zip(z_iter)).enumerate() {
            z_target[index] = z;
            policy_target[index * G::POSSIBLE_ACTION_COUNT..(index + 1) * G::POSSIBLE_ACTION_COUNT]
                .copy_from_slice(policy);
        }

        (
            Tensor::from_slice(&z_target).to(self.config.device),
            Tensor::from_slice(&policy_target).to(self.config.device),
        )
    }
}

unsafe impl<G> Send for NN<G> where G: Game {}
unsafe impl<G> Sync for NN<G> where G: Game {}
