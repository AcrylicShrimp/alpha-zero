use game::Game;
use std::{borrow::Borrow, marker::PhantomData, ops::Add};
use tch::{
    nn::{batch_norm2d, conv2d, linear, BatchNorm, Conv2D, ConvConfig, Linear, ModuleT, Path},
    Kind, Tensor,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelConfig {
    pub kind: Kind,
    pub residual_blocks: usize,
    pub residual_block_channels: usize,
    pub fc0_channels: usize,
    pub fc1_channels: usize,
}

pub struct Model<G>
where
    G: Game,
{
    config: ModelConfig,
    match_channel_conv: Conv2D,
    residual_blocks: Vec<ResidualBlock>,
    fc0: Linear,
    fc1: Linear,
    value_head: Linear,
    policy_head: Linear,
    _game: PhantomData<G>,
}

impl<G> Model<G>
where
    G: Game,
{
    pub fn new(vs: &Path, config: ModelConfig) -> Self {
        let layer_count = G::BOARD_SHAPE.layer_count as i64;
        let height = G::BOARD_SHAPE.height as i64;
        let width = G::BOARD_SHAPE.width as i64;

        let match_channel_conv = conv2d(
            vs,
            layer_count,
            config.residual_block_channels as i64,
            1,
            Default::default(),
        );
        let mut residual_blocks = Vec::with_capacity(config.residual_blocks);

        for _ in 0..config.residual_blocks {
            residual_blocks.push(residual_block(vs, config.residual_block_channels as i64));
        }

        let fc0 = linear(
            vs,
            config.residual_block_channels as i64 * height * width,
            config.fc0_channels as i64,
            Default::default(),
        );
        let fc1 = linear(
            vs,
            config.fc0_channels as i64,
            config.fc1_channels as i64,
            Default::default(),
        );

        let value_head = linear(vs, config.fc1_channels as i64, 1, Default::default());
        let policy_head = linear(
            vs,
            config.fc1_channels as i64,
            G::POSSIBLE_ACTION_COUNT as i64,
            Default::default(),
        );

        Self {
            config,
            match_channel_conv,
            residual_blocks,
            fc0,
            fc1,
            value_head,
            policy_head,
            _game: PhantomData,
        }
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> (Tensor, Tensor) {
        let xs = xs.to_kind(self.config.kind);

        let layer_count = G::BOARD_SHAPE.layer_count as i64;
        let height = G::BOARD_SHAPE.height as i64;
        let width = G::BOARD_SHAPE.width as i64;

        let mut xs = xs
            .view([-1, layer_count, height, width])
            .apply(&self.match_channel_conv)
            .relu();

        for residual in &self.residual_blocks {
            xs = xs.apply_t(residual, train);
        }

        let flatten_size = self.config.residual_block_channels as i64 * height * width;

        xs = xs
            .view([-1, flatten_size])
            .apply(&self.fc0)
            .relu()
            .apply(&self.fc1)
            .relu();

        let value = xs.apply(&self.value_head).tanh();
        let mut policy = xs.apply(&self.policy_head);

        if !train {
            policy = policy.softmax(-1, Kind::Float).to_kind(self.config.kind);
        }

        (value, policy)
    }

    pub fn forward_pi(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = xs.to_kind(self.config.kind);

        let layer_count = G::BOARD_SHAPE.layer_count as i64;
        let height = G::BOARD_SHAPE.height as i64;
        let width = G::BOARD_SHAPE.width as i64;

        let mut xs = xs
            .view([-1, layer_count, height, width])
            .apply(&self.match_channel_conv)
            .relu();

        for residual in &self.residual_blocks {
            xs = xs.apply_t(residual, train);
        }

        let flatten_size = self.config.residual_block_channels as i64 * height * width;

        xs = xs
            .view([-1, flatten_size])
            .apply(&self.fc0)
            .relu()
            .apply(&self.fc1)
            .relu();

        xs.apply(&self.policy_head)
            .softmax(-1, Kind::Float)
            .to_kind(self.config.kind)
    }
}

unsafe impl<G> Send for Model<G> where G: Game {}
unsafe impl<G> Sync for Model<G> where G: Game {}

#[derive(Debug)]
struct ResidualBlock {
    pub conv1: Conv2D,
    pub bn1: BatchNorm,
    pub conv2: Conv2D,
    pub bn2: BatchNorm,
}

fn residual_block<'a>(vs: impl Borrow<Path<'a>>, channels: i64) -> ResidualBlock {
    let vs = vs.borrow();
    let conv1 = conv2d(
        vs,
        channels,
        channels,
        3,
        ConvConfig {
            padding: 1,
            ..Default::default()
        },
    );
    let bn1 = batch_norm2d(vs, channels, Default::default());
    let conv2 = conv2d(
        vs,
        channels,
        channels,
        3,
        ConvConfig {
            padding: 1,
            ..Default::default()
        },
    );
    let bn2 = batch_norm2d(vs, channels, Default::default());

    ResidualBlock {
        conv1,
        bn1,
        conv2,
        bn2,
    }
}

impl ModuleT for ResidualBlock {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.apply(&self.conv1)
            .apply_t(&self.bn1, train)
            .relu()
            .apply(&self.conv2)
            .apply_t(&self.bn2, train)
            .add(xs)
            .relu()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::nn::VarStore;

    #[test]
    fn test_model_cpu() {
        let vs = VarStore::new(tch::Device::Cpu);
        let model = Model::<game::games::TicTacToe>::new(
            &vs.root(),
            ModelConfig {
                kind: Kind::Float,
                residual_blocks: 2,
                residual_block_channels: 32,
                fc0_channels: 32,
                fc1_channels: 32,
            },
        );

        let board_shape = game::games::TicTacToe::BOARD_SHAPE;
        let layer_count = board_shape.layer_count as i64;
        let height = board_shape.height as i64;
        let width = board_shape.width as i64;

        let batch = 16;

        let xs = Tensor::randn(&[batch, layer_count * height * width], tch::kind::FLOAT_CPU);
        let (value, policy) = model.forward(&xs, true);

        assert_eq!(value.size(), &[batch, 1]);
        assert_eq!(
            policy.size(),
            &[batch, game::games::TicTacToe::POSSIBLE_ACTION_COUNT as i64]
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_nn_mps() {
        let vs = VarStore::new(tch::Device::Mps);
        let model = Model::<game::games::TicTacToe>::new(
            &vs.root(),
            ModelConfig {
                kind: Kind::Float,
                residual_blocks: 2,
                residual_block_channels: 32,
                fc0_channels: 32,
                fc1_channels: 32,
            },
        );

        let board_shape = game::games::TicTacToe::BOARD_SHAPE;
        let layer_count = board_shape.layer_count as i64;
        let height = board_shape.height as i64;
        let width = board_shape.width as i64;

        let batch = 16;

        let xs = Tensor::randn(&[batch, layer_count * height * width], tch::kind::FLOAT_CPU);
        let (value, policy) = model.forward(&xs, true);

        assert_eq!(value.size(), &[batch, 1]);
        assert_eq!(
            policy.size(),
            &[batch, game::games::TicTacToe::POSSIBLE_ACTION_COUNT as i64]
        );
    }
}
