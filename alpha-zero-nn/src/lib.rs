use game::Game;
use std::{borrow::Borrow, fmt::Debug, marker::PhantomData, ops::Add};
use tch::{
    nn::{self, ConvConfig, ModuleT, Path},
    Tensor,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NNConfig {
    pub residual_blocks: usize,
    pub residual_block_channels: usize,
    pub fc0_channels: usize,
    pub fc1_channels: usize,
}

pub struct NN<G>
where
    G: Game,
{
    config: NNConfig,
    match_channel_conv: nn::Conv2D,
    residual_blocks: Vec<ResidualBlock>,
    fc0: nn::Linear,
    fc1: nn::Linear,
    value_head: nn::Linear,
    policy_head: nn::Linear,
    _game: PhantomData<G>,
}

impl<G> NN<G>
where
    G: Game,
{
    pub fn new(vs: &Path, config: NNConfig) -> Self {
        let layer_count = G::BOARD_SHAPE.layer_count as i64;
        let height = G::BOARD_SHAPE.height as i64;
        let width = G::BOARD_SHAPE.width as i64;

        let match_channel_conv = nn::conv2d(
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

        let fc0 = nn::linear(
            vs,
            config.residual_block_channels as i64 * height * width,
            config.fc0_channels as i64,
            Default::default(),
        );
        let fc1 = nn::linear(
            vs,
            config.fc0_channels as i64,
            config.fc1_channels as i64,
            Default::default(),
        );

        let value_head = nn::linear(vs, config.fc1_channels as i64, 1, Default::default());
        let policy_head = nn::linear(
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

    pub fn forward(&self, xs: &Tensor, train: bool) -> (Tensor, Tensor) {
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
        let policy = xs
            .apply(&self.policy_head)
            .log_softmax(-1, tch::Kind::Float);

        (value, policy)
    }
}

impl<G> Debug for NN<G>
where
    G: Game,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NN")
            .field("residual_blocks", &self.residual_blocks)
            .field("fc0", &self.fc0)
            .field("fc1", &self.fc1)
            .field("value_head", &self.value_head)
            .field("policy_head", &self.policy_head)
            .field("_game", &self._game)
            .finish()
    }
}

#[derive(Debug)]
struct ResidualBlock {
    pub conv1: nn::Conv2D,
    pub bn1: nn::BatchNorm,
    pub conv2: nn::Conv2D,
    pub bn2: nn::BatchNorm,
}

fn residual_block<'a>(vs: impl Borrow<Path<'a>>, channels: i64) -> ResidualBlock {
    let vs = vs.borrow();
    let conv1 = nn::conv2d(
        vs,
        channels,
        channels,
        3,
        ConvConfig {
            padding: 1,
            ..Default::default()
        },
    );
    let bn1 = nn::batch_norm2d(vs, channels, Default::default());
    let conv2 = nn::conv2d(
        vs,
        channels,
        channels,
        3,
        ConvConfig {
            padding: 1,
            ..Default::default()
        },
    );
    let bn2 = nn::batch_norm2d(vs, channels, Default::default());

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

    #[test]
    fn test_nn_cpu() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let nn = NN::<game::games::TicTacToe>::new(
            &vs.root(),
            NNConfig {
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
        let (value, policy) = nn.forward(&xs, true);

        assert_eq!(value.size(), &[batch, 1]);
        assert_eq!(
            policy.size(),
            &[batch, game::games::TicTacToe::POSSIBLE_ACTION_COUNT as i64]
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_nn_mps() {
        let vs = nn::VarStore::new(tch::Device::Mps);
        let nn = NN::<game::games::TicTacToe>::new(
            &vs.root(),
            NNConfig {
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

        let xs = Tensor::randn(
            &[batch, layer_count * height * width],
            (tch::Kind::Float, tch::Device::Mps),
        );
        let (value, policy) = nn.forward(&xs, true);

        assert_eq!(value.size(), &[batch, 1]);
        assert_eq!(
            policy.size(),
            &[batch, game::games::TicTacToe::POSSIBLE_ACTION_COUNT as i64]
        );
    }
}
