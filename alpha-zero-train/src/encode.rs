use game::Game;
use log::trace;
use tch::{Device, Tensor};

/// The perspective from which the game is encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EncodingPerspective {
    /// The player's perspective whos turn it is.
    Player,
    /// The opponent's perspective.
    Opponent,
}

pub fn encode_nn_input<'g, G>(
    device: Device,
    batch_size: usize,
    perspective: EncodingPerspective,
    game_iter: impl Iterator<Item = &'g G>,
) -> Tensor
where
    G: 'g + Game,
{
    let mut input = vec![0f32; batch_size * G::BOARD_SHAPE.size()];

    trace!("batch_size={}", batch_size);

    for (index, game) in game_iter.enumerate() {
        let board = game.board_state(perspective == EncodingPerspective::Opponent);
        let dst = &mut input[index * G::BOARD_SHAPE.size()..(index + 1) * G::BOARD_SHAPE.size()];

        for (index, value) in board.into_iter().enumerate() {
            dst[index] = if value { 1.0 } else { 0.0 };
        }
    }

    Tensor::from_slice(&input).to(device)
}

pub fn encode_nn_targets<'g, G>(
    device: Device,
    batch_size: usize,
    z_iter: impl Iterator<Item = f32>,
    policy_iter: impl Iterator<Item = &'g [f32]>,
) -> (Tensor, Tensor)
where
    G: Game,
{
    let mut z_target = vec![0f32; batch_size];
    let mut policy_target = vec![0f32; batch_size * G::POSSIBLE_ACTION_COUNT];

    for (index, (policy, z)) in (policy_iter.zip(z_iter)).enumerate() {
        z_target[index] = z;
        policy_target[index * G::POSSIBLE_ACTION_COUNT..(index + 1) * G::POSSIBLE_ACTION_COUNT]
            .copy_from_slice(policy);
    }

    (
        Tensor::from_slice(&z_target).to(device),
        Tensor::from_slice(&policy_target).to(device),
    )
}
