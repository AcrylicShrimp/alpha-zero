use num_traits::Float;

fn main() {
    println!("Hello, world!");
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Turn {
    Player1,
    Player2,
}

/// Represents a game, especially a complete information, turn-based two-player zero-sum game such as chess or go.
/// The board must be square and the size of the board must be known at compile time.
pub trait Game<F: Float> {
    /// Size of the board. Since the board must be square, total number of fields is `BOARD_SIZE * BOARD_SIZE`.
    const BOARD_SIZE: usize;

    /// Total number of fields on the board.
    const CELL_COUNT: usize = Self::BOARD_SIZE * Self::BOARD_SIZE;

    /// Number of layers in the board representation. For example, in chess, there are 6 types of pieces and 2 players, so there are 12 layers.
    const BOARD_LAYER_COUNT: usize;

    /// Maximum number of possible actions. All possible actions should be representable in form of an one-hot vector (zeroes and one).
    const POSSIBLE_ACTION_COUNT: usize;

    /// Create a new game.
    fn new() -> Self;

    /// Return the current turn.
    fn turn(&self) -> Turn;

    /// Return all possible actions in form of a one-hot vector. Returned vector should have length of `POSSIBLE_ACTION_COUNT`.
    fn possible_actions(&self) -> Vec<F>;

    /// Encode the current board state in form of a one-hot vector, in perspective of the current player (i.e. the player who is about to make a move).
    /// Returned vector should have shape of `(BOARD_LAYER_COUNT, BOARD_SIZE, BOARD_SIZE)`.
    fn board_state(&self) -> Vec<F>;
}
