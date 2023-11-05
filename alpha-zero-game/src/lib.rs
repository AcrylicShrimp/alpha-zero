mod augmentation_utils;
pub mod games;

use bitvec::vec::BitVec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Represents the shape of a board.
pub struct BoardShape {
    /// Number of layers in the board representation. For example, in chess, there are 6 types of pieces and 2 players, so there are 12 layers.
    pub layer_count: u32,
    /// Height of the board.
    pub height: u32,
    /// Width of the board.
    pub width: u32,
}

impl BoardShape {
    /// Create a new board shape.
    pub const fn new(layer_count: u32, height: u32, width: u32) -> Self {
        Self {
            layer_count,
            height,
            width,
        }
    }

    /// Return the total number of fields on a single layer.
    pub const fn single_layer_size(self) -> usize {
        (self.height * self.width) as usize
    }

    /// Return the total number of fields on the board.
    pub const fn size(self) -> usize {
        (self.layer_count * self.height * self.width) as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Represents a turn; the player who is about to make a move.
pub enum Turn {
    Player1,
    Player2,
}

impl Turn {
    pub fn opposite(&self) -> Turn {
        match self {
            Turn::Player1 => Turn::Player2,
            Turn::Player2 => Turn::Player1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Represents a game status.
pub enum Status {
    Ongoing,
    Player1Won,
    Player2Won,
    Draw,
}

impl Status {
    pub fn is_turn(self, turn: Turn) -> bool {
        match self {
            Status::Ongoing => false,
            Status::Player1Won => turn == Turn::Player1,
            Status::Player2Won => turn == Turn::Player2,
            Status::Draw => false,
        }
    }
}

/// Represents a game, especially a complete information, turn-based two-player zero-sum game such as chess or go.
/// The board must be square and the size of the board must be known at compile time.
pub trait Game
where
    Self: Clone + Send + SerdeGame,
{
    /// Shape of the board, in form of `(BOARD_LAYER_COUNT, BOARD_SIZE_HEIGHT, BOARD_SIZE_WIDTH)`.
    /// Normally boards are square.
    const BOARD_SHAPE: BoardShape;

    /// Maximum number of possible actions. All possible actions should be representable in form of an one-hot vector (zeroes and one).
    const POSSIBLE_ACTION_COUNT: usize;

    /// Create a new game.
    fn new() -> Self;

    /// Return the current turn.
    fn turn(&self) -> Turn;

    /// Encode the current board state in form of a one-hot vector, in perspective of the current player (i.e. the player who is about to make a move).
    /// Returned vector should have shape of `BOARD_SHAPE`.
    fn board_state(&self) -> BitVec;

    /// Return the number of possible actions in the current state.
    fn possible_action_count(&self) -> usize;

    /// Return all possible actions in form of a one-hot vector. Returned vector should have length of `POSSIBLE_ACTION_COUNT`.
    fn possible_actions(&self) -> BitVec;

    /// Test if the given action is available.
    fn is_action_available(&self, action: usize) -> bool;

    /// Take the given action and return the game status.
    /// If the action is not available, return `None`.
    fn take_action(&mut self, action: usize) -> Option<Status>;

    /// Return games that represents same game state but have different board state.
    /// For example, in chess, the board can be flipped.
    fn augment(&self) -> Vec<Self>;
}

#[cfg(feature = "serde")]
pub trait SerdeGame
where
    Self: serde::Serialize + serde::DeserializeOwned,
{
}

#[cfg(not(feature = "serde"))]
pub trait SerdeGame {}

/// Represents a game that can be played by a human.
pub trait PlayableGame {
    /// Display the board in a human-readable format.
    fn display_board(&self) -> String;

    /// Parse the given action. If the action is invalid, return `None`.
    fn parse_action(&self, input: &str) -> Option<usize>;
}
