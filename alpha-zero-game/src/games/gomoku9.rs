use crate::{BoardShape, Game, SerdeGame, Status, Turn};
use bitvec::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Gomoku9 {
    turn: Turn,
    board_player1: BitVec,
    board_player2: BitVec,
}

impl Gomoku9 {
    fn has_player1_stone_at(&self, index: usize) -> bool {
        (self.board_player1 >> index) & 1 == 1
    }

    fn has_player2_stone_at(&self, index: usize) -> bool {
        (self.board_player2 >> index) & 1 == 1
    }

    fn place_player1_stone_at(&mut self, index: usize) {
        self.board_player1 |= 1 << index;
    }

    fn place_player2_stone_at(&mut self, index: usize) {
        self.board_player2 |= 1 << index;
    }
}

impl Game for Gomoku9 {
    const BOARD_SHAPE: BoardShape = BoardShape::new(3, 3, 3);

    const POSSIBLE_ACTION_COUNT: usize = 9;

    fn new() -> Self {
        Self {
            turn: Turn::Player1,
            board_player1: 0,
            board_player2: 0,
        }
    }

    fn turn(&self) -> Turn {
        self.turn
    }

    fn board_state(&self, flip_perspective: bool) -> BitVec {
        let mut board = BitVec::repeat(false, Self::BOARD_SHAPE.size());

        let offset = if (self.turn == Turn::Player1) ^ flip_perspective {
            0
        } else {
            Self::BOARD_SHAPE.single_layer_size()
        };

        for index in 0..Self::BOARD_SHAPE.single_layer_size() {
            if self.has_player1_stone_at(index) {
                board.set(index + offset, true);
            }
        }

        let offset = if (self.turn == Turn::Player2) ^ flip_perspective {
            0
        } else {
            Self::BOARD_SHAPE.single_layer_size()
        };

        for index in 0..Self::BOARD_SHAPE.single_layer_size() {
            if self.has_player2_stone_at(index) {
                board.set(index + offset, true);
            }
        }

        if self.turn == Turn::Player1 {
            board[Self::BOARD_SHAPE.single_layer_size() * 2..].fill(true);
        }

        board
    }

    fn possible_action_count(&self) -> usize {
        let mut count = 0;

        for action in 0..Self::POSSIBLE_ACTION_COUNT {
            if self.is_action_available(action) {
                count += 1;
            }
        }

        count
    }

    fn possible_actions(&self) -> BitVec {
        let mut board = BitVec::repeat(true, Self::POSSIBLE_ACTION_COUNT);

        for index in 0..Self::BOARD_SHAPE.single_layer_size() {
            if self.has_player1_stone_at(index) || self.has_player2_stone_at(index) {
                board.set(index, false);
            }
        }

        board
    }

    fn is_action_available(&self, action: usize) -> bool {
        !self.has_player1_stone_at(action) && !self.has_player2_stone_at(action)
    }

    fn take_action(&mut self, action: usize) -> Option<Status> {
        if !self.is_action_available(action) {
            return None;
        }

        let status = match self.turn {
            Turn::Player1 => {
                self.place_player1_stone_at(action);

                if self.board_player1 & 0b111 == 0b111
                    || self.board_player1 & 0b111_000 == 0b111_000
                    || self.board_player1 & 0b111_000_000 == 0b111_000_000
                    || self.board_player1 & 0b100_100_100 == 0b100_100_100
                    || self.board_player1 & 0b010_010_010 == 0b010_010_010
                    || self.board_player1 & 0b001_001_001 == 0b001_001_001
                    || self.board_player1 & 0b100_010_001 == 0b100_010_001
                    || self.board_player1 & 0b001_010_100 == 0b001_010_100
                {
                    Status::Player1Won
                } else if self.board_player1 | self.board_player2 == 0b111_111_111 {
                    Status::Draw
                } else {
                    Status::Ongoing
                }
            }
            Turn::Player2 => {
                self.place_player2_stone_at(action);

                if self.board_player2 & 0b111 == 0b111
                    || self.board_player2 & 0b111_000 == 0b111_000
                    || self.board_player2 & 0b111_000_000 == 0b111_000_000
                    || self.board_player2 & 0b100_100_100 == 0b100_100_100
                    || self.board_player2 & 0b010_010_010 == 0b010_010_010
                    || self.board_player2 & 0b001_001_001 == 0b001_001_001
                    || self.board_player2 & 0b100_010_001 == 0b100_010_001
                    || self.board_player2 & 0b001_010_100 == 0b001_010_100
                {
                    Status::Player2Won
                } else if self.board_player1 | self.board_player2 == 0b111_111_111 {
                    Status::Draw
                } else {
                    Status::Ongoing
                }
            }
        };

        self.turn = self.turn.opposite();

        Some(status)
    }
}

impl SerdeGame for Gomoku9 {}
