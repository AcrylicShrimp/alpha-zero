use crate::{BoardShape, Game, PlayableGame, SerdeGame, Status, Turn};
use bitvec::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Gomoku9 {
    turn: Turn,
    possible_action_count: usize,
    board_player1: BitVec,
    board_player2: BitVec,
}

impl Gomoku9 {
    pub const SERIAL_STONE_COUNT: usize = 5;

    fn has_player1_stone_at(&self, index: usize) -> bool {
        self.board_player1
            .get(index)
            .map(|occupied| *occupied)
            .unwrap_or(false)
    }

    fn has_player2_stone_at(&self, index: usize) -> bool {
        self.board_player2
            .get(index)
            .map(|occupied| *occupied)
            .unwrap_or(false)
    }

    fn place_player1_stone_at(&mut self, index: usize) {
        self.board_player1.set(index, true);
        self.possible_action_count -= 1;
    }

    fn place_player2_stone_at(&mut self, index: usize) {
        self.board_player2.set(index, true);
        self.possible_action_count -= 1;
    }

    fn count_serial_stones(&self, turn: Turn, index: usize, offset: &[(isize, isize)]) -> usize {
        let board = match turn {
            Turn::Player1 => &self.board_player1,
            Turn::Player2 => &self.board_player2,
        };

        let board_shape = Self::BOARD_SHAPE;
        let height = board_shape.height as isize;
        let width = board_shape.width as isize;

        let x = (index as isize % width) as isize;
        let y = (index as isize / width) as isize;
        let mut count = 0;

        for &(offset_x, offset_y) in offset {
            let x = x + offset_x;
            let y = y + offset_y;
            if x < 0 || width <= x || y < 0 || height <= y {
                break;
            }

            if !board
                .get((y * width + x) as usize)
                .map(|occupied| *occupied)
                .unwrap_or(false)
            {
                break;
            }

            count += 1;
        }

        count
    }

    fn display_stone_at(&self, x: usize, y: usize) -> char {
        let index = y * 9 + x;

        if self.has_player1_stone_at(index) {
            'X'
        } else if self.has_player2_stone_at(index) {
            'O'
        } else {
            ' '
        }
    }
}

impl Game for Gomoku9 {
    const BOARD_SHAPE: BoardShape = BoardShape::new(3, 9, 9);

    const POSSIBLE_ACTION_COUNT: usize = 81;

    fn new() -> Self {
        Self {
            turn: Turn::Player1,
            possible_action_count: Self::POSSIBLE_ACTION_COUNT,
            board_player1: BitVec::repeat(false, Self::BOARD_SHAPE.single_layer_size()),
            board_player2: BitVec::repeat(false, Self::BOARD_SHAPE.single_layer_size()),
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
        self.possible_action_count
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

        match self.turn {
            Turn::Player1 => {
                self.place_player1_stone_at(action);
            }
            Turn::Player2 => {
                self.place_player2_stone_at(action);
            }
        };

        let horizontal_count =
            1 + self.count_serial_stones(
                self.turn,
                action,
                &[(-1, 0), (-2, 0), (-3, 0), (-4, 0), (-5, 0)],
            ) + self.count_serial_stones(
                self.turn,
                action,
                &[(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)],
            );
        let vertical_count =
            1 + self.count_serial_stones(
                self.turn,
                action,
                &[(0, -1), (0, -2), (0, -3), (0, -4), (0, -5)],
            ) + self.count_serial_stones(
                self.turn,
                action,
                &[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
            );
        let diagonal_lt_rb_count =
            1 + self.count_serial_stones(
                self.turn,
                action,
                &[(-1, -1), (-2, -2), (-3, -3), (-4, -4), (-5, -5)],
            ) + self.count_serial_stones(
                self.turn,
                action,
                &[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],
            );
        let diagonal_lb_rt_count =
            1 + self.count_serial_stones(
                self.turn,
                action,
                &[(-1, 1), (-2, 2), (-3, 3), (-4, 4), (-5, 5)],
            ) + self.count_serial_stones(
                self.turn,
                action,
                &[(1, -1), (2, -2), (3, -3), (4, -4), (5, -5)],
            );

        let status = if horizontal_count == Self::SERIAL_STONE_COUNT
            || vertical_count == Self::SERIAL_STONE_COUNT
            || diagonal_lt_rb_count == Self::SERIAL_STONE_COUNT
            || diagonal_lb_rt_count == Self::SERIAL_STONE_COUNT
        {
            match self.turn {
                Turn::Player1 => Status::Player1Won,
                Turn::Player2 => Status::Player2Won,
            }
        } else if self.possible_action_count == 0 {
            Status::Draw
        } else {
            Status::Ongoing
        };

        self.turn = self.turn.opposite();

        Some(status)
    }
}

impl SerdeGame for Gomoku9 {}

impl PlayableGame for Gomoku9 {
    fn display_board(&self) -> String {
        let mut rows = Vec::with_capacity(4);

        rows.push(" 0 1 2 3 4 5 6 7 8".to_owned());

        for y in 0..9 {
            let (x0, x1, x2, x3, x4, x5, x6, x7, x8) = (
                self.display_stone_at(0, y),
                self.display_stone_at(1, y),
                self.display_stone_at(2, y),
                self.display_stone_at(3, y),
                self.display_stone_at(4, y),
                self.display_stone_at(5, y),
                self.display_stone_at(6, y),
                self.display_stone_at(7, y),
                self.display_stone_at(8, y),
            );

            rows.push(format!(
                "{}{} {} {} {} {} {} {} {} {}",
                y, x0, x1, x2, x3, x4, x5, x6, x7, x8
            ));
        }

        rows.join("\n")
    }

    fn parse_action(&self, input: &str) -> Option<usize> {
        let mut slices = input.trim().split_ascii_whitespace();
        let x = slices.next()?.parse::<usize>().ok()?;
        let y = slices.next()?.parse::<usize>().ok()?;
        Some(y * 9 + x)
    }
}
