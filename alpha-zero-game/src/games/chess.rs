use crate::{BoardShape, Game, SerdeGame, Status, Turn};
use bitvec::vec::BitVec;
use chess::*;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Chess {
    chess: chess::Game,
}

#[repr(u8)]
enum Direction {
    North,
    NorthEast,
    East,
    SouthEast,
    South,
    SouthWest,
    West,
    NorthWest,
}

#[repr(u8)]
enum KnightMove {
    NorthNorthEast,
    NorthEastEast,
    SouthEastEast,
    SouthSouthEast,
    SouthSouthWest,
    SouthWestWest,
    NorthWestWest,
    NorthNorthWest,
}

#[repr(u8)]
enum PawnPromotionWay {
    NorthOrSouth,
    /// Capture to the east.
    East,
    /// Capture to the west.
    West,
}

impl Chess {
    /// Converts a chess move to an index in the range [0, 4671].
    pub fn move_to_index(chess_move: ChessMove) -> usize {
        let source = chess_move.get_source();
        let dest = chess_move.get_dest();

        let file_diff = i8::abs_diff(dest.get_file() as i8, source.get_file() as i8);
        let rank_diff = i8::abs_diff(dest.get_rank() as i8, source.get_rank() as i8);

        let movement;

        if (chess_move.get_promotion().is_none()
            || chess_move.get_promotion() == Some(Piece::Queen))
            && (file_diff == 0 || rank_diff == 0 || file_diff == rank_diff)
        {
            // it looks like a queen move; there are 56 possible moves
            // note that this includes pawn promotions to queen
            let (direction, distance) = if file_diff == 0 {
                // vertical
                let direction = if dest.get_rank() > source.get_rank() {
                    Direction::North as u8
                } else {
                    Direction::South as u8
                };
                let distance = rank_diff as u8;
                (direction, distance)
            } else if rank_diff == 0 {
                // horizontal
                let direction = if dest.get_file() > source.get_file() {
                    Direction::East as u8
                } else {
                    Direction::West as u8
                };
                let distance = file_diff as u8;
                (direction, distance)
            } else {
                // diagonal
                let direction = if dest.get_file() > source.get_file() {
                    if dest.get_rank() > source.get_rank() {
                        Direction::NorthEast as u8
                    } else {
                        Direction::SouthEast as u8
                    }
                } else {
                    if dest.get_rank() > source.get_rank() {
                        Direction::NorthWest as u8
                    } else {
                        Direction::SouthWest as u8
                    }
                };
                let distance = file_diff as u8;
                (direction, distance)
            };

            // 8 directions, 7 possible distances = 56 possible moves
            movement = direction * 7 + distance - 1;
        } else if chess_move.get_promotion().is_none() {
            // if there's no queen move, it's a knight move
            // there are 8 possible moves

            let file_diff = dest.get_file() as i8 - source.get_file() as i8;
            let rank_diff = dest.get_rank() as i8 - source.get_rank() as i8;

            let direction = if file_diff == 1 && rank_diff == 2 {
                KnightMove::NorthNorthEast as u8
            } else if file_diff == 2 && rank_diff == 1 {
                KnightMove::NorthEastEast as u8
            } else if file_diff == 2 && rank_diff == -1 {
                KnightMove::SouthEastEast as u8
            } else if file_diff == 1 && rank_diff == -2 {
                KnightMove::SouthSouthEast as u8
            } else if file_diff == -1 && rank_diff == -2 {
                KnightMove::SouthSouthWest as u8
            } else if file_diff == -2 && rank_diff == -1 {
                KnightMove::SouthWestWest as u8
            } else if file_diff == -2 && rank_diff == 1 {
                KnightMove::NorthWestWest as u8
            } else if file_diff == -1 && rank_diff == 2 {
                KnightMove::NorthNorthWest as u8
            } else {
                unreachable!()
            };

            // 8 directions, 1 possible distance = 8 possible moves
            movement = 56 + direction;
        } else {
            // it's a pawn underpromotion
            // there are 9 possible moves

            let target_piece = chess_move.get_promotion().unwrap();

            let piece = match target_piece {
                Piece::Knight => 0,
                Piece::Bishop => 1,
                Piece::Rook => 2,
                _ => unreachable!(),
            };

            let way = if file_diff == 0 {
                PawnPromotionWay::NorthOrSouth as u8
            } else if dest.get_file() > source.get_file() {
                PawnPromotionWay::East as u8
            } else {
                PawnPromotionWay::West as u8
            };

            // 3 pieces, 3 promotions = 9 possible moves
            movement = 56 + 8 + piece * 3 + way;
        }

        (source.get_file() as usize * 8 + source.get_rank() as usize) * 73 + movement as usize
    }

    /// Converts an index in the range [0, 4671] to a chess move.
    pub fn index_to_move(index: usize) -> ChessMove {
        let source = index / 73;
        let file = File::from_index(source / 8);
        let rank = Rank::from_index(source % 8);
        let source = Square::make_square(rank, file);

        match index % 73 {
            index @ 0..=55 => {
                // queen move
                let direction = index / 7;
                let distance = index % 7 + 1;

                return match direction {
                    0 => {
                        let dest =
                            Square::make_square(Rank::from_index(rank.to_index() + distance), file);
                        ChessMove::new(source, dest, None)
                    }
                    1 => {
                        let dest = Square::make_square(
                            Rank::from_index(rank.to_index() + distance),
                            File::from_index(file.to_index() + distance),
                        );
                        ChessMove::new(source, dest, None)
                    }
                    2 => {
                        let dest =
                            Square::make_square(rank, File::from_index(file.to_index() + distance));
                        ChessMove::new(source, dest, None)
                    }
                    3 => {
                        let dest = Square::make_square(
                            Rank::from_index(rank.to_index() - distance),
                            File::from_index(file.to_index() + distance),
                        );
                        ChessMove::new(source, dest, None)
                    }
                    4 => {
                        let dest =
                            Square::make_square(Rank::from_index(rank.to_index() - distance), file);
                        ChessMove::new(source, dest, None)
                    }
                    5 => {
                        let dest = Square::make_square(
                            Rank::from_index(rank.to_index() - distance),
                            File::from_index(file.to_index() - distance),
                        );
                        ChessMove::new(source, dest, None)
                    }
                    6 => {
                        let dest =
                            Square::make_square(rank, File::from_index(file.to_index() - distance));
                        ChessMove::new(source, dest, None)
                    }
                    7 => {
                        let dest = Square::make_square(
                            Rank::from_index(rank.to_index() + distance),
                            File::from_index(file.to_index() - distance),
                        );
                        ChessMove::new(source, dest, None)
                    }
                    _ => unreachable!(),
                };
            }
            index @ 56..=63 => {
                // knight move
                let direction = index - 56;

                return match direction {
                    0 => {
                        let dest = Square::make_square(
                            Rank::from_index(rank.to_index() + 2),
                            File::from_index(file.to_index() + 1),
                        );
                        ChessMove::new(source, dest, None)
                    }
                    1 => {
                        let dest = Square::make_square(
                            Rank::from_index(rank.to_index() + 1),
                            File::from_index(file.to_index() + 2),
                        );
                        ChessMove::new(source, dest, None)
                    }
                    2 => {
                        let dest = Square::make_square(
                            Rank::from_index(rank.to_index() - 1),
                            File::from_index(file.to_index() + 2),
                        );
                        ChessMove::new(source, dest, None)
                    }
                    3 => {
                        let dest = Square::make_square(
                            Rank::from_index(rank.to_index() - 2),
                            File::from_index(file.to_index() + 1),
                        );
                        ChessMove::new(source, dest, None)
                    }
                    4 => {
                        let dest = Square::make_square(
                            Rank::from_index(rank.to_index() - 2),
                            File::from_index(file.to_index() - 1),
                        );
                        ChessMove::new(source, dest, None)
                    }
                    5 => {
                        let dest = Square::make_square(
                            Rank::from_index(rank.to_index() - 1),
                            File::from_index(file.to_index() - 2),
                        );
                        ChessMove::new(source, dest, None)
                    }
                    6 => {
                        let dest = Square::make_square(
                            Rank::from_index(rank.to_index() + 1),
                            File::from_index(file.to_index() - 2),
                        );
                        ChessMove::new(source, dest, None)
                    }
                    7 => {
                        let dest = Square::make_square(
                            Rank::from_index(rank.to_index() + 2),
                            File::from_index(file.to_index() - 1),
                        );
                        ChessMove::new(source, dest, None)
                    }
                    _ => unreachable!(),
                };
            }
            index @ 64..=72 => {
                // pawn underpromotion
                let piece = (index - 64) / 3;
                let way = (index - 64) % 3;

                let target_piece = match piece {
                    0 => Piece::Knight,
                    1 => Piece::Bishop,
                    2 => Piece::Rook,
                    _ => unreachable!(),
                };
                let dest = match way {
                    0 => Square::make_square(Rank::from_index(rank.to_index() + 1), file),
                    1 => Square::make_square(
                        Rank::from_index(rank.to_index() + 1),
                        File::from_index(file.to_index() + 1),
                    ),
                    2 => Square::make_square(
                        Rank::from_index(rank.to_index() + 1),
                        File::from_index(file.to_index() - 1),
                    ),
                    _ => unreachable!(),
                };

                return ChessMove::new(source, dest, Some(target_piece));
            }
            _ => unreachable!(),
        }
    }
}

impl Game for Chess {
    const BOARD_SHAPE: BoardShape = BoardShape::new(12, 8, 8);

    const POSSIBLE_ACTION_COUNT: usize = 4672;

    fn new() -> Self {
        Self {
            chess: chess::Game::new(),
        }
    }

    fn turn(&self) -> Turn {
        match self.chess.side_to_move() {
            Color::White => Turn::Player1,
            Color::Black => Turn::Player2,
        }
    }

    fn board_state(&self) -> BitVec {
        let board = self.chess.current_position();
        let mut state = BitVec::with_capacity(Self::BOARD_SHAPE.size());

        for (piece_index, piece) in [
            Piece::Pawn,
            Piece::Knight,
            Piece::Bishop,
            Piece::Rook,
            Piece::Queen,
            Piece::King,
        ]
        .iter()
        .enumerate()
        {
            let white = board.pieces(*piece) & board.color_combined(Color::White);
            let black = board.pieces(*piece) & board.color_combined(Color::Black);

            for square in white {
                let square_index = square.to_index();
                let index = piece_index * Self::BOARD_SHAPE.single_layer_size() + square_index;
                state.set(index, true);
            }

            for square in black {
                let square_index = square.to_index();
                let index = piece_index * Self::BOARD_SHAPE.single_layer_size()
                    + square_index
                    + 6 * Self::BOARD_SHAPE.single_layer_size();
                state.set(index, true);
            }
        }

        state
    }

    fn possible_action_count(&self) -> usize {
        let board = self.chess.current_position();
        let move_gen = MoveGen::new_legal(&board);
        move_gen.len()
    }

    fn possible_actions(&self) -> BitVec {
        let board = self.chess.current_position();
        let move_gen = MoveGen::new_legal(&board);
        let mut actions = BitVec::repeat(false, Self::POSSIBLE_ACTION_COUNT);

        for action in move_gen {
            let index = Self::move_to_index(action);
            actions.set(index, true);
        }

        actions
    }

    fn is_action_available(&self, action: usize) -> bool {
        let board = self.chess.current_position();
        let chess_move = Self::index_to_move(action);
        board.legal(chess_move)
    }

    fn take_action(&mut self, action: usize) -> Option<Status> {
        let chess_move = Self::index_to_move(action);

        if !self.chess.make_move(chess_move) {
            return None;
        }

        match self.chess.result() {
            Some(status) => match status {
                GameResult::WhiteCheckmates => Some(Status::Player1Won),
                GameResult::WhiteResigns => Some(Status::Player2Won),
                GameResult::BlackCheckmates => Some(Status::Player2Won),
                GameResult::BlackResigns => Some(Status::Player1Won),
                GameResult::Stalemate => Some(Status::Draw),
                GameResult::DrawAccepted => Some(Status::Draw),
                GameResult::DrawDeclared => Some(Status::Ongoing),
            },
            None => Some(Status::Ongoing),
        }
    }

    fn augment(&self) -> Vec<Self> {
        vec![]
    }
}

impl SerdeGame for Chess {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_encoding() {
        let chess = Chess::new();
        let move_gen = MoveGen::new_legal(&chess.chess.current_position());
        let count = move_gen.len();
        let mut indices = HashSet::with_capacity(count);

        for chess_move in move_gen {
            let index = Chess::move_to_index(chess_move);
            let chess_move2 = Chess::index_to_move(index);

            assert_eq!(chess_move, chess_move2);

            indices.insert(index);
        }

        assert_eq!(indices.len(), count);
    }
}
