use crate::{BoardShape, Game, SerdeGame, Status, Turn};
use bitvec::vec::BitVec;
use chess::*;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Chess {
    chess: chess::Game,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

impl Direction {
    pub fn from_index(index: u8) -> Self {
        match index {
            0 => Direction::North,
            1 => Direction::NorthEast,
            2 => Direction::East,
            3 => Direction::SouthEast,
            4 => Direction::South,
            5 => Direction::SouthWest,
            6 => Direction::West,
            7 => Direction::NorthWest,
            _ => unreachable!(),
        }
    }

    pub fn to_dest(self, source: Square, distance: usize) -> Option<Square> {
        let rank = source.get_rank();
        let file = source.get_file();

        match self {
            Direction::North => {
                if rank.to_index() + distance > 7 {
                    return None;
                }

                Some(Square::make_square(
                    Rank::from_index(rank.to_index() + distance),
                    file,
                ))
            }
            Direction::NorthEast => {
                if rank.to_index() + distance > 7 || file.to_index() + distance > 7 {
                    return None;
                }

                Some(Square::make_square(
                    Rank::from_index(rank.to_index() + distance),
                    File::from_index(file.to_index() + distance),
                ))
            }
            Direction::East => {
                if file.to_index() + distance > 7 {
                    return None;
                }

                Some(Square::make_square(
                    rank,
                    File::from_index(file.to_index() + distance),
                ))
            }
            Direction::SouthEast => {
                if rank.to_index() < distance || file.to_index() + distance > 7 {
                    return None;
                }

                Some(Square::make_square(
                    Rank::from_index(rank.to_index() - distance),
                    File::from_index(file.to_index() + distance),
                ))
            }
            Direction::South => {
                if rank.to_index() < distance {
                    return None;
                }

                Some(Square::make_square(
                    Rank::from_index(rank.to_index() - distance),
                    file,
                ))
            }
            Direction::SouthWest => {
                if rank.to_index() < distance || file.to_index() < distance {
                    return None;
                }

                Some(Square::make_square(
                    Rank::from_index(rank.to_index() - distance),
                    File::from_index(file.to_index() - distance),
                ))
            }
            Direction::West => {
                if file.to_index() < distance {
                    return None;
                }

                Some(Square::make_square(
                    rank,
                    File::from_index(file.to_index() - distance),
                ))
            }
            Direction::NorthWest => {
                if rank.to_index() + distance > 7 || file.to_index() < distance {
                    return None;
                }

                Some(Square::make_square(
                    Rank::from_index(rank.to_index() + distance),
                    File::from_index(file.to_index() - distance),
                ))
            }
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

impl KnightMove {
    pub fn from_index(index: u8) -> Self {
        match index {
            0 => KnightMove::NorthNorthEast,
            1 => KnightMove::NorthEastEast,
            2 => KnightMove::SouthEastEast,
            3 => KnightMove::SouthSouthEast,
            4 => KnightMove::SouthSouthWest,
            5 => KnightMove::SouthWestWest,
            6 => KnightMove::NorthWestWest,
            7 => KnightMove::NorthNorthWest,
            _ => unreachable!(),
        }
    }

    pub fn to_dest(self, source: Square) -> Option<Square> {
        let rank = source.get_rank();
        let file = source.get_file();

        match self {
            KnightMove::NorthNorthEast => {
                if rank.to_index() + 2 > 7 || file.to_index() + 1 > 7 {
                    return None;
                }

                Some(Square::make_square(
                    Rank::from_index(rank.to_index() + 2),
                    File::from_index(file.to_index() + 1),
                ))
            }
            KnightMove::NorthEastEast => {
                if rank.to_index() + 1 > 7 || file.to_index() + 2 > 7 {
                    return None;
                }

                Some(Square::make_square(
                    Rank::from_index(rank.to_index() + 1),
                    File::from_index(file.to_index() + 2),
                ))
            }
            KnightMove::SouthEastEast => {
                if rank.to_index() < 1 || file.to_index() + 2 > 7 {
                    return None;
                }

                Some(Square::make_square(
                    Rank::from_index(rank.to_index() - 1),
                    File::from_index(file.to_index() + 2),
                ))
            }
            KnightMove::SouthSouthEast => {
                if rank.to_index() < 2 || file.to_index() + 1 > 7 {
                    return None;
                }

                Some(Square::make_square(
                    Rank::from_index(rank.to_index() - 2),
                    File::from_index(file.to_index() + 1),
                ))
            }
            KnightMove::SouthSouthWest => {
                if rank.to_index() < 2 || file.to_index() < 1 {
                    return None;
                }

                Some(Square::make_square(
                    Rank::from_index(rank.to_index() - 2),
                    File::from_index(file.to_index() - 1),
                ))
            }
            KnightMove::SouthWestWest => {
                if rank.to_index() < 1 || file.to_index() < 2 {
                    return None;
                }

                Some(Square::make_square(
                    Rank::from_index(rank.to_index() - 1),
                    File::from_index(file.to_index() - 2),
                ))
            }
            KnightMove::NorthWestWest => {
                if rank.to_index() + 1 > 7 || file.to_index() < 2 {
                    return None;
                }

                Some(Square::make_square(
                    Rank::from_index(rank.to_index() + 1),
                    File::from_index(file.to_index() - 2),
                ))
            }
            KnightMove::NorthNorthWest => {
                if rank.to_index() + 2 > 7 || file.to_index() < 1 {
                    return None;
                }

                Some(Square::make_square(
                    Rank::from_index(rank.to_index() + 2),
                    File::from_index(file.to_index() - 1),
                ))
            }
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PawnPromotionWay {
    NorthOrSouth,
    /// Capture to the east.
    East,
    /// Capture to the west.
    West,
}

impl PawnPromotionWay {
    pub fn from_index(index: u8) -> Self {
        match index {
            0 => PawnPromotionWay::NorthOrSouth,
            1 => PawnPromotionWay::East,
            2 => PawnPromotionWay::West,
            _ => unreachable!(),
        }
    }
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
    pub fn index_to_move(board: &Board, index: usize) -> Option<ChessMove> {
        let source = index / 73;
        let file = File::from_index(source / 8);
        let rank = Rank::from_index(source % 8);
        let source = Square::make_square(rank, file);

        match index % 73 {
            index @ 0..=55 => {
                // queen move
                let direction = Direction::from_index((index / 7) as u8);
                let distance = index % 7 + 1;
                let dest = direction.to_dest(source, distance as usize)?;
                let promotion = if (direction == Direction::North || direction == Direction::South)
                    && board.piece_on(source) == Some(Piece::Pawn)
                    && (dest.get_rank() == Rank::First || dest.get_rank() == Rank::Eighth)
                {
                    Some(Piece::Queen)
                } else {
                    None
                };

                return Some(ChessMove::new(source, dest, promotion));
            }
            index @ 56..=63 => {
                // knight move
                let direction = KnightMove::from_index((index - 56) as u8);
                let dest = direction.to_dest(source)?;

                return Some(ChessMove::new(source, dest, None));
            }
            index @ 64..=72 => {
                // pawn underpromotion
                let piece = (index - 64) / 3;
                let way = PawnPromotionWay::from_index(((index - 64) % 3) as u8);
                let dest_rank = if board.color_on(source) == Some(Color::White) {
                    Rank::Eighth
                } else {
                    Rank::First
                };

                let target_piece = match piece {
                    0 => Piece::Knight,
                    1 => Piece::Bishop,
                    2 => Piece::Rook,
                    _ => unreachable!(),
                };
                let dest = match way {
                    PawnPromotionWay::NorthOrSouth => Square::make_square(dest_rank, file),
                    PawnPromotionWay::East => {
                        if file.to_index() == 7 {
                            return None;
                        }

                        Square::make_square(dest_rank, File::from_index(file.to_index() + 1))
                    }
                    PawnPromotionWay::West => {
                        if file.to_index() == 0 {
                            return None;
                        }

                        Square::make_square(dest_rank, File::from_index(file.to_index() - 1))
                    }
                };

                return Some(ChessMove::new(source, dest, Some(target_piece)));
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
        let mut state = BitVec::repeat(false, Self::BOARD_SHAPE.size());

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
        let chess_move = match Self::index_to_move(&board, action) {
            Some(chess_move) => chess_move,
            None => return false,
        };
        board.legal(chess_move)
    }

    fn take_action(&mut self, action: usize) -> Option<Status> {
        let board = self.chess.current_position();
        let chess_move = Self::index_to_move(&board, action)?;

        if !board.legal(chess_move) {
            return None;
        }

        self.chess.make_move(chess_move);

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
            None if self.chess.can_declare_draw() => {
                // immediately accept the draw if 3-fold repetition or 50-move rule is met
                self.chess.declare_draw();
                self.chess.accept_draw();
                Some(Status::Draw)
            }
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
    use std::{collections::HashSet, str::FromStr};

    fn test_encoding(board: Board) {
        let move_gen = MoveGen::new_legal(&board);
        let count = move_gen.len();
        let mut indices = HashSet::with_capacity(count);

        for chess_move in move_gen {
            let index = Chess::move_to_index(chess_move);
            let chess_move2 = Chess::index_to_move(&board, index).unwrap();

            assert_eq!(chess_move, chess_move2);

            indices.insert(index);
        }

        assert_eq!(indices.len(), count);
    }

    #[test]
    fn test_encoding_initial() {
        test_encoding(Board::default());
    }

    #[test]
    fn test_encoding_fen_1() {
        test_encoding(
            Board::from_str("r2qr1k1/1b5p/ppn1pbp1/1B1p4/3P4/P4NP1/1P1NQPP1/3R1RK1 w - - 0 19")
                .unwrap(),
        );
    }

    #[test]
    fn test_encoding_fen_2() {
        test_encoding(
            Board::from_str("2b5/p2NBp1p/1bp1nPPr/3P4/2pRnr1P/1k1B1Ppp/1P1P1pQP/Rq1N3K b - - 0 1")
                .unwrap(),
        );
    }

    #[test]
    fn test_encoding_fen_3() {
        test_encoding(
            Board::from_str("2qb3R/rn3kPB/p4R1p/P2P2bK/2p1P3/1PBPp1pP/1NrppPp1/3N2Q1 b - - 0 1")
                .unwrap(),
        );
    }

    #[test]
    fn test_encoding_fen_4() {
        test_encoding(
            Board::from_str("1K1N4/R3b3/qP1P2pp/PP1PrBB1/p2np1b1/2pnPPpp/pP1Rr1N1/2k3Q1 b - - 0 1")
                .unwrap(),
        );
    }
}
