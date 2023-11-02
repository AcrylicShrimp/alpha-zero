use game::Game;

/// A trajectory of a game.
pub struct Trajectory<G>
where
    G: Game,
{
    /// The game that was played.
    pub game: G,
    /// The policy vector obtained from the MCTS.
    pub policy: Vec<f32>,
    /// The z observed from the game.
    pub z: f32,
}
