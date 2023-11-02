use game::Game;
use std::fmt::Debug;

/// A trajectory of a game.
#[derive(Clone)]
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

impl<G> Debug for Trajectory<G>
where
    G: Game,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Trajectory")
            .field("policy", &self.policy)
            .field("z", &self.z)
            .finish()
    }
}
