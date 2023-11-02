use game::{Game, Status};

/// An agent that can play a game.
pub trait Agent<G>
where
    G: Game,
{
    /// Select the action to play.
    /// Returns `None` if the game is already finished (no action available).
    fn select_action(&mut self) -> Option<usize>;

    /// Takes the action and returns the status of the game.
    /// Returns `None` if:
    /// - The action is illegal.
    /// - The game is already finished.
    /// - The action is not applicable due to the internal state of the agent. (e.g. MCTS in the AlphaZero agent is not ready yet.)
    ///
    /// If you eliminate the last case, call `ensure_action_exists` before calling this method.
    fn take_action(&mut self, action: usize) -> Option<Status>;

    /// Ensures that the action exists in the internal state of the agent.
    fn ensure_action_exists(&mut self, action: usize);
}
