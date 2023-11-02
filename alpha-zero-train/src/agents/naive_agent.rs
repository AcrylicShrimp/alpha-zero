use crate::agent::Agent;
use game::{Game, Status};
use rand::{rngs::ThreadRng, seq::IteratorRandom, thread_rng};

pub struct NaiveAgent<G>
where
    G: Game,
{
    game: G,
    rng: ThreadRng,
}

impl<G> NaiveAgent<G>
where
    G: Game,
{
    pub fn new() -> Self {
        Self {
            game: G::new(),
            rng: thread_rng(),
        }
    }

    pub fn game(&self) -> &G {
        &self.game
    }
}

impl<G> Agent<G> for NaiveAgent<G>
where
    G: Game,
{
    fn select_action(&mut self) -> Option<usize> {
        let turn = self.game.turn();

        for action in 0..G::POSSIBLE_ACTION_COUNT {
            if !self.game.is_action_available(action) {
                continue;
            }

            let won = self
                .game
                .clone()
                .take_action(action)
                .map(|status| match status {
                    status @ (Status::Player1Won | Status::Player2Won) => status.is_turn(turn),
                    _ => false,
                })
                .unwrap_or(false);

            if won {
                return Some(action);
            }
        }

        let action = (0..G::POSSIBLE_ACTION_COUNT)
            .into_iter()
            .filter(|action| self.game.is_action_available(*action))
            .choose(&mut self.rng);

        action
    }

    fn take_action(&mut self, action: usize) -> Option<Status> {
        self.game.take_action(action)
    }

    fn ensure_action_exists(&mut self, _action: usize) {
        // do nothing
    }
}
