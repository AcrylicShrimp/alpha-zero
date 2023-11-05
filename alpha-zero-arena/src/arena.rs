use game::{Game, PlayableGame, Status, Turn};
use nn::NN;
use tch::Device;
use train::{
    agent::Agent,
    agents::{ActionSamplingMode, AlphaZeroAgent},
    encode::encode_nn_input,
    parallel_mcts_executor::{MCTSExecutorConfig, ParallelMCTSExecutor},
};

pub struct Arena<G>
where
    G: Game + PlayableGame,
{
    device: Device,
    nn: NN<G>,
    mcts_executor: ParallelMCTSExecutor,
}

impl<G> Arena<G>
where
    G: Game + PlayableGame,
{
    pub fn new(device: Device, nn: NN<G>) -> Self {
        Self {
            device,
            nn,
            mcts_executor: ParallelMCTSExecutor::new(MCTSExecutorConfig { num_threads: None })
                .unwrap(),
        }
    }

    pub fn play(&self) {
        let mut game = G::new();
        let mut agent = vec![AlphaZeroAgent::new(self.device, &self.nn)];

        loop {
            let input = encode_nn_input(self.device, 1, std::iter::once(&game));
            let (v, _) = self.nn.forward(&input, false);
            println!("v={}", v.double_value(&[]));

            let action = match game.turn() {
                Turn::Player1 => loop {
                    println!();
                    println!("{}", game.display_board());
                    println!();
                    println!("your turn, enter action: ");

                    let mut action = String::new();
                    std::io::stdin()
                        .read_line(&mut action)
                        .expect("failed to read line");

                    if let Some(action) = game.parse_action(&action) {
                        if game.is_action_available(action) {
                            break action;
                        }
                    }

                    println!();
                    println!("invalid action, try again");
                },
                Turn::Player2 => {
                    self.mcts_executor.execute(
                        self.device,
                        1600,
                        1,
                        1f32,
                        0.03f32,
                        0.25f32,
                        &self.nn,
                        &agent,
                    );
                    agent[0]
                        .sample_action_from_policy(ActionSamplingMode::Best)
                        .unwrap()
                        .0
                }
            };
            let status = game.take_action(action).unwrap();

            println!();
            println!("{}", game.display_board());

            match status {
                Status::Ongoing => {}
                Status::Player1Won => {
                    println!();
                    println!("you win!");
                    return;
                }
                Status::Player2Won => {
                    println!();
                    println!("alpha-zero won!");
                    return;
                }
                Status::Draw => {
                    println!();
                    println!("draw");
                    return;
                }
            }

            agent[0].ensure_action_exists(action);
            agent[0].take_action(action);
        }
    }
}
