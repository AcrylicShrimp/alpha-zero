pub mod agent;
pub mod agents;
pub mod encode;
pub mod parallel_mcts_executor;
pub mod trajectory;

use crate::{
    agent::Agent,
    agents::{ActionSamplingMode, AlphaZeroAgent},
    encode::{encode_nn_input, encode_nn_targets, EncodingPerspective},
};
use agents::NaiveAgent;
use game::{Game, Status, Turn};
use indicatif::{ProgressBar, ProgressStyle};
use log::{info, trace, warn};
use nn::{
    nn_optimizer::{NNOptimizer, NNOptimizerConfig},
    NNConfig, NN,
};
use parallel_mcts_executor::{MCTSExecutorConfig, ParallelMCTSExecutor};
use rand::{seq::IteratorRandom, thread_rng};
use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
};
use tch::{
    nn::{Adam, VarStore},
    Device,
};
use thiserror::Error;
use trajectory::Trajectory;

#[derive(Error, Debug)]
pub enum TrainerBuildError {
    #[error("failed to build thread pool: {0}")]
    ThreadPoolBuildError(#[from] rayon::ThreadPoolBuildError),
    #[error("failed to build neural network optimizer: {0}")]
    TchError(#[from] tch::TchError),
}

/// Neural network save/load configuration.
/// The neural network will be saved every iteration.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TrainerNNSaveConfig {
    /// Path to the neural network (and optimizer state) to be saved or loaded.
    pub path: PathBuf,
    /// Neural network backup interval. The neural network will be backed up every `backup_interval` iterations.
    /// This is useful if you want "freeze" the neural network for a while and see how the agent performs.
    pub backup_interval: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
/// Configuration for the trainer.
pub struct TrainerConfig {
    /// Device to run NN on.
    pub device: Device,
    /// MCTS executor configuration.
    pub mcts_executor_config: MCTSExecutorConfig,
    /// Neural network configuration.
    pub nn_config: NNConfig,
    /// Neural network optimizer configuration.
    pub nn_optimizer_config: NNOptimizerConfig,
    /// Number of trajectories to keep. Rest will be discarded in chronological order.
    pub replay_buffer_size: usize,
    /// Number of episodes to play.
    pub episodes: usize,
    /// Number of MCTS simulations to run.
    pub mcts_count: usize,
    /// Number of nodes to expand in parallel.
    pub batch_size: usize,
    /// A constant that controls the level of exploration.
    pub c_puct: f32,
    /// Dirichlet noise alpha.
    pub alpha: f32,
    /// Dirichlet noise epsilon.
    pub epsilon: f32,
    /// The temperature to use for action sampling.
    pub temperature: f32,
    /// The temperature threshold after which the temperature will be set to 0.
    /// Unit is number of turns.
    pub temperature_threshold: usize,
    /// The number of parameter updates to perform.
    pub parameter_update_count: usize,
    /// The number of trajectories to use for a single parameter update.
    pub parameter_update_batch_size: usize,
    /// Number of matches to play against other agents to test the performance.
    pub match_count: usize,
    /// Neural network save/load configuration.
    pub nn_save_config: Option<TrainerNNSaveConfig>,
}

/// A trainer for an alpha zero agent.
pub struct Trainer<G>
where
    G: Game,
{
    /// Trainer configuration.
    config: TrainerConfig,
    /// VarStore for the neural network.
    vs: VarStore,
    /// Neural network that is being trained.
    nn_optimizer: NNOptimizer<G>,
    /// Parallel MCTS executor.
    mcts_executor: ParallelMCTSExecutor,
}

impl<G> Trainer<G>
where
    G: Game,
{
    /// Creates a new trainer.
    pub fn new(config: TrainerConfig) -> Result<Self, TrainerBuildError> {
        let mut vs = VarStore::new(config.device);
        let nn = NN::new(&vs.root(), config.nn_config.clone());
        let nn_optimizer =
            NNOptimizer::new(&vs, config.nn_optimizer_config.clone(), nn, Adam::default())?;
        let mcts_executor = ParallelMCTSExecutor::new(config.mcts_executor_config.clone())?;

        if let Some(config) = &config.nn_save_config {
            let path = config.path.with_extension("ot");

            if path.is_file() {
                info!("loading neural network from `{}`", path.display());
                vs.load(path)?;
            } else {
                let path = path.with_extension("bak.ot");

                if path.is_file() {
                    info!("loading neural network from `{}`", path.display());
                    vs.load(path)?;
                }
            }
        }

        Ok(Self {
            config,
            vs,
            nn_optimizer,
            mcts_executor,
        })
    }

    /// Returns a reference to the VarStore.
    pub fn vs(&self) -> &VarStore {
        &self.vs
    }

    /// Returns a reference to the neural network.
    pub fn nn(&self) -> &NN<G> {
        &self.nn_optimizer.nn()
    }

    /// Trains the neural network.
    pub fn train(&mut self, iteration: usize) {
        let mut rng = thread_rng();
        let mut replay_buffer = VecDeque::with_capacity(self.config.replay_buffer_size);
        let mut loss_buffer = VecDeque::with_capacity(100);

        for iter in 0..iteration {
            if iter % 10 == 0 {
                info!(
                    "(iter={}/{}) measuring agent performance",
                    iter + 1,
                    iteration
                );

                let (player1_won, player2_won, draw) =
                    self.play_against_naive_agent(self.config.match_count);

                info!(
                    "(iter={}/{}) naive agent won {} times, alpha zero agent won {} times, draw {} times",
                    iter + 1,
                    iteration,
                    player1_won,
                    player2_won,
                    draw
                );

                if let Some(config) = &self.config.nn_save_config {
                    let path = config.path.with_extension("bak.ot");

                    if path.is_file() {
                        if let Some((player1_won, player2_won, draw)) =
                            self.play_against_backuped_nn(path, self.config.match_count)
                        {
                            info!(
                                "(iter={}/{}) old-version agent won {} times, alpha zero agent won {} times, draw {} times",
                                iter + 1,
                                iteration,
                                player1_won,
                                player2_won,
                                draw
                            );
                        }
                    }
                }
            }

            info!("(iter={}/{}) self playing", iter + 1, iteration);

            let trajectories = self.self_play(self.config.episodes);

            trace!("trajectories={:?}", trajectories);

            replay_buffer.extend(trajectories);

            while self.config.replay_buffer_size < replay_buffer.len() {
                replay_buffer.pop_front();
            }

            info!("(iter={}/{}) training", iter + 1, iteration);

            let progress_bar = ProgressBar::new(self.config.parameter_update_count as u64);
            progress_bar.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.white} [{bar:40.green/white}] {pos:>7}/{len:7}")
                    .unwrap()
                    .progress_chars("#>-"),
            );
            progress_bar.tick();

            for _ in 0..self.config.parameter_update_count {
                let trajectories = replay_buffer
                    .iter()
                    .choose_multiple(&mut rng, self.config.parameter_update_batch_size);
                let losses = self.train_nn(trajectories.as_slice());

                loss_buffer.push_back(losses);

                while 100 < loss_buffer.len() {
                    loss_buffer.pop_front();
                }

                progress_bar.inc(1);
            }

            progress_bar.finish();

            let (loss, v_loss, pi_loss) = (
                loss_buffer.iter().cloned().map(|l| l.0).sum::<f32>() / loss_buffer.len() as f32,
                loss_buffer.iter().cloned().map(|l| l.1).sum::<f32>() / loss_buffer.len() as f32,
                loss_buffer.iter().cloned().map(|l| l.2).sum::<f32>() / loss_buffer.len() as f32,
            );

            info!(
                "(iter={}/{}) loss={:.4} (v_loss={:.4}, pi_loss={:.4})",
                iter + 1,
                iteration,
                loss,
                v_loss,
                pi_loss
            );

            if let Some(config) = &self.config.nn_save_config {
                let path = config.path.with_extension("ot");

                match self.vs.save(&path) {
                    Ok(_) => {
                        info!("model has been saved to `{}`", path.display());
                    }
                    Err(err) => {
                        warn!(
                            "failed to save model to `{}` due to: {}",
                            path.display(),
                            err
                        );
                    }
                }

                if let Some(backup_interval) = config.backup_interval {
                    if iter % backup_interval == 0 {
                        let path = path.with_extension("bak.ot");

                        match self.vs.save(&path) {
                            Ok(_) => {
                                info!("model backup has been saved to `{}`", path.display());
                            }
                            Err(err) => {
                                warn!(
                                    "failed to save model backup to `{}` due to: {}",
                                    path.display(),
                                    err
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Plays the neural network against a naive agent to test the performance.
    fn play_against_naive_agent(&self, match_count: usize) -> (u32, u32, u32) {
        let mut player1_won = 0;
        let mut player2_won = 0;
        let mut draw = 0;
        let mut agents_1 = Vec::with_capacity(match_count);
        let mut agents_2 = Vec::with_capacity(match_count);

        let progress_bar = ProgressBar::new(match_count as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.white} [{bar:40.green/white}] {pos:>7}/{len:7}")
                .unwrap()
                .progress_chars("#>-"),
        );
        progress_bar.tick();

        for _ in 0..match_count {
            agents_1.push(NaiveAgent::<G>::new());
            agents_2.push(AlphaZeroAgent::<G>::new(
                self.config.device,
                self.nn_optimizer.nn(),
            ));
        }

        while !agents_1.is_empty() {
            let turn = agents_1[0].game().turn();

            trace!("turn={:?}", turn);

            if turn == Turn::Player2 {
                self.mcts_executor.execute(
                    self.config.device,
                    self.config.mcts_count,
                    self.config.batch_size,
                    self.config.c_puct,
                    self.config.alpha,
                    self.config.epsilon,
                    self.nn_optimizer.nn(),
                    &agents_2,
                );
                progress_bar.tick();
            }

            let mut index = 0;

            while index < agents_1.len() {
                let (agent, opponent_agent) = match turn {
                    Turn::Player1 => (
                        &mut agents_1[index] as &mut dyn Agent<G>,
                        &mut agents_2[index] as &mut dyn Agent<G>,
                    ),
                    Turn::Player2 => (
                        &mut agents_2[index] as &mut dyn Agent<G>,
                        &mut agents_1[index] as &mut dyn Agent<G>,
                    ),
                };

                let action = agent.select_action().unwrap();
                let status = agent.take_action(action).unwrap();

                opponent_agent.ensure_action_exists(action);
                opponent_agent.take_action(action);

                let is_terminal = match status {
                    Status::Ongoing => false,
                    Status::Player1Won => {
                        player1_won += 1;
                        true
                    }
                    Status::Player2Won => {
                        player2_won += 1;
                        true
                    }
                    Status::Draw => {
                        draw += 1;
                        true
                    }
                };

                if is_terminal {
                    agents_1.swap_remove(index);
                    agents_2.swap_remove(index);

                    progress_bar.inc(1);

                    continue;
                }

                index += 1;
            }
        }

        progress_bar.finish();

        (player1_won, player2_won, draw)
    }

    fn play_against_backuped_nn(
        &self,
        path: impl AsRef<Path>,
        match_count: usize,
    ) -> Option<(u32, u32, u32)> {
        let mut vs = VarStore::new(self.config.device);
        let old_version_nn = NN::new(&vs.root(), self.config.nn_config.clone());

        if let Err(err) = vs.load(path.as_ref()) {
            warn!(
                "failed to load neural network backup from `{}` due to: {}",
                path.as_ref().display(),
                err
            );
            return None;
        }

        let mut player1_won = 0;
        let mut player2_won = 0;
        let mut draw = 0;
        let mut agents_1 = Vec::with_capacity(match_count);
        let mut agents_2 = Vec::with_capacity(match_count);

        let progress_bar = ProgressBar::new(match_count as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.white} [{bar:40.green/white}] {pos:>7}/{len:7}")
                .unwrap()
                .progress_chars("#>-"),
        );
        progress_bar.tick();

        for _ in 0..match_count {
            agents_1.push(AlphaZeroAgent::<G>::new(
                self.config.device,
                &old_version_nn,
            ));
            agents_2.push(AlphaZeroAgent::<G>::new(
                self.config.device,
                self.nn_optimizer.nn(),
            ));
        }

        while !agents_1.is_empty() {
            let turn = agents_1[0].mcts().root().game.turn();

            trace!("turn={:?}", turn);

            self.mcts_executor.execute(
                self.config.device,
                self.config.mcts_count,
                self.config.batch_size,
                self.config.c_puct,
                self.config.alpha,
                self.config.epsilon,
                self.nn_optimizer.nn(),
                match turn {
                    Turn::Player1 => &agents_1,
                    Turn::Player2 => &agents_2,
                },
            );
            progress_bar.tick();

            let mut index = 0;

            while index < agents_1.len() {
                let (agent, opponent_agent) = match turn {
                    Turn::Player1 => (&mut agents_1[index], &mut agents_2[index]),
                    Turn::Player2 => (&mut agents_2[index], &mut agents_1[index]),
                };

                let action = agent.select_action().unwrap();
                let status = agent.take_action(action).unwrap();

                opponent_agent.ensure_action_exists(action);
                opponent_agent.take_action(action);

                let is_terminal = match status {
                    Status::Ongoing => false,
                    Status::Player1Won => {
                        player1_won += 1;
                        true
                    }
                    Status::Player2Won => {
                        player2_won += 1;
                        true
                    }
                    Status::Draw => {
                        draw += 1;
                        true
                    }
                };

                if is_terminal {
                    agents_1.swap_remove(index);
                    agents_2.swap_remove(index);

                    progress_bar.inc(1);

                    continue;
                }

                index += 1;
            }
        }

        progress_bar.finish();

        Some((player1_won, player2_won, draw))
    }

    /// Plays the neural network against itself and collects the trajectories.
    fn self_play(&self, episodes: usize) -> Vec<Trajectory<G>> {
        let mut turn_count = 0;
        let mut agents_1 = Vec::with_capacity(episodes);
        let mut agents_2 = Vec::with_capacity(episodes);
        let mut trajectories_set = Vec::with_capacity(episodes);
        let mut trajectories_indices = Vec::from_iter(0..episodes);

        let progress_bar = ProgressBar::new(episodes as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.white} [{bar:40.green/white}] {pos:>7}/{len:7}")
                .unwrap()
                .progress_chars("#>-"),
        );
        progress_bar.tick();

        for _ in 0..episodes {
            agents_1.push(AlphaZeroAgent::<G>::new(
                self.config.device,
                self.nn_optimizer.nn(),
            ));
            agents_2.push(AlphaZeroAgent::<G>::new(
                self.config.device,
                self.nn_optimizer.nn(),
            ));
            trajectories_set.push(Vec::with_capacity(64));
        }

        while !agents_1.is_empty() {
            let turn = agents_1[0].mcts().root().game.turn();

            self.mcts_executor.execute(
                self.config.device,
                self.config.mcts_count,
                self.config.batch_size,
                self.config.c_puct,
                self.config.alpha,
                self.config.epsilon,
                self.nn_optimizer.nn(),
                match turn {
                    Turn::Player1 => &agents_1,
                    Turn::Player2 => &agents_2,
                },
            );

            progress_bar.tick();

            let mut index = 0;
            let (agents, opponent_agents) = match turn {
                Turn::Player1 => (&mut agents_1, &mut agents_2),
                Turn::Player2 => (&mut agents_2, &mut agents_1),
            };

            while index < agents.len() {
                let agent = &mut agents[index];
                let opponent_agent = &mut opponent_agents[index];

                #[cfg(debug_assertions)]
                {
                    let root_n_s_a = agent
                        .mcts()
                        .root()
                        .n_s_a
                        .load(std::sync::atomic::Ordering::Relaxed);
                    let root_w_s_a = agent
                        .mcts()
                        .root()
                        .w_s_a
                        .load(std::sync::atomic::Ordering::Relaxed);

                    trace!("root_n_s_a={}, root_w_s_a={}", root_n_s_a, root_w_s_a);

                    for child in agent.mcts().root().children.read().iter() {
                        let p_s_a = child.p_s_a.load(std::sync::atomic::Ordering::Relaxed);
                        let n_s_a = child.n_s_a.load(std::sync::atomic::Ordering::Relaxed);
                        let w_s_a = child.w_s_a.load(std::sync::atomic::Ordering::Relaxed);

                        trace!(
                            "action={}, p_s_a={}, n_s_a={}, w_s_a={}",
                            child.action.unwrap(),
                            p_s_a,
                            n_s_a,
                            w_s_a
                        );
                    }
                }

                let (action, policy) = agent
                    .sample_action_from_policy(if turn_count < self.config.temperature_threshold {
                        ActionSamplingMode::Boltzmann(self.config.temperature)
                    } else {
                        ActionSamplingMode::Best
                    })
                    .unwrap();

                turn_count += 1;

                let game_before_action = agent.mcts().root().game.clone();
                let turn = game_before_action.turn();
                let z = match agent.take_action(action).unwrap() {
                    Status::Ongoing => None,
                    status @ (Status::Player1Won | Status::Player2Won) => {
                        Some(if status.is_turn(turn) { 1f32 } else { -1f32 })
                    }
                    Status::Draw => Some(0f32),
                };

                opponent_agent.ensure_action_exists(action);
                opponent_agent.take_action(action);

                let trajectories = &mut trajectories_set[trajectories_indices[index]];
                trajectories.push(Trajectory {
                    game: game_before_action,
                    policy,
                    z: z.unwrap_or(0f32),
                });

                if z.is_some() {
                    trace!("trajectory={:?}", trajectories.last().unwrap());

                    agents.swap_remove(index);
                    opponent_agents.swap_remove(index);
                    trajectories_indices.swap_remove(index);

                    progress_bar.inc(1);

                    continue;
                }

                index += 1;
            }
        }

        for trajectories in &mut trajectories_set {
            let mut z = trajectories.last().unwrap().z;

            for trajectory in trajectories.iter_mut().rev().skip(1) {
                z = -z;
                trajectory.z = z;
            }

            // TODO: augment the trajectories
        }

        progress_bar.finish();

        trajectories_set.into_iter().flatten().collect()
    }

    /// Trains the neural network using the trajectories.
    fn train_nn(&mut self, trajectories: &[&Trajectory<G>]) -> (f32, f32, f32) {
        let input = encode_nn_input(
            self.config.device,
            trajectories.len(),
            EncodingPerspective::Player,
            trajectories.iter().map(|t| &t.game),
        );
        let (z_target, policy_target) = encode_nn_targets::<G>(
            self.config.device,
            trajectories.len(),
            trajectories.iter().map(|t| t.z),
            trajectories.iter().map(|t| t.policy.as_slice()),
        );

        self.nn_optimizer.step(&input, &z_target, &policy_target)
    }
}
