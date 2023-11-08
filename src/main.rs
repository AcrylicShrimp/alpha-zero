use log::info;
use nn::{nn_optimizer::NNOptimizerConfig, NNConfig};
use tch::{Device, Kind};
use train::{
    parallel_mcts_executor::MCTSExecutorConfig, Trainer, TrainerConfig, TrainerNNSaveConfig,
};

fn main() {
    env_logger::init();

    let device = if tch::Cuda::is_available() {
        info!("CUDA is available, using CUDA");
        Device::cuda_if_available()
    } else if tch::utils::has_mps() {
        info!("MPS is available, using MPS");
        Device::Mps
    } else {
        info!("no accelerator available, using CPU");
        Device::Cpu
    };

    let mut trainer = Trainer::<game::games::TicTacToe>::new(TrainerConfig {
        device,
        mcts_executor_config: MCTSExecutorConfig { num_threads: None },
        nn_config: NNConfig {
            device,
            kind: Kind::Float,
            residual_blocks: 3,
            residual_block_channels: 64,
            fc0_channels: 128,
            fc1_channels: 128,
        },
        nn_optimizer_config: NNOptimizerConfig {
            lr: 0.001f64,
            gradient_clip_norm: 1f64,
            gradient_scale_update_interval: 50,
        },
        replay_buffer_size: 500000,
        episodes: 10,
        mcts_count: 25,
        batch_size: 4,
        c_puct: 1f32,
        alpha: 0.03f32,
        epsilon: 0.25f32,
        temperature: 1f32,
        temperature_threshold: 30,
        parameter_update_count: 1600,
        parameter_update_batch_size: 128,
        match_count: 10,
        nn_save_config: Some(TrainerNNSaveConfig {
            path: "saves/tictactoe-adam-fp32".into(),
            backup_interval: Some(100),
        }),
    })
    .unwrap();

    trainer.train(100000);
}
