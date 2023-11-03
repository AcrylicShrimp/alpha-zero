use log::info;
use nn::{nn_optimizer::NNOptimizerConfig, NNConfig};
use tch::Device;
use train::{parallel_mcts_executor::MCTSExecutorConfig, Trainer, TrainerConfig};

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

    let mut trainer = Trainer::<game::games::Gomoku9>::new(TrainerConfig {
        device,
        mcts_executor_config: MCTSExecutorConfig { num_threads: None },
        nn_config: NNConfig {
            residual_blocks: 3,
            residual_block_channels: 128,
            residual_block_mid_channels: Some(32),
            fc0_channels: 512,
            fc1_channels: 512,
        },
        nn_optimizer_config: NNOptimizerConfig { lr: 0.001f64 },
        replay_buffer_size: 100000,
        episodes: 100,
        mcts_count: 400,
        batch_size: 4,
        c_puct: 1f32,
        alpha: 0.03f32,
        epsilon: 0.25f32,
        temperature: 1f32,
        temperature_threshold: 30,
        parameter_update_count: 400,
        parameter_update_batch_size: 64,
        nn_path: Some("saves/alpha-zero-gomoku9-adam".into()),
    })
    .unwrap();

    trainer.train(1000);
}
