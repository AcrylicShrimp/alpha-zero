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

    let mut trainer = Trainer::<game::games::Gomoku9>::new(TrainerConfig {
        device,
        mcts_executor_config: MCTSExecutorConfig { num_threads: None },
        nn_config: NNConfig {
            device,
            kind: Kind::Half,
            residual_blocks: 3,
            residual_block_channels: 128,
            fc0_channels: 256,
            fc1_channels: 256,
        },
        nn_optimizer_config: NNOptimizerConfig {
            lr: 0.001f64,
            weight_decay: 0.0001f64,
            gradient_clip_norm: 5f64,
            gradient_scale_update_interval: 2000,
        },
        replay_buffer_size: 50000,
        episodes: 100,
        mcts_count: 100,
        batch_size: 4,
        c_puct: 1f32,
        alpha: 0.03f32,
        epsilon: 0.25f32,
        temperature: 1f32,
        temperature_threshold: 30,
        parameter_update_count: 400,
        parameter_update_batch_size: 64,
        match_count: 100,
        nn_save_config: Some(TrainerNNSaveConfig {
            path: "saves/gomoku-adam-fp16".into(),
            backup_interval: Some(100),
        }),
    })
    .unwrap();

    trainer.train(100000);
}
