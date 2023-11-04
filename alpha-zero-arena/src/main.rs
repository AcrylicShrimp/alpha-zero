mod arena;

use arena::Arena;
use game::games::Gomoku9;
use nn::{NNConfig, NN};
use tch::{nn::VarStore, Device};

fn main() {
    let device = if tch::Cuda::is_available() {
        println!("CUDA is available, using CUDA");
        Device::cuda_if_available()
    } else if tch::utils::has_mps() {
        println!("MPS is available, using MPS");
        Device::Mps
    } else {
        println!("no accelerator available, using CPU");
        Device::Cpu
    };

    let mut vs = VarStore::new(device);
    let nn = NN::new(
        &vs.root(),
        NNConfig {
            residual_blocks: 7,
            residual_block_channels: 256,
            residual_block_mid_channels: None,
            fc0_channels: 512,
            fc1_channels: 512,
        },
    );
    let arena = Arena::<Gomoku9>::new(device, nn);

    vs.load("saves/alpha-zero-gomoku9-adam.ot").unwrap();

    loop {
        println!("=============================================");
        println!("==== A new game has started, good luck! =====");
        println!("=============================================");
        arena.play();
    }
}
