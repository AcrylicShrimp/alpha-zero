mod arena;

use arena::Arena;
use game::games::Gomoku9;
use nn::{NNConfig, NN};
use std::io::Cursor;
use tch::{Device, Kind};

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

    let mut nn = NN::new(NNConfig {
        device,
        kind: Kind::Float,
        residual_blocks: 7,
        residual_block_channels: 256,
        fc0_channels: 512,
        fc1_channels: 512,
    });

    let weights = Cursor::new(std::fs::read("saves/alpha-zero-gomoku9-adam.ot").unwrap());
    nn.load_weights(weights).unwrap();

    let arena = Arena::<Gomoku9>::new(nn);

    loop {
        println!("=============================================");
        println!("==== A new game has started, good luck! =====");
        println!("=============================================");
        arena.play();
    }
}
