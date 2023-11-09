use crate::NN;
use game::Game;
use tch::{
    nn::{Optimizer, OptimizerConfig},
    TchError,
};

#[derive(Debug, Clone, PartialEq)]
/// Configuration for the neural network optimizer.
pub struct NNOptimizerConfig {
    /// The learning rate.
    pub lr: f64,
    /// The maximum gradient norm.
    pub gradient_clip_norm: f64,
    /// The number of training steps after which the gradient scale is updated.
    pub gradient_scale_update_interval: usize,
}

/// A neural network optimizer which supports mixed precision training.
pub struct NNOptimizer<G>
where
    G: Game,
{
    config: NNOptimizerConfig,
    nn: NN<G>,
    optimizer: Optimizer,
}

impl<G> NNOptimizer<G>
where
    G: Game,
{
    /// Creates a new optimizer for the given neural network.
    pub fn new(
        config: NNOptimizerConfig,
        nn: NN<G>,
        optimizer: impl OptimizerConfig,
    ) -> Result<Self, TchError> {
        let optimizer = optimizer.build(&nn.vs_master(), config.lr)?;

        Ok(Self {
            config,
            nn,
            optimizer,
        })
    }

    pub fn nn(&self) -> &NN<G> {
        &self.nn
    }

    pub fn nn_mut(&mut self) -> &mut NN<G> {
        &mut self.nn
    }

    pub fn optimizer_mut(&mut self) -> &mut Optimizer {
        &mut self.optimizer
    }

    /// Performs a single training step.
    /// Returns the total loss, the value loss and the policy loss.
    pub fn step<'g>(
        &mut self,
        batch_size: usize,
        game_iter: impl Iterator<Item = &'g G>,
        z_iter: impl Iterator<Item = f32>,
        policy_iter: impl Iterator<Item = &'g [f32]>,
    ) -> (f32, f32, f32)
    where
        G: 'g,
    {
        if batch_size == 0 {
            return (0f32, 0f32, 0f32);
        }

        let (v_loss, pi_loss) = self
            .nn
            .loss(true, batch_size, game_iter, z_iter, policy_iter);
        let loss = &v_loss + &pi_loss;

        self.optimizer.zero_grad();

        loss.backward();

        self.optimizer
            .clip_grad_norm(self.config.gradient_clip_norm);
        self.optimizer.step();

        (
            f32::try_from(loss).unwrap(),
            f32::try_from(v_loss).unwrap(),
            f32::try_from(pi_loss).unwrap(),
        )
    }
}

#[cfg(test)]
mod tests {
    use tch::{
        nn::{linear, Module, VarStore},
        Device, Tensor,
    };

    #[test]
    fn copy_gradient() {
        let vs = VarStore::new(Device::Cpu);
        let linear = linear(vs.root(), 4, 4, Default::default());

        assert!(0 < vs.trainable_variables().len());

        let input = Tensor::randn(&[4, 4], tch::kind::FLOAT_CPU);
        let output = linear.forward(&input).mean(None);

        output.backward();

        for param in &mut vs.trainable_variables() {
            let mut grad = param.grad();
            grad.copy_(&Tensor::from_slice(&[-100f32, -100f32, -100f32, -100f32]));
            let _ = grad.divide_(&Tensor::from_slice(&[-10f32, -10f32, -10f32, -10f32]));
        }

        for param in &vs.trainable_variables() {
            let grad = param.grad();

            let min = grad.min().double_value(&[]);
            assert_eq!(min, 10f64);
        }
    }
}
