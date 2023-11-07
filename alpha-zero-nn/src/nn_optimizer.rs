use crate::NN;
use game::Game;
use tch::{
    nn::{Optimizer, OptimizerConfig},
    Kind, TchError, Tensor,
};

#[derive(Debug, Clone, PartialEq)]
/// Configuration for the neural network optimizer.
pub struct NNOptimizerConfig {
    /// The learning rate.
    pub lr: f64,
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
    gradient_scale: f32,
    step_count: usize,
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
            // default value, but can be changed
            gradient_scale: (2 << 15) as f32,
            step_count: 0,
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
        let (v_loss, pi_loss) = self
            .nn
            .loss(true, batch_size, game_iter, z_iter, policy_iter);
        let loss = &v_loss + &pi_loss;
        let gradient_scale = Tensor::from_slice(&[self.gradient_scale]).to(self.nn.config().device);
        let scaled_loss = &loss * &gradient_scale;

        // zero out gradients for fp16 weights
        for param in &mut self.nn.vs_cloned().trainable_variables() {
            param.zero_grad();
        }

        // compute gradients for fp16 weights
        scaled_loss.backward();

        let mut skip_update = false;

        for param in &self.nn.vs_cloned().trainable_variables() {
            let grad = param.grad();

            if !grad.defined() {
                continue;
            }

            if bool::try_from(grad.isinf().max()).unwrap()
                || bool::try_from(grad.isnan().max()).unwrap()
            {
                // inf or nan detected, use lower gradient scale and skip weight update
                self.gradient_scale *= 0.5f32;
                self.step_count = 0;
                skip_update = true;
            }
        }

        if !skip_update {
            for (param_cloned, param_master) in self
                .nn
                .vs_cloned()
                .trainable_variables()
                .iter()
                .zip(self.nn.vs_master().trainable_variables().iter())
            {
                let grad_cloned = param_cloned.grad();
                let mut grad_master = param_master.grad();

                if !grad_cloned.defined() || !grad_master.defined() {
                    continue;
                }

                // copy unscaled gradients into master
                grad_master.copy_(&(grad_cloned.to_kind(Kind::Float) / &gradient_scale));
            }

            self.step_count += 1;

            if self.config.gradient_scale_update_interval <= self.step_count {
                // increase gradient scale
                self.gradient_scale *= 2f32;
                self.step_count = 0;
            }
        }

        // now gradients are prepared for fp32 weights, run optimizer
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
        nn::{linear, VarStore},
        Device,
    };

    #[test]
    fn copy_gradient() {
        let vs = VarStore::new(Device::Cpu);
        linear(vs.root(), 4, 4, Default::default());

        assert!(0 < vs.trainable_variables().len());

        for param in &mut vs.trainable_variables() {
            let grad = param.grad();

            if grad.defined() {
                let _ = param.grad().fill(1f64);
            }
        }

        for param in &vs.trainable_variables() {
            let grad = param.grad();

            if grad.defined() {
                let min = grad.min().double_value(&[]);
                assert_eq!(min, 1f64);
            }
        }
    }
}
