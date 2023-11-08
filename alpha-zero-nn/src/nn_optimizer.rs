use crate::NN;
use game::Game;
use tch::{
    nn::{Optimizer, OptimizerConfig},
    TchError, Tensor,
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
    gradient_scale: f32,
    step_count: usize,
    master_grad_created: bool,
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
            master_grad_created: false,
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
        game_iter: impl Iterator<Item = &'g G> + Clone,
        z_iter: impl Iterator<Item = f32> + Clone,
        policy_iter: impl Iterator<Item = &'g [f32]> + Clone,
    ) -> (f32, f32, f32)
    where
        G: 'g,
    {
        if batch_size == 0 {
            return (0f32, 0f32, 0f32);
        }

        if !self.master_grad_created {
            self.nn.run_backward_on_master(
                1,
                std::iter::once(game_iter.clone().next().unwrap()),
                std::iter::once(z_iter.clone().next().unwrap()),
                std::iter::once(policy_iter.clone().next().unwrap()),
            );
            self.master_grad_created = true;
        }

        let (v_loss, pi_loss) = self
            .nn
            .loss(true, batch_size, game_iter, z_iter, policy_iter);
        let loss = &v_loss + &pi_loss;
        let gradient_scale = Tensor::from_slice(&[self.gradient_scale]).to(self.nn.config().device);
        let scaled_loss = &loss * &gradient_scale;

        // zero out gradients for fp16 weights
        for (_, param) in &mut self.nn.vs_cloned().variables() {
            param.zero_grad();
        }

        // compute gradients for fp16 weights
        scaled_loss.backward();

        let mut skip_update = false;

        for (_, param) in &self.nn.vs_cloned().variables() {
            let grad = param.grad();

            if !grad.defined() {
                continue;
            }

            if (grad.isinf().any().int64_value(&[]) != 0)
                || (grad.isnan().any().int64_value(&[]) != 0)
            {
                // inf or nan detected, use lower gradient scale and skip weight update
                self.gradient_scale *= 0.5f32;
                self.step_count = 0;
                skip_update = true;

                break;
            }
        }

        if !skip_update {
            // copy unscaled gradients into master
            let params_cloned = self.nn.vs_cloned().variables();

            for (param_name, param) in self.nn.vs_master().variables() {
                let grad = param.grad();

                if !grad.defined() {
                    continue;
                }

                let param_cloned = params_cloned.get(&param_name).unwrap();
                let grad_cloned = param_cloned.grad();
                let mut grad_master = param.grad();

                grad_master.copy_(&grad_cloned);
                let _ = grad_master.divide_(&gradient_scale);
            }

            // now gradients are prepared for fp32 weights, run optimizer
            self.optimizer
                .clip_grad_norm(self.config.gradient_clip_norm);
            self.optimizer.step();
            self.step_count += 1;

            if self.config.gradient_scale_update_interval <= self.step_count {
                // increase gradient scale
                self.gradient_scale *= 2f32;
                self.step_count = 0;
            }

            // update fp16 weights
            self.nn.copy_weights_to_cloned().unwrap();
        }

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
