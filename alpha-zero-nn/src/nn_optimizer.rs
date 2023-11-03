use crate::NN;
use game::Game;
use tch::{
    nn::{Adam, Optimizer, OptimizerConfig, VarStore},
    Reduction, TchError, Tensor,
};

#[derive(Debug, Clone, PartialEq)]
/// Configuration for the neural network optimizer.
pub struct NNOptimizerConfig {
    /// The learning rate.
    pub lr: f64,
}

/// A neural network optimizer.
pub struct NNOptimizer<G>
where
    G: Game,
{
    optimizer: Optimizer,
    nn: NN<G>,
}

impl<G> NNOptimizer<G>
where
    G: Game,
{
    /// Creates a new optimizer for the given neural network, using Adam as the optimizer.
    pub fn new_adam(vs: &VarStore, config: NNOptimizerConfig, nn: NN<G>) -> Result<Self, TchError> {
        let optimizer = Adam::default().build(vs, config.lr)?;

        Ok(Self { optimizer, nn })
    }

    pub fn nn(&self) -> &NN<G> {
        &self.nn
    }

    /// Performs a single training step.
    /// Returns the total loss, the value loss and the policy loss.
    pub fn step(
        &mut self,
        input: &Tensor,
        z_target: &Tensor,
        policy_target: &Tensor,
    ) -> (f32, f32, f32) {
        let z_target = z_target.view([-1, 1]);
        let policy_target = policy_target.view([-1, G::POSSIBLE_ACTION_COUNT as i64]);

        let (v, pi) = self.nn.forward(input, true);

        #[cfg(debug_assertions)]
        {
            println!("pi={:?}", Vec::<f32>::try_from(pi.view([-1])).unwrap());
        }

        let v_loss = v.mse_loss(&z_target, Reduction::Mean);
        let pi_loss =
            pi.cross_entropy_loss::<&Tensor>(&policy_target, None, Reduction::Mean, -100, 0.0);
        let loss = &v_loss + &pi_loss;

        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.step();

        (
            f32::try_from(loss).unwrap(),
            f32::try_from(v_loss).unwrap(),
            f32::try_from(pi_loss).unwrap(),
        )
    }
}
