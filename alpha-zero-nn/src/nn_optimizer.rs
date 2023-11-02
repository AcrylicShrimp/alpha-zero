use crate::NN;
use game::Game;
use tch::{
    nn::{Adam, Optimizer, OptimizerConfig, VarStore},
    Kind, Reduction, TchError, Tensor,
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
        let (v, pi) = self.nn.forward(input, true);
        let v_loss = v.mse_loss(&z_target.view([-1, 1]), Reduction::Mean);
        let pi_loss = (-policy_target.view([-1, G::POSSIBLE_ACTION_COUNT as i64])
            * pi.log_softmax(-1, Kind::Float))
        .mean(Kind::Float);
        let loss = &v_loss + &pi_loss;

        loss.backward();
        self.optimizer.step();

        (
            loss.double_value(&[]) as f32,
            v_loss.double_value(&[]) as f32,
            pi_loss.double_value(&[]) as f32,
        )
    }
}
