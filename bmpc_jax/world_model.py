import copy
from functools import partial
from typing import *
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState
from jaxtyping import PRNGKeyArray, PyTree
from tensorflow_probability.substrates.jax import distributions as tfd

from bmpc_jax.common.activations import mish, simnorm
from bmpc_jax.common.util import symlog, two_hot_inv
from bmpc_jax.networks import Ensemble, NormedLinear

MIN_LOG_STD = -5
MAX_LOG_STD = 1


class WorldModel(struct.PyTreeNode):
  # Models
  encoder: TrainState
  dynamics_model: TrainState
  reward_model: TrainState
  policy_model: TrainState
  value_model: TrainState
  target_value_model: TrainState
  continue_model: TrainState
  # Spaces
  action_dim: int = struct.field(pytree_node=False)
  # Architecture
  latent_dim: int = struct.field(pytree_node=False)
  simnorm_dim: int = struct.field(pytree_node=False)
  num_value_nets: int = struct.field(pytree_node=False)
  num_bins: int = struct.field(pytree_node=False)
  symlog_min: float
  symlog_max: float
  predict_continues: bool = struct.field(pytree_node=False)
  symlog_obs: bool = struct.field(pytree_node=False)

  @classmethod
  def create(cls,
             # Spaces
             action_dim: int,
             # Encoder module
             encoder: TrainState,
             # World model
             latent_dim: int,
             value_dropout: float,
             num_value_nets: int,
             num_bins: int,
             symlog_min: float,
             symlog_max: float,
             simnorm_dim: int,
             predict_continues: bool,
             symlog_obs: bool,
             # Optimization
             learning_rate: float,
             max_grad_norm: float = 20,
             # Misc
             tabulate: bool = False,
             dtype: jnp.dtype = jnp.float32,
             *,
             key: PRNGKeyArray,
             ):
    (
        dynamics_key, reward_key, value_key, policy_key, continue_key
    ) = jax.random.split(key, 5)

    # Latent forward dynamics model
    dynamics_module = nn.Sequential([
        NormedLinear(latent_dim, activation=mish, dtype=dtype),
        NormedLinear(latent_dim, activation=mish, dtype=dtype),
        NormedLinear(latent_dim, activation=None, dtype=dtype),
    ])
    dynamics_model = TrainState.create(
        apply_fn=dynamics_module.apply,
        params=dynamics_module.init(
            dynamics_key, jnp.zeros(latent_dim + action_dim))['params'],
        tx=optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(max_grad_norm),
            optax.adamw(learning_rate),
        )
    )

    # Transition reward model
    reward_module = nn.Sequential([
        NormedLinear(latent_dim, activation=mish, dtype=dtype),
        NormedLinear(latent_dim, activation=mish, dtype=dtype),
        nn.Dense(
            num_bins, kernel_init=nn.initializers.zeros, dtype=dtype
        )
    ])
    reward_model = TrainState.create(
        apply_fn=reward_module.apply,
        params=reward_module.init(
            reward_key, jnp.zeros(latent_dim + action_dim))['params'],
        tx=optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(max_grad_norm),
            optax.adamw(learning_rate),
        )
    )

    # Policy model
    policy_module = nn.Sequential([
        NormedLinear(latent_dim, activation=mish, dtype=dtype),
        NormedLinear(latent_dim, activation=mish, dtype=dtype),
        nn.Dense(
            2*action_dim,
            kernel_init=nn.initializers.truncated_normal(0.02),
            dtype=dtype
        )
    ])
    policy_model = TrainState.create(
        apply_fn=policy_module.apply,
        params=policy_module.init(policy_key, jnp.zeros(latent_dim))['params'],
        tx=optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(max_grad_norm),
            optax.adamw(learning_rate),
        )
    )

    # Value model
    value_param_key, value_dropout_key = jax.random.split(value_key)
    value_base = partial(nn.Sequential, [
        NormedLinear(
            latent_dim,
            activation=mish,
            dropout_rate=value_dropout,
            dtype=dtype
        ),
        NormedLinear(
            latent_dim,
            activation=mish,
            dropout_rate=value_dropout,
            dtype=dtype
        ),
        nn.Dense(
            num_bins, kernel_init=nn.initializers.zeros, dtype=dtype
        )
    ])
    value_ensemble = Ensemble(value_base, num=num_value_nets)
    value_model = TrainState.create(
        apply_fn=value_ensemble.apply,
        params=value_ensemble.init(
            {'params': value_param_key, 'dropout': value_dropout_key},
            jnp.zeros(latent_dim))['params'],
        tx=optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(max_grad_norm),
            optax.adamw(learning_rate),
        )
    )
    target_value_model = TrainState.create(
        apply_fn=value_ensemble.apply,
        params=copy.deepcopy(value_model.params),
        tx=optax.GradientTransformation(lambda _: None, lambda _: None))

    if predict_continues:
      continue_module = nn.Sequential([
          NormedLinear(latent_dim, activation=mish, dtype=dtype),
          NormedLinear(latent_dim, activation=mish, dtype=dtype),
          nn.Dense(1, kernel_init=nn.initializers.zeros, dtype=dtype)
      ])
      continue_model = TrainState.create(
          apply_fn=continue_module.apply,
          params=continue_module.init(
              continue_key, jnp.zeros(latent_dim))['params'],
          tx=optax.chain(
              optax.zero_nans(),
              optax.clip_by_global_norm(max_grad_norm),
              optax.adamw(learning_rate),
          )
      )
    else:
      continue_model = None

    if tabulate:
      print("Dynamics Model")
      print("--------------")
      print(
          dynamics_module.tabulate(
              jax.random.key(0),
              jnp.ones(latent_dim + action_dim),
              compute_flops=True
          )
      )

      print("Reward Model")
      print("------------")
      print(
          reward_module.tabulate(
              jax.random.key(0),
              jnp.ones(latent_dim + action_dim),
              compute_flops=True
          )
      )

      print("Policy Model")
      print("------------")
      print(
          policy_module.tabulate(
              jax.random.key(0), jnp.ones(latent_dim), compute_flops=True
          )
      )

      print("Value Model")
      print("-----------")
      print(
          value_ensemble.tabulate(
              {'params': jax.random.key(0), 'dropout': jax.random.key(0)},
              jnp.ones(latent_dim + action_dim),
              compute_flops=True
          )
      )

      if predict_continues:
        print("Continue Model")
        print("--------------")
        print(
            continue_module.tabulate(
                jax.random.key(0), jnp.ones(latent_dim), compute_flops=True
            )
        )

    return cls(
        # Spaces
        action_dim=action_dim,
        # Models
        encoder=encoder,
        dynamics_model=dynamics_model,
        reward_model=reward_model,
        policy_model=policy_model,
        value_model=value_model,
        target_value_model=target_value_model,
        continue_model=continue_model,
        # Architecture
        latent_dim=latent_dim,
        simnorm_dim=simnorm_dim,
        num_value_nets=num_value_nets,
        num_bins=num_bins,
        symlog_min=float(symlog_min),
        symlog_max=float(symlog_max),
        predict_continues=predict_continues,
        symlog_obs=symlog_obs
    )

  @jax.jit
  def encode(self, obs: PyTree, params: Dict, key: PRNGKeyArray) -> jax.Array:
    if self.symlog_obs:
      obs = jax.tree.map(lambda x: symlog(x), obs)
    z = self.encoder.apply_fn({'params': params}, obs, rngs={'dropout': key})
    return simnorm(z, simplex_dim=self.simnorm_dim) 

  @jax.jit
  def next(self, z: jax.Array, a: jax.Array, params: Dict) -> jax.Array:
    z = self.dynamics_model.apply_fn(
        {'params': params}, jnp.concatenate([z, a], axis=-1)
    )
    return simnorm(z, simplex_dim=self.simnorm_dim)

  @jax.jit
  def reward(self, z: jax.Array, a: jax.Array, params: Dict
             ) -> Tuple[jax.Array, jax.Array]:
    z = jnp.concatenate([z, a], axis=-1)
    logits = self.reward_model.apply_fn({'params': params}, z)
    reward = two_hot_inv(
        logits, self.symlog_min, self.symlog_max, self.num_bins
    )
    return reward, logits

  @jax.jit
  def sample_actions(self,
                     z: jax.Array,
                     params: Dict,
                     std_scale: float = 1,
                     *,
                     key: PRNGKeyArray
                     ) -> Tuple[jax.Array, ...]:
    # Chunk the policy model output to get mean and logstd
    mean, log_std = jnp.split(
        self.policy_model.apply_fn({'params': params}, z), 2, axis=-1
    )
    mean = jnp.tanh(mean)
    log_std = MIN_LOG_STD + (MAX_LOG_STD - MIN_LOG_STD) * \
        0.5 * (jnp.tanh(log_std) + 1)
    std = std_scale * jnp.exp(log_std)

    # Sample action and compute logprobs
    dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=std)
    action = dist.sample(seed=key)
    log_probs = dist.log_prob(action)

    return action.clip(-1, 1), mean, log_std, log_probs

  @jax.jit
  def V(self, z: jax.Array, params: Dict, key: PRNGKeyArray
        ) -> Tuple[jax.Array, jax.Array]:
    logits = self.value_model.apply_fn(
        {'params': params}, z, rngs={'dropout': key}
    )

    V = two_hot_inv(logits, self.symlog_min, self.symlog_max, self.num_bins)
    return V, logits
