from __future__ import annotations

from functools import partial
from typing import Any, Dict, Optional, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from jaxtyping import PRNGKeyArray, PyTree
from tensorflow_probability.substrates.jax import distributions as tfd

from bmpc_jax.common.loss import soft_crossentropy
from bmpc_jax.common.scale import percentile_normalization
from bmpc_jax.common.util import sg
from bmpc_jax.world_model import WorldModel


class BMPC(struct.PyTreeNode):
  model: WorldModel
  kl_scale: jax.Array

  # Planning
  horizon: int = struct.field(pytree_node=False)
  mppi_iterations: int = struct.field(pytree_node=False)
  population_size: int = struct.field(pytree_node=False)
  policy_prior_samples: int = struct.field(pytree_node=False)
  num_elites: int = struct.field(pytree_node=False)
  normalize_elite_values: bool = struct.field(pytree_node=False)
  normalized_elite_scale: float
  min_plan_std: float
  max_plan_std: float
  min_policy_std: float
  max_policy_std: float
  temperature: float
  # Optimization
  batch_size: int = struct.field(pytree_node=False)
  discount: float
  rho: float
  consistency_loss_scale: float
  reward_loss_scale: float
  value_loss_scale: float
  continue_loss_scale: float
  tau: float

  @classmethod
  def create(cls,
             world_model: WorldModel,
             # Planning
             horizon: int,
             mppi_iterations: int,
             population_size: int,
             policy_prior_samples: int,
             num_elites: int,
             normalize_elite_values: bool,
             normalized_elite_scale: float,
             min_plan_std: float,
             max_plan_std: float,
             min_policy_std: float,
             max_policy_std: float,
             temperature: float,
             # Optimization
             discount: float,
             batch_size: int,
             rho: float,
             consistency_loss_scale: float,
             reward_loss_scale: float,
             value_loss_scale: float,
             continue_loss_scale: float,
             tau: float
             ) -> BMPC:

    return cls(model=world_model,
               horizon=horizon,
               mppi_iterations=mppi_iterations,
               population_size=population_size,
               policy_prior_samples=policy_prior_samples,
               num_elites=num_elites,
               normalize_elite_values=normalize_elite_values,
               normalized_elite_scale=normalized_elite_scale,
               min_plan_std=min_plan_std,
               max_plan_std=max_plan_std,
               min_policy_std=min_policy_std,
               max_policy_std=max_policy_std,
               temperature=temperature,
               discount=discount,
               batch_size=batch_size,
               rho=rho,
               consistency_loss_scale=consistency_loss_scale,
               reward_loss_scale=reward_loss_scale,
               value_loss_scale=value_loss_scale,
               continue_loss_scale=continue_loss_scale,
               tau=tau,
               kl_scale=jnp.array([1.0]),
               )

  @partial(jax.jit, static_argnames=('mpc', 'deterministic', 'train'))
  def act(self,
          obs: PyTree,
          prev_plan: Optional[Tuple[jax.Array, jax.Array]] = None,
          mpc: bool = True,
          deterministic: bool = False,
          train: bool = False,
          *,
          key: PRNGKeyArray
          ) -> Tuple[np.ndarray, Optional[Tuple[jax.Array]]]:
    encoder_key, action_key = jax.random.split(key, 2)
    z = self.model.encode(
        obs=obs,
        params=self.model.encoder.params,
        key=encoder_key
    )

    if mpc:
      action, plan = self.plan(
          z=z,
          horizon=self.horizon,
          prev_plan=prev_plan,
          deterministic=deterministic,
          train=train,
          key=action_key
      )
    else:
      action = self.model.sample_actions(
          z=z,
          params=self.model.policy_model.params,
          deterministic=deterministic,
          key=action_key
      )[0]
      plan = None

    return action, plan

  @partial(jax.jit, static_argnames=('horizon', 'deterministic', 'train'))
  def plan(self,
           z: jax.Array,
           horizon: int,
           prev_plan: Tuple[jax.Array, jax.Array] = None,
           deterministic: bool = False,
           train: bool = False,
           *,
           key: PRNGKeyArray,
           ) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
    batch_shape = z.shape[:-1]
    actions = jnp.zeros(
        (
            *batch_shape,
            self.population_size,
            horizon,
            self.model.action_dim
        )
    )

    ###########################################################
    # Policy prior samples
    ###########################################################
    if self.policy_prior_samples > 0:
      key, *prior_noise_keys = jax.random.split(key, 1+horizon)
      policy_actions = jnp.zeros(
          (
              *batch_shape,
              self.policy_prior_samples,
              horizon,
              self.model.action_dim
          )
      )
      z_t = z[..., None, :].repeat(self.policy_prior_samples, axis=-2)
      for t in range(horizon):
        policy_actions = policy_actions.at[..., t, :].set(
            self.model.sample_actions(
                z=z_t,
                params=self.model.policy_model.params,
                deterministic=False,
                min_std=self.min_policy_std,
                max_std=self.max_policy_std,
                key=prior_noise_keys[t]
            )[0]
        )
        if t < horizon-1:  # Don't need for the last time step
          z_t = self.model.next(
              z=z_t,
              a=policy_actions[..., t, :],
              params=self.model.dynamics_model.params
          )

      actions = actions.at[..., :self.policy_prior_samples, :, :].set(
          policy_actions
      )

    ###########################################################
    # MPPI planning
    ###########################################################
    z_t = z[..., None, :].repeat(self.population_size, axis=-2)
    key, mppi_noise_key, *value_keys = jax.random.split(
        key, 2+self.mppi_iterations
    )
    noise = jax.random.normal(
        mppi_noise_key,
        shape=(
            *batch_shape,
            self.population_size - self.policy_prior_samples,
            self.mppi_iterations,
            horizon,
            self.model.action_dim
        )
    )
    # Initialize population state
    mean = jnp.zeros((*batch_shape, horizon, self.model.action_dim))
    std = jnp.full(
        (*batch_shape, horizon, self.model.action_dim), self.max_plan_std
    )
    if prev_plan is not None:
      mean = mean.at[..., :-1, :].set(prev_plan[0][..., 1:, :])

    for i in range(self.mppi_iterations):
      actions = actions.at[..., self.policy_prior_samples:, :, :].set(
          mean[..., None, :, :] + std[..., None, :, :] * noise[..., i, :, :]
      ).clip(-1, 1)

      # Compute elites
      values = self.estimate_value(
          z=z_t, actions=actions, horizon=horizon, key=value_keys[i]
      )
      elite_values, elite_inds = jax.lax.top_k(values, self.num_elites)
      elite_actions = jnp.take_along_axis(
          actions, elite_inds[..., None, None], axis=-3
      )

      if self.normalize_elite_values:
        # Normalize elites to make softmax invariant to value scale as in [1]
        # [1] Williams2016 - Aggressive driving with model predictive integral control
        mean_value = (
            values - values.min(axis=-1, keepdims=True).clip(None, 0)
        ).mean(axis=-1, keepdims=True).clip(1, None)
        elite_values = elite_values / mean_value * self.normalized_elite_scale

      # Update population distribution
      score = jax.nn.softmax(self.temperature * elite_values)
      mean = jnp.sum(score[..., None, None] * elite_actions, axis=-3)
      std = jnp.sqrt(
          jnp.sum(
              score[..., None, None] *
              (elite_actions - mean[..., None, :, :])**2,
              axis=-3
          ) + 1e-6
      ).clip(self.min_plan_std, self.max_plan_std)

    # Sample final action
    if deterministic:  # Use best trajectory
      action_ind = jnp.argmax(elite_values, axis=-1)
    else:  # Sample from elites
      key, final_action_key = jax.random.split(key)
      action_ind = jax.random.categorical(
          final_action_key, logits=jnp.log(score), shape=batch_shape
      )
    action = jnp.take_along_axis(
        elite_actions, action_ind[..., None, None, None], axis=-3
    ).squeeze(-3)
    if train:
      key, final_noise_key = jax.random.split(key)
      final_action = action[..., 0, :] + std[..., 0, :] * \
          jax.random.normal(
              final_noise_key, shape=batch_shape + (self.model.action_dim,)
      )
    else:
      final_action = action[..., 0, :]

    return final_action.clip(-1, 1), (mean, std)

  @partial(jax.jit, static_argnames=('horizon'))
  def estimate_value(self,
                     z: jax.Array,
                     actions: jax.Array,
                     horizon: int,
                     key: PRNGKeyArray
                     ) -> jax.Array:
    G, discount = 0.0, 1.0
    for t in range(horizon):
      reward, _ = self.model.reward(
          z, actions[..., t, :], self.model.reward_model.params
      )
      z = self.model.next(
          z, actions[..., t, :], self.model.dynamics_model.params
      )
      G += discount * reward
      discount *= self.discount

      if self.model.predict_continues:
        continues = jax.nn.sigmoid(
            self.model.continue_model.apply_fn(
                {'params': self.model.continue_model.params}, z
            )
        ).squeeze(-1) > 0.5
        discount *= continues

    Vs, _ = self.model.V(z, self.model.value_model.params, key=key)
    V = Vs.mean(axis=0)
    return G + discount * V

  @jax.jit
  def update_world_model(self,
                         observations: PyTree,
                         actions: jax.Array,
                         rewards: jax.Array,
                         next_observations: PyTree,
                         terminated: jax.Array,
                         truncated: jax.Array,
                         *,
                         key: PRNGKeyArray
                         ) -> Tuple[BMPC, Dict[str, Any]]:

    def world_model_loss_fn(encoder_params: flax.core.FrozenDict,
                            dynamics_params: flax.core.FrozenDict,
                            value_params: flax.core.FrozenDict,
                            reward_params: flax.core.FrozenDict,
                            continue_params: flax.core.FrozenDict,
                            ) -> Tuple[jax.Array, Dict[str, Any]]:
      encoder_key, value_key = jax.random.split(key, 2)
      lam = self.rho**jnp.arange(self.horizon)
      lam /= jnp.sum(lam)

      ###########################################################
      # Encoder forward pass
      ###########################################################
      all_obs = jax.tree.map(
          lambda x, y: jnp.stack([x, y], axis=0),
          observations, next_observations
      )
      all_zs = self.model.encode(
          obs=all_obs, params=encoder_params, key=encoder_key
      )
      encoder_zs = jax.tree.map(lambda x: x[0], all_zs)
      next_zs = jax.tree.map(lambda x: x[1], all_zs)

      ###########################################################
      # Latent rollout (dynamics + consistency loss)
      ###########################################################
      done = jnp.logical_or(terminated, truncated)
      finished = jnp.zeros((self.horizon, self.batch_size), dtype=bool)
      latent_zs = jnp.zeros(
          (self.horizon, self.batch_size, self.model.latent_dim)
      )
      latent_zs = latent_zs.at[0].set(encoder_zs[0])
      consistency_loss = 0
      for t in range(self.horizon):
        z = self.model.next(latent_zs[t], actions[t], dynamics_params)
        consistency_loss += lam[t] * \
            jnp.mean((z - sg(next_zs[t]))**2, where=~finished[t][:, None])
        if t < self.horizon-1:
          latent_zs = latent_zs.at[t+1].set(z)
          finished = finished.at[t+1].set(jnp.logical_or(finished[t], done[t]))

      ###########################################################
      # Reward loss
      ###########################################################
      reward_pred, reward_logits = self.model.reward(
          latent_zs, actions, reward_params
      )
      reward_loss = jnp.sum(
          lam[:, None] * soft_crossentropy(
              pred_logits=reward_logits,
              target=rewards,
              low=self.model.symlog_min,
              high=self.model.symlog_max,
              num_bins=self.model.num_bins,
          ), axis=0, where=~finished
      ).mean()

      reward_mae = jnp.sum(
          lam[:, None] * abs(rewards - reward_pred), axis=0, where=~finished
      ).mean()

      ###########################################################
      # Value loss
      ###########################################################
      value_key, value_target_key = jax.random.split(value_key, 2)

      # TD targets
      value_pred, value_logits = self.model.V(
          latent_zs, value_params, key=value_key
      )
      td_targets = self.td_target(z=encoder_zs, key=value_target_key)
      value_loss = jnp.sum(
          lam[:, None] * soft_crossentropy(
              pred_logits=value_logits,
              target=sg(td_targets),
              low=self.model.symlog_min,
              high=self.model.symlog_max,
              num_bins=self.model.num_bins,
          ), axis=1, where=~finished
      ).mean()

      value_mae = jnp.sum(
          lam[:, None] * abs(td_targets - value_pred), axis=1, where=~finished
      ).mean()
      value_scale = jnp.diff(
          jnp.percentile(value_pred[:, 0], jnp.array([5, 95]), axis=-1),
          axis=0
      ).mean()

      ###########################################################
      # Continue loss
      ###########################################################
      if self.model.predict_continues:
        continue_logits = self.model.continue_model.apply_fn(
            {'params': continue_params}, latent_zs
        ).squeeze(-1)
        continue_loss = optax.sigmoid_binary_cross_entropy(
            continue_logits, 1 - terminated
        ).mean()
      else:
        continue_loss = 0.0

      total_loss = (
          self.consistency_loss_scale * consistency_loss +
          self.reward_loss_scale * reward_loss +
          self.value_loss_scale * value_loss +
          self.continue_loss_scale * continue_loss
      )

      return total_loss, {
          'consistency_loss': consistency_loss,
          'reward_loss': reward_loss,
          'value_loss': value_loss,
          'continue_loss': continue_loss,
          'total_loss': total_loss,
          'encoder_zs': encoder_zs,
          'latent_zs': latent_zs,
          'finished': finished,
          # Additional logging metrics
          'reward_mae': reward_mae,
          'value_mae': value_mae,
          'value_scale': value_scale,
      }

    # Update world model
    (encoder_grads, dynamics_grads, value_grads, reward_grads, continue_grads), info = jax.grad(
        world_model_loss_fn, argnums=(0, 1, 2, 3, 4), has_aux=True)(
            self.model.encoder.params,
            self.model.dynamics_model.params,
            self.model.value_model.params,
            self.model.reward_model.params,
            self.model.continue_model.params if self.model.predict_continues else None
    )

    new_encoder = self.model.encoder.apply_gradients(grads=encoder_grads)
    new_dynamics_model = self.model.dynamics_model.apply_gradients(
        grads=dynamics_grads
    )
    new_reward_model = self.model.reward_model.apply_gradients(
        grads=reward_grads
    )
    new_value_model = self.model.value_model.apply_gradients(
        grads=value_grads)
    new_target_value_model = self.model.target_value_model.replace(
        params=optax.incremental_update(
            new_value_model.params,
            self.model.target_value_model.params,
            self.tau
        )
    )
    if self.model.predict_continues:
      new_continue_model = self.model.continue_model.apply_gradients(
          grads=continue_grads
      )
    else:
      new_continue_model = self.model.continue_model

    # Update model
    new_agent = self.replace(
        model=self.model.replace(
            encoder=new_encoder,
            dynamics_model=new_dynamics_model,
            reward_model=new_reward_model,
            value_model=new_value_model,
            target_value_model=new_target_value_model,
            continue_model=new_continue_model
        ),
    )
    return new_agent, info

  def td_target(self,
                z: jax.Array,
                num_td_steps: int = 1,
                *,
                key: PRNGKeyArray
                ) -> jax.Array:
    key, *action_keys = jax.random.split(key, 1+num_td_steps)
    G, discount = 0, 1
    for t in range(num_td_steps):
      action = self.model.sample_actions(
          z=z,
          params=self.model.policy_model.params,
          deterministic=False,
          key=action_keys[t]
      )[0]
      reward, _ = self.model.reward(z, action, self.model.reward_model.params)
      z = self.model.next(z, action, self.model.dynamics_model.params)
      G += discount * reward
      discount *= self.discount

      if self.model.predict_continues:
        continues = jax.nn.sigmoid(
            self.model.continue_model.apply_fn(
                {'params': self.model.continue_model.params}, z
            )
        ).squeeze(-1) > 0.5
        discount *= continues

    Vs, _ = self.model.V(
        z, self.model.target_value_model.params, key=key
    )

    V = Vs.mean(axis=0)
    td_target = G + discount * V
    return td_target

  @jax.jit
  def update_policy(self,
                    zs: jax.Array,
                    expert_mean: jax.Array,
                    expert_std: jax.Array,
                    finished: jax.Array,
                    key: PRNGKeyArray
                    ):
    def policy_loss_fn(actor_params: flax.core.FrozenDict):
      lam = self.rho**jnp.arange(self.horizon)
      lam /= jnp.sum(lam)
      _, mean, log_std, log_probs = self.model.sample_actions(
          z=zs,
          params=actor_params,
          deterministic=False,
          key=key
      )

      # Compute KL divergence between policy and expert
      action_dist = tfd.MultivariateNormalDiag(mean, jnp.exp(log_std))
      expert_dist = tfd.MultivariateNormalDiag(expert_mean, expert_std)
      kl_div = tfd.kl_divergence(action_dist, expert_dist)
      kl_scale = percentile_normalization(
          kl_div[0], self.kl_scale
      ).clip(1, None)

      policy_loss = jnp.sum(
          lam[:, None] * kl_div / kl_scale, axis=0, where=~finished
      ).mean()

      return policy_loss, {
          'policy_loss': policy_loss,
          'kl_scale': kl_scale,
          'policy_entropy': -log_probs.mean(),
          'policy_log_std': log_std.mean(),
      }

    policy_grads, policy_info = jax.grad(policy_loss_fn, has_aux=True)(
        self.model.policy_model.params
    )
    new_policy = self.model.policy_model.apply_gradients(grads=policy_grads)
    new_agent = self.replace(
        model=self.model.replace(policy_model=new_policy),
        kl_scale=policy_info['kl_scale'],
    )
    return new_agent, policy_info
