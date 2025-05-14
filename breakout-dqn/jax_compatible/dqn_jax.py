import os
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import checkpoints
from typing import Any, Dict
from es import BaseModel


class QNetwork(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32).reshape(-1)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_actions)(x)
        return x


class BreakoutDQN(BaseModel):
    def __init__(self, env, config):
        self.env = env
        self.params_rng = jax.random.PRNGKey(config.get("seed", 0))
        self.q_model = QNetwork(num_actions=env.num_actions)

        self.obs_shape = env.observation_space(env.default_params).shape
        self.params = self.q_model.init(self.params_rng, jnp.zeros(self.obs_shape))
        self.learning_rate = config.get("learning_rate", 1e-3)
        self.optimizer = optax.adam(self.learning_rate)
        self.opt_state = self.optimizer.init(self.params)

        self.gamma = config.get("gamma", 0.99)
        self.num_eval_env_steps = config.get("num_eval_steps", 1000)

    def get_parameters(self) -> np.ndarray:
        flat_params, _ = jax.tree_util.tree_flatten(self.params)
        return np.concatenate([np.ravel(p) for p in flat_params])

    def set_parameters(self, flat_params: np.ndarray) -> None:
        tree_def = jax.tree_util.tree_structure(self.params)
        shapes = [p.shape for p in jax.tree_util.tree_flatten(self.params)[0]]
        sizes = [np.prod(s) for s in shapes]

        chunks = np.split(flat_params, np.cumsum(sizes)[:-1])
        new_leaves = [jnp.array(c).reshape(s) for c, s in zip(chunks, shapes)]
        self.params = jax.tree_util.tree_unflatten(tree_def, new_leaves)

    def evaluate(self, num_episodes: int) -> float:
        total_rewards = []
        key = self.params_rng

        for _ in range(num_episodes):
            obs, state = self.env.reset_env(key, self.env.default_params)
            done = False
            episode_reward = 0

            while not done:
                q_values = self.q_model.apply(self.params, obs)
                action = int(jnp.argmax(q_values))
                key, subkey = jax.random.split(key)
                obs, state, reward, done, _ = self.env.step_env(
                    subkey, state, action, self.env.default_params
                )
                episode_reward += reward

            total_rewards.append(episode_reward)

        return float(np.mean(total_rewards))

    def save_checkpoint(self, path: str, generation: int, metrics: Dict[str, Any]) -> None:
        ckpt = {
            "params": self.params,
            "opt_state": self.opt_state,
            "metrics": metrics,
        }
        checkpoints.save_checkpoint(
            ckpt_dir=os.path.dirname(path), target=ckpt, step=generation, prefix="ckpt_"
        )
        print(f"Saved checkpoint to {path}")
