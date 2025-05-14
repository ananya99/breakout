import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax
import random
from typing import Tuple, Deque
from collections import deque, namedtuple
from breakout_jax import MinBreakout
from dqn_jax import QNetwork

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def add(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*batch))
        return {
            "state": jnp.array(batch.state),
            "action": jnp.array(batch.action),
            "reward": jnp.array(batch.reward),
            "next_state": jnp.array(batch.next_state),
            "done": jnp.array(batch.done),
        }

    def __len__(self):
        return len(self.buffer)


class DQAgentJAX:
    def __init__(self, env: MinBreakout, config: dict):
        self.env = env
        self.obs_shape = env.observation_space(env.default_params).shape
        self.num_actions = env.num_actions

        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.1)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)

        self.gamma = config.get("gamma", 0.99)
        self.batch_size = config.get("batch_size", 32)
        self.target_update_freq = config.get("target_update", 100)
        self.train_start = config.get("learning_starts", 1000)
        self.seed = config.get("seed", 0)

        self.key = jax.random.PRNGKey(self.seed)

        self.q_network = QNetwork(self.num_actions)
        self.target_network = QNetwork(self.num_actions)

        self.params = self.q_network.init(self.key, jnp.zeros(self.obs_shape))
        self.target_params = self.params

        self.optimizer = optax.adam(config.get("lr", 1e-3))
        self.opt_state = self.optimizer.init(self.params)

        self.replay_buffer = ReplayBuffer(capacity=config.get("buffer_size", 100_000))
        self.step_count = 0

    def select_action(self, obs):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        q_values = self.q_network.apply(self.params, obs)
        return int(jnp.argmax(q_values))

    @jax.jit
    def train_step(self, params, target_params, opt_state, batch):
        def loss_fn(p):
            q_values = self.q_network.apply(p, batch["state"])
            q_action = jnp.take_along_axis(q_values, batch["action"].reshape(-1, 1), axis=1).squeeze()

            target_q = self.target_network.apply(target_params, batch["next_state"])
            target_max = jnp.max(target_q, axis=1)
            target = batch["reward"] + self.gamma * (1.0 - batch["done"]) * target_max

            loss = jnp.mean((q_action - target) ** 2)
            return loss

        grads = jax.grad(loss_fn)(params)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    def train(self, episodes: int = 500):
        for ep in range(episodes):
            obs, state = self.env.reset_env(self.key, self.env.default_params)
            done = False
            episode_reward = 0

            while not done:
                action = self.select_action(obs)
                self.key, subkey = jax.random.split(self.key)
                next_obs, state, reward, done, _ = self.env.step_env(subkey, state, action, self.env.default_params)

                self.replay_buffer.add(obs, action, reward, next_obs, float(done))
                obs = next_obs
                episode_reward += reward
                self.step_count += 1

                if len(self.replay_buffer) >= self.train_start:
                    batch = self.replay_buffer.sample(self.batch_size)
                    self.params, self.opt_state = self.train_step(
                        self.params, self.target_params, self.opt_state, batch
                    )

                if self.step_count % self.target_update_freq == 0:
                    self.target_params = self.params

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)
            print(f"Episode {ep}: Reward = {episode_reward:.2f}, Epsilon = {self.epsilon:.3f}")

    def play(self, episodes=5):
        for ep in range(episodes):
            obs, state = self.env.reset_env(self.key, self.env.default_params)
            done = False
            reward_sum = 0
            while not done:
                q_vals = self.q_network.apply(self.params, obs)
                action = int(jnp.argmax(q_vals))
                self.key, subkey = jax.random.split(self.key)
                obs, state, reward, done, _ = self.env.step_env(subkey, state, action, self.env.default_params)
                reward_sum += reward
            print(f"Test Episode {ep}: Reward = {reward_sum:.2f}")