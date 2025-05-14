import jax
from breakout_jax import MinBreakout
from jax_dqn_agent import DQAgentJAX  

# Create environment and config
env = MinBreakout()
config = {
    "epsilon_start": 1.0,
    "epsilon_end": 0.1,
    "epsilon_decay": 0.995,
    "gamma": 0.99,
    "lr": 1e-3,
    "batch_size": 32,
    "buffer_size": 100_000,
    "learning_starts": 1000,
    "target_update": 100,
    "seed": 0,
}

# Instantiate and train agent
agent = DQAgentJAX(env, config)
agent.train(episodes=500)   # <- run training loop
agent.play(episodes=5)      # <- test the trained agent
