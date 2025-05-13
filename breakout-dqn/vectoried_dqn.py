import os
import numpy as np
import torch
from copy import deepcopy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.dqn import DQN
from typing import Dict, Any

import wandb
import ale_py

from es import BaseModel


class BreakoutVectorizedDQN(BaseModel):
    def __init__(self, envs, config):
        # Initialize the env and DQN model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.envs = envs
        self.policy_params = []
        self.base_model = DQN(
            "CnnPolicy",
            envs,
            learning_rate=config["learning_rate"],
            buffer_size=config["buffer_size"],
            batch_size=config["batch_size"],
            learning_starts=config["learning_starts"],
            target_update_interval=config["target_update_interval"],
            train_freq=config["train_freq"],
            gradient_steps=config["gradient_steps"],
            exploration_fraction=config["exploration_fraction"],
            exploration_final_eps=config["exploration_final_eps"],
            optimize_memory_usage=config["optimize_memory_usage"],
            verbose=config["verbose"],
            tensorboard_log=config["tensorboard_log"],
        )
        self.population_size = config["population_size"]
        
        base_params = self.get_base_parameters()
        # Create population_size copies of the policy network and initialize all policies with the same parameters
        for _ in range(self.population_size):
            params = base_params.clone()
            self.policy_params.append(params)
        
        # Stack all parameters into a single tensor [population_size, param_dim]
        self.policy_params = torch.stack(self.policy_params).to(self.device)
        
    def get_base_parameters(self) -> torch.Tensor:
        return self.base_model.policy.parameters().to(self.device)
    
    def get_parameters(self) -> torch.Tensor:
        return self.policy_params
    
    def set_parameters(self, params_batch: torch.Tensor) -> None:
        self.policy_params = params_batch.to(self.device)
        
    def update_policy(self, params: torch.Tensor, policy: nn.Module) -> None:
        start = 0
        for param in policy.parameters():
            size = param.numel()
            param.data.copy_(params[start:start + size].view(param.size()))
            start += size
        
    def forward_batch(self, observations: torch.Tensor) -> torch.Tensor:
        batch_actions = []
        
        for i in range(self.population_size):
            self.update_policy(self.policy_params[i], self.base_model.policy)
            
            with torch.no_grad():
                observation = observations[i:i+1].to(self.device)
                action = self.base_model.policy(observation)[0].argmax(dim=1)
                batch_actions.append(action)
        
        return torch.stack(batch_actions)
       
    
    def evaluate(self, num_episodes: int) -> float:
        total_rewards = torch.zeros(self.population_size).to(self.device)
        
        for _ in range(num_episodes):
            observations = torch.float32(self.envs.reset()).to(self.device)
            dones = [False] * self.population_size
            episode_rewards = torch.zeros(self.population_size).to(self.device)
            
            while not all(dones):
                actions = self.forward_batch(observations)
                next_observations, next_rewards, next_dones, _ = self.envs.step(actions.cpu().numpy())
                episode_rewards += torch.tensor(next_rewards).to(self.device)
            
            total_rewards += episode_rewards
        
        return total_rewards / num_episodes
    
    def save_checkpoint(self, path: str, generation: int, metrics: Dict[str, Any]) -> None:
        """Save a checkpoint of the model."""
        checkpoint = {
            "generation": generation,
            "model_state_dict": self.model.policy.state_dict(),
            "metrics": metrics,
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")