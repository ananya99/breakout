import os
import numpy as np
import torch
from copy import deepcopy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.dqn import DQN
from typing import Dict, Any
import numpy as np
import torch.nn as nn
import wandb
import ale_py

from base_model import BaseModel


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
        # Convert parameters generator to a list of tensors and concatenate them
        params = []
        for param in self.base_model.policy.parameters():
            params.append(param.data.view(-1))
        return torch.cat(params).to(self.device)
    
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
            obs = self.envs.reset()
            observations = torch.tensor(obs, dtype=torch.float32).to(self.device)
            dones = [False] * self.population_size
            episode_rewards = torch.zeros(self.population_size).to(self.device)
            
            while not all(dones):
                actions = self.forward_batch(observations)
                next_observations, next_rewards, next_dones, _ = self.envs.step(actions.cpu().numpy())
                observations = torch.tensor(next_observations, dtype=torch.float32).to(self.device)
                print(f"Next rewards shape= {next_rewards.shape}")
                episode_rewards += torch.tensor(next_rewards).to(self.device)
                dones = next_dones
            
            total_rewards += episode_rewards
        
        return total_rewards / num_episodes
    
    def evaluate_batch(self, num_episodes: int) -> float:
        return self.evaluate(num_episodes)
    
    def save_checkpoint(self, path: str, generation: int, metrics: Dict[str, Any]) -> None:
        """Save a checkpoint of the model."""
        try:
            # Ensure directory exists
            save_dir = os.path.dirname(path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Save state dict and metrics separately
            state_dict_path = os.path.join(save_dir, f"state_dict_{generation}.pt")
            metrics_path = os.path.join(save_dir, f"metrics_{generation}.pt")
            
            # Save state dict
            torch.save(self.base_model.policy.state_dict(), state_dict_path)
            
            # Save metrics separately
            metrics_dict = {
                "generation": generation,
                "metrics": metrics
            }
            torch.save(metrics_dict, metrics_path)
            
            print(f"Saved checkpoint files to {save_dir}")
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
            # Try to save in home directory as fallback
            home_dir = os.path.expanduser("~")
            fallback_dir = os.path.join(home_dir, "breakout_checkpoints")
            os.makedirs(fallback_dir, exist_ok=True)
            
            state_dict_path = os.path.join(fallback_dir, f"state_dict_{generation}.pt")
            metrics_path = os.path.join(fallback_dir, f"metrics_{generation}.pt")
            
            torch.save(self.base_model.policy.state_dict(), state_dict_path)
            torch.save(metrics_dict, metrics_path)
            print(f"Saved checkpoint files to fallback location: {fallback_dir}")