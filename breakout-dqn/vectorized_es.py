import os
import numpy as np
import torch
import wandb
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Type
from stable_baselines3.common.vec_env import VecEnv
from base_model import BaseModel

class EvolutionStrategy:
    def __init__(
        self,
        envs: VecEnv,
        model: Type[BaseModel],
        dqn_config: Dict[str, Any],
        population_size: int = 50,
        sigma: float = 0.1,
        learning_rate: float = 0.01,
        num_episodes: int = 5,
        save_freq: int = 10,
        checkpoint_dir: str = "es_checkpoints",
    ):
        """
        Initialize Evolution Strategy.
        
        Args:
            model: Model that implements BaseModel interface
            population_size: Number of individuals in population
            sigma: Standard deviation of noise
            learning_rate: Learning rate for parameter updates
            num_episodes: Number of episodes to evaluate each individual
            save_freq: How often to save checkpoints (in generations)
            checkpoint_dir: Directory to save checkpoints
        """
        self.envs = envs
        self.model = model(envs, dqn_config)
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.num_episodes = num_episodes
        self.save_freq = save_freq
        
        # Setup directories
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = f"run_{timestamp}"
        self.checkpoint_dir = os.path.join(checkpoint_dir, self.run_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def _evaluate_population(
        self, 
        theta: torch.Tensor, 
        noises: torch.Tensor
    ) -> List[float]:
        """Evaluate entire population and return rewards."""
         # Create perturbed parameters
        perturbed_params = theta.unsqueeze(0) + self.sigma * noises
                
        self.model.set_parameters(perturbed_params)
        rewards = self.model.evaluate_batch(self.num_episodes)
        print(f"Rewards = {rewards}")

        return rewards

    def train(self, num_generations: int = 1000) -> None:
        """Train using evolution strategy."""
        # Get initial parameters
        theta = self.model.get_base_parameters()
        best_reward = float('-inf')
        
        for generation in range(num_generations):
            print(f"\nGeneration {generation}/{num_generations}")
            
            # Generate random noise for each member of the population
            noises = torch.randn(self.population_size, *theta.shape, device=self.device)
            
            # Evaluate population
            rewards = self._evaluate_population(theta, noises)
            
            # Compute reward statistics
            mean_reward = rewards.mean()
            max_reward = rewards.max()
            
            normalised_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            weighted_sum = noises*normalised_rewards.unsqueeze(-1).sum(dim=0)
            
            # Update best reward and save checkpoint if needed
            if max_reward > best_reward:
                best_reward = max_reward
                if generation % self.save_freq == 0:
                    metrics = {
                        "reward": best_reward,
                        "generation": generation,
                        "mean_reward": mean_reward,
                        "max_reward": max_reward,
                    }
                    checkpoint_path = os.path.join(
                        self.checkpoint_dir, 
                        f"es_checkpoint_{generation}.pt"
                    )
                    self.model.save_checkpoint(checkpoint_path, generation, metrics)
            
            # Update parameters
            theta = theta + self.learning_rate / (self.population_size * self.sigma) * weighted_sum
            
            # Log metrics
            wandb.log({
                "generation": generation,
                "mean_reward": mean_reward,
                "max_reward": max_reward,
                "best_reward": best_reward,
            })
            
            print(f"Generation {generation} stats:")
            print(f"Mean Reward: {mean_reward:.2f}")
            print(f"Max Reward: {max_reward:.2f}")
            print(f"Best Reward: {best_reward:.2f}")
        
        # Save final checkpoint
        final_metrics = {
            "reward": best_reward,
            "generation": num_generations,
            "mean_reward": mean_reward,
            "max_reward": max_reward,
        }
        final_path = os.path.join(self.checkpoint_dir, f"es_checkpoint_final.pt")
        self.model.save_checkpoint(final_path, num_generations, final_metrics)