from policy_gradient import REINFORCEAgent
from omegaconf import DictConfig
import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn




class Policy(nn.Module):
    """
    Multi-layer perceptron mapping states to action probabilities.

    Implements a linear feed-forward network with one hidden layer and softmax output.

    Parameters
    ----------
    state_space : gym.spaces.Box
        Observation space defining the dimensionality of inputs.
    action_space : gym.spaces.Discrete
        Action space defining number of output classes.
    hidden_size : int, optional
        Number of units in the hidden layer (default is 128).
    """

    def __init__(
        self,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        hidden_size: int = 128,
    ):
        """
        Initialize the policy network.

        Parameters
        ----------
        state_space : gym.spaces.Box
            Observation space of the environment.
        action_space : gym.spaces.Discrete
            Action space of the environment.
        hidden_size : int, optional
            Number of hidden units. Defaults to 128.
        """
        super().__init__()
        self.state_dim = int(np.prod(state_space.shape))
        self.n_actions = action_space.n

        # TODO: Define two linear layers: self.fc1 and self.fc2
        # self.fc1 should map from self.state_dim to hidden_size
        # self.fc2 should map from hidden_size to self.n_actions
        self.fc1 = nn.Linear(self.state_dim, hidden_size)
        self.fc2 = nn.Linear(self.state_dim, hidden_size)
        self.fc3 = nn.Linear(hidden_size, self.n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute action probabilities for given state(s).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (state_dim,) or (batch_size, state_dim).

        Returns
        -------
        torch.Tensor
            Softmax probabilities over actions, shape (batch_size, n_actions).
        """
        # TODO: Apply fc1 followed by ReLU (Flatten input if needed)
        # TODO: Apply fc2 to get logits
        # TODO: Return softmax over logits along the last dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)

        #return torch.unflatten(torch.softmax(logits, dim=-1), 0, (1, self.n_actions))
        return torch.softmax(logits, dim=-1)


def set_seed(env: gym.Env, seed: int = 0) -> None:
    """
    Seed random number generators for reproducibility.

    Parameters
    ----------
    env : gym.Env
        Gymnasium environment to seed.
    seed : int, optional
        Seed value for NumPy, PyTorch, and environment (default is 0).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)

@hydra.main(
    config_path="../configs/agent/", config_name="reinforce", version_base="1.1"
)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training with Hydra configuration.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration with fields:
          env:
            name: str        # Gym environment id
          seed: int
          agent:
            lr: float
            gamma: float
            hidden_size: int
          train:
            episodes: int
            eval_interval: int
            eval_episodes: int
    """
    # Initialize environment and seed
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    # Instantiate agent with hyperparameters from config
    agent = REINFORCEAgent(
        env=env,
        lr=cfg.agent.lr,
        gamma=cfg.agent.gamma,
        seed=cfg.seed,
        hidden_size=cfg.agent.hidden_size,
    )

    # Train agent
    agent.train(
        num_episodes=cfg.train.episodes,
        eval_interval=cfg.train.eval_interval,
        eval_episodes=cfg.train.eval_episodes,
    )


if __name__ == "__main__":
    main()