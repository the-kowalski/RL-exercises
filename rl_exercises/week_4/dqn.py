"""
Deep Q-Learning implementation.
"""

from typing import Any, Dict, List, Tuple

import os
from math import exp

import gymnasium as gym
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_4.buffers import ReplayBuffer
from rl_exercises.week_4.networks import QNetwork
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve


def set_seed(env: gym.Env, seed: int = 0) -> None:
    """
    Seed Python, NumPy, PyTorch and the Gym environment for reproducibility.

    Parameters
    ----------
    env : gym.Env
        The Gym environment to seed.
    seed : int
        Random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    # some spaces also support .seed()
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


def moving_average(values: List[float], window_size: int = 10) -> List[float]:
    """
    Compute a moving average over a list of values.
    """
    if len(values) == 0:
        return []

    averaged = []
    for idx in range(len(values)):
        start = max(0, idx - window_size + 1)
        averaged.append(float(np.mean(values[start : idx + 1])))

    return averaged


def plot_training_curve(
    frames: List[int],
    rewards: List[float],
    output_path: str = "dqn_training_curve.png",
    window_size: int = 10,
) -> None:
    """
    Plot and save the Level 1 DQN training curve.
    """
    if len(frames) == 0 or len(rewards) == 0:
        print("No completed episodes were recorded, so no plot was created.")
        return

    mean_rewards = moving_average(rewards, window_size=window_size)

    plt.figure(figsize=(8, 5))
    plt.plot(
        frames, mean_rewards, label=f"Mean reward over last {window_size} episodes"
    )
    plt.xlabel("Frames")
    plt.ylabel("Mean reward")
    plt.title("DQN Training Curve on CartPole-v1")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved training curve to {output_path}")


def rewards_to_frame_grid(
    episode_frames: List[int],
    episode_rewards: List[float],
    frame_grid: np.ndarray,
    window_size: int = 10,
) -> np.ndarray:
    """
    Convert variable-length episodic rewards to a fixed frame grid.

    RLiable expects all seeds to have scores at matching evaluation points.
    DQN episodes finish at different frames for different seeds, so this helper
    maps each run to the same frame grid using the latest available smoothed
    episode reward.

    Parameters
    ----------
    episode_frames : list[int]
        Frame numbers at which episodes finished.
    episode_rewards : list[float]
        Episode returns.
    frame_grid : np.ndarray
        Shared x-axis for all seeds.
    window_size : int
        Window size for smoothing rewards before interpolation.

    Returns
    -------
    np.ndarray
        Rewards aligned to ``frame_grid``.
    """
    if len(episode_frames) == 0 or len(episode_rewards) == 0:
        return np.zeros_like(frame_grid, dtype=np.float32)

    smoothed_rewards = np.asarray(
        moving_average(episode_rewards, window_size=window_size),
        dtype=np.float32,
    )
    frames = np.asarray(episode_frames, dtype=np.int64)

    aligned_rewards = np.zeros_like(frame_grid, dtype=np.float32)
    for grid_idx, frame in enumerate(frame_grid):
        reward_idx = np.searchsorted(frames, frame, side="right") - 1
        if reward_idx >= 0:
            aligned_rewards[grid_idx] = smoothed_rewards[reward_idx]
        else:
            aligned_rewards[grid_idx] = smoothed_rewards[0]

    return aligned_rewards


def aggregate_over_time(metric_fn):
    """
    Wrap a RLiable aggregate metric so it can be applied at each frame.

    Parameters
    ----------
    metric_fn : callable
        RLiable aggregate metric.

    Returns
    -------
    callable
        Function mapping scores shaped ``(n_seeds, n_frames)`` to
        ``(n_frames,)``.
    """

    def aggregate(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores)

        if scores.ndim != 2:
            raise ValueError(
                f"Expected scores with shape (n_seeds, n_frames), got {scores.shape}."
            )

        return np.asarray(
            [
                metric_fn(scores[:, frame_idx : frame_idx + 1])
                for frame_idx in range(scores.shape[-1])
            ]
        )

    return aggregate


def run_level2_rliable_experiment(cfg: DictConfig) -> None:
    """
    Run DQN across multiple seeds and create RLiable plots for Level 2.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration.
    """
    seeds = [0, 1, 2, 3, 4]
    frame_grid = np.linspace(
        1,
        int(cfg.train.num_frames),
        num=100,
        dtype=np.int64,
    )

    all_seed_scores = []

    for seed in seeds:
        print(f"\nStarting DQN run for seed {seed}")

        env = gym.make(cfg.env.name)
        set_seed(env, seed)

        agent_kwargs = dict(
            buffer_capacity=cfg.agent.buffer_capacity,
            batch_size=cfg.agent.batch_size,
            lr=cfg.agent.learning_rate,
            gamma=cfg.agent.gamma,
            epsilon_start=cfg.agent.epsilon_start,
            epsilon_final=cfg.agent.epsilon_final,
            epsilon_decay=cfg.agent.epsilon_decay,
            target_update_freq=cfg.agent.target_update_freq,
            seed=seed,
        )

        agent = DQNAgent(env, **agent_kwargs)
        frames, rewards = agent.train(cfg.train.num_frames, cfg.train.eval_interval)
        env.close()

        aligned_rewards = rewards_to_frame_grid(
            episode_frames=frames,
            episode_rewards=rewards,
            frame_grid=frame_grid,
            window_size=10,
        )
        all_seed_scores.append(aligned_rewards)

    train_scores = {
        "dqn": np.asarray(all_seed_scores, dtype=np.float32),
    }

    plot_rliable_training_curves(
        frame_grid=frame_grid,
        train_scores=train_scores,
        optimal_score=500.0,
        reps=2000,
    )


def plot_rliable_training_curves(
    frame_grid: np.ndarray,
    train_scores: Dict[str, np.ndarray],
    optimal_score: float = 500.0,
    reps: int = 2000,
) -> None:
    """
    Plot Level 2 DQN training curves with RLiable.

    Parameters
    ----------
    frame_grid : np.ndarray
        Shared frame grid.
    train_scores : dict[str, np.ndarray]
        Scores per algorithm. Each array must have shape
        ``(n_seeds, n_frames)``.
    optimal_score : float
        Maximum expected CartPole-v1 return. The default is 500.
    reps : int
        Number of bootstrap repetitions for confidence intervals.
    """
    metric_specs = {
        "iqm": (
            aggregate_over_time(metrics.aggregate_iqm),
            "IQM Reward",
            "dqn_rliable_iqm_curve.png",
        ),
        "mean": (
            aggregate_over_time(metrics.aggregate_mean),
            "Mean Reward",
            "dqn_rliable_mean_curve.png",
        ),
        "median": (
            aggregate_over_time(metrics.aggregate_median),
            "Median Reward",
            "dqn_rliable_median_curve.png",
        ),
        "optimality_gap": (
            aggregate_over_time(
                lambda scores: metrics.aggregate_optimality_gap(
                    scores,
                    gamma=optimal_score,
                )
            ),
            "Optimality Gap",
            "dqn_rliable_optimality_gap_curve.png",
        ),
    }

    for metric_name, (metric_fn, ylabel, output_path) in metric_specs.items():
        estimates, confidence_intervals = get_interval_estimates(
            train_scores,
            metric_fn,
            reps=reps,
        )

        plt.figure(figsize=(8, 5))

        plot_sample_efficiency_curve(
            frame_grid,
            estimates,
            confidence_intervals,
            algorithms=list(train_scores.keys()),
            xlabel="Frames",
            ylabel=ylabel,
        )
        plt.title(f"DQN {metric_name.replace('_', ' ').title()} on CartPole-v1")
        plt.legend()
        plt.tight_layout()

        temporary_output_path = f"{output_path}.tmp.png"
        plt.savefig(
            temporary_output_path,
            format="png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        os.replace(temporary_output_path, output_path)

        print(f"Saved RLiable {metric_name} curve to {output_path}")


class DQNAgent(AbstractAgent):
    """
    Deep Q‐Learning agent with ε‐greedy policy and target network.

    Derives from AbstractAgent by implementing:
      - predict_action
      - save / load
      - update_agent
    """

    def __init__(
        self,
        env: gym.Env,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        seed: int = 0,
    ) -> None:
        """
        Initialize replay buffer, Q‐networks, optimizer, and hyperparameters.

        Parameters
        ----------
        env : gym.Env
            The Gym environment.
        buffer_capacity : int
            Max experiences stored.
        batch_size : int
            Mini‐batch size for updates.
        lr : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon_start : float
            Initial ε for exploration.
        epsilon_final : float
            Final ε.
        epsilon_decay : int
            Exponential decay parameter.
        target_update_freq : int
            How many updates between target‐network syncs.
        seed : int
            RNG seed.
        """
        super().__init__(
            env,
            buffer_capacity,
            batch_size,
            lr,
            gamma,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            target_update_freq,
            seed,
        )
        self.env = env
        set_seed(env, seed)

        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        # main Q‐network and frozen target
        self.q = QNetwork(obs_dim, n_actions)
        self.target_q = QNetwork(obs_dim, n_actions)
        self.target_q.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        # hyperparams
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        self.total_steps = 0  # for ε decay and target sync

    def epsilon(self) -> float:
        """
        Compute current ε by exponential decay.

        Returns
        -------
        float
            Exploration rate.
        """
        # TODO: implement exponential‐decayin
        # ε = ε_final + (ε_start - ε_final) * exp(-total_steps / ε_decay)
        # Currently, it is constant and returns the starting value ε
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * exp(
            -self.total_steps / self.epsilon_decay
        )

    def predict_action(
        self, state: np.ndarray, info: Dict[str, Any] = {}, evaluate: bool = False
    ) -> Tuple[int, Dict]:
        """
        Choose action via ε‐greedy (or purely greedy in eval mode).

        Parameters
        ----------
        state : np.ndarray
            Current observation.
        info : dict
            Gym info dict (unused here).
        evaluate : bool
            If True, always pick argmax(Q).

        Returns
        -------
        action : int
        info_out : dict
            Empty dict (compatible with interface).
        """
        if evaluate:
            # TODO: select purely greedy action from Q(s)
            # purely greedy
            t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                qvals = self.q.forward(t)
            action = int(torch.argmax(qvals, dim=1).item())
        else:
            # ε-greedy
            if np.random.rand() < self.epsilon():
                # TODO: sample random action
                action = int(np.random.randint(self.env.action_space.n))
            else:
                # TODO: select purely greedy action from Q(s)
                t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                qvals = self.q.forward(t)
                action = int(torch.argmax(qvals, dim=1).item())

        return action

    def save(self, path: str) -> None:
        """
        Save model & optimizer state to disk.

        Parameters
        ----------
        path : str
            File path.
        """
        torch.save(
            {
                "parameters": self.q.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load model & optimizer state from disk.

        Parameters
        ----------
        path : str
            File path.
        """
        checkpoint = torch.load(path)
        self.q.load_state_dict(checkpoint["parameters"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def update_agent(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        """
        Perform one gradient update on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, reward, next_state, done, info).

        Returns
        -------
        loss_val : float
            MSE loss value.
        """
        # unpack
        states, actions, rewards, next_states, dones, _ = zip(*training_batch)
        s = torch.tensor(np.array(states), dtype=torch.float32)
        a = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)
        r = torch.tensor(np.array(rewards), dtype=torch.float32)
        s_next = torch.tensor(np.array(next_states), dtype=torch.float32)
        mask = torch.tensor(np.array(dones), dtype=torch.float32)

        # current Q estimates for taken actions
        # TODO: pass batched states through self.q and gather Q(s,a)
        pred = self.q.forward(s).gather(1, a).squeeze(1)

        # TODO: compute TD target with frozen network
        with torch.no_grad():
            target_pred = self.target_q.forward(s_next)
            next_q = torch.max(target_pred, dim=1).values
            target = r + self.gamma * next_q * (1.0 - mask)

        loss = nn.MSELoss()(pred, target)

        # gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # occasionally sync target network
        if self.total_steps % self.target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        self.total_steps += 1
        return float(loss.item())

    def train(
        self, num_frames: int, eval_interval: int = 1000
    ) -> Tuple[List[int], List[float]]:
        """
        Run a training loop for a fixed number of frames.

        Parameters
        ----------
        num_frames : int
            Total environment steps.
        eval_interval : int
            Every this many episodes, print average reward.
        """
        state, _ = self.env.reset()
        ep_reward = 0.0
        recent_rewards: List[float] = []
        episode_frames: List[int] = []
        episode_rewards: List[float] = []

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            # store and step
            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            state = next_state
            ep_reward += reward

            # update if ready
            if len(self.buffer) >= self.batch_size:
                # TODO: sample batch from replay buffer
                batch = self.buffer.sample(self.batch_size)
                _ = self.update_agent(batch)

            if done or truncated:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                episode_frames.append(frame)
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                # logging
                if len(recent_rewards) % 10 == 0:
                    # TODO: compute avg over last eval_interval episodes and print
                    avg = sum(recent_rewards[-eval_interval:]) / (
                        eval_interval
                        if len(recent_rewards) >= eval_interval
                        else len(recent_rewards)
                    )
                    print(
                        f"Frame {frame}, AvgReward(10): {avg:.2f}, ε={self.epsilon():.3f}"
                    )

        print("Training complete.")
        return episode_frames, episode_rewards


@hydra.main(config_path="../configs/agent/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    if cfg.get("run_level2", False):
        run_level2_rliable_experiment(cfg)
        return

    # 1) build env
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    # 2) TODO: map config → agent kwargs
    agent_kwargs = dict(
        buffer_capacity=cfg.agent.buffer_capacity,
        batch_size=cfg.agent.batch_size,
        lr=cfg.agent.learning_rate,
        gamma=cfg.agent.gamma,
        epsilon_start=cfg.agent.epsilon_start,
        epsilon_final=cfg.agent.epsilon_final,
        epsilon_decay=cfg.agent.epsilon_decay,
        target_update_freq=cfg.agent.target_update_freq,
        seed=cfg.seed,
    )

    # 3) TODO:instantiate & train
    agent = DQNAgent(env, **agent_kwargs)
    frames, rewards = agent.train(cfg.train.num_frames, cfg.train.eval_interval)

    plot_training_curve(
        frames=frames,
        rewards=rewards,
        output_path="dqn_training_curve.png",
        window_size=10,
    )


if __name__ == "__main__":
    main()
