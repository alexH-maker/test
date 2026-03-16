"""Implementation of MAPF-EGT algorithm and utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from mapf_egt.env import manhattan_distance, MultiAgentGridEnv

Position = Tuple[int, int]  # paper: state s = (x, y)


@dataclass
class MAPFEGTConfig:
    theta: float = 0.15  # paper: good-trajectory threshold η (0 <= u <= η <= 1)
    v: float = 2.5       # paper: bad-trajectory threshold α (α > 1)
    U: float = 1.2       # paper: reward coefficient U (r = 1 - (1-u)^U)
    alpha: float = 0.5   # paper: positive update step α (count += α)
    beta: float = 0.3    # paper: negative update step β (count -= β)
    eta: float = 0.08    # paper:  epusen for π_trained
    goal_bias: float = 0.9  # implementation: optional bias toward nearer targets (set 0 for strict paper)
    total_episodes: int = 300  # paper: total episodes
    max_steps: int = 400      # paper: max steps per episode T
    initial_count: float = 1.0  # paper: count(s,a) initialized to 1
    seed: int | None = None  # implementation: RNG seed策略采样的随机性。

    def __post_init__(self) -> None:
        assert 0 <= self.theta <= 1, "theta must be between 0 and 1"
        assert self.v >= 1, "V must be at least 1"
        assert self.U > 1, "U must be greater than 1"
        assert 0 <= self.alpha
        assert 0 <= self.beta
        assert 0 <= self.eta <= 1
        assert 0 <= self.goal_bias <= 1
        assert 0 <= self.initial_count


class MAPFEGTTrainer:
    """Trainer that follows the MAPF-EGT pseudocode described in the paper."""

    def __init__(self, env: MultiAgentGridEnv, config: MAPFEGTConfig) -> None:
        self.env = env
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.n_actions = len(env.ACTIONS)
        self.pi_init = np.full(self.n_actions, 1 / self.n_actions, dtype=float)  # paper: π_init(a|s)=1/|A|
        self.counts: Dict[Position, np.ndarray] = {}  # paper: count(s,a)
        self.pi_trained: Dict[Position, np.ndarray] = {}  # paper: π_trained(a|s)
        self.last_episode_paths: List[List[Position]] = []  # logging/visualization only
        # ensure a valid policy exists before training starts
        self.construct_policy()

    def _ensure_state_counts(self, state: Position) -> np.ndarray:
        if state not in self.counts:
            # paper: initialize count(s,a)=1 for new state
            self.counts[state] = np.full(self.n_actions, self.config.initial_count, dtype=float)
        return self.counts[state]

    def _actions_toward_target(self, state: Position) -> List[int]:
        if not self.env.targets:
            return []
        current_dist = min(manhattan_distance(state, t) for t in self.env.targets)
        good_actions: List[int] = []
        for action_idx, (dx, dy) in self.env.MOVE_MAP.items():
            candidate = (state[0] + dx, state[1] + dy)
            if not self.env._is_permissible(candidate):
                continue
            cand_dist = min(manhattan_distance(candidate, t) for t in self.env.targets)
            if cand_dist < current_dist:
                good_actions.append(action_idx)
        return good_actions

    def construct_policy(self) -> None:
        """Derive a smoothed strategy pi_trained from the current counts."""
        new_policy: Dict[Position, np.ndarray] = {}
        base = np.full(self.n_actions, 1 / self.n_actions, dtype=float)
        for state, counts in self.counts.items():
            positive_counts = np.maximum(counts, 0.0)  # paper: count(s,a)<=0 treated as 0 prob
            total = positive_counts.sum()
            if total <= 0:
                proportion = base
            else:
                proportion = positive_counts / total  # paper: count(s,a) / sum_a count(s,a)
            smoothed = (1 - self.config.eta) * proportion + self.config.eta * self.pi_init  # paper: smoothing

            """数值安全处理，防止概率分布失效,处理不能归一化的问题"""

            if smoothed.sum() <= 0:
                smoothed = self.pi_init.copy()
            else:
                smoothed = smoothed / smoothed.sum()
            new_policy[state] = smoothed

        self.pi_trained = new_policy

    def sample_action(self, state: Position) -> int:
        probs = self.pi_trained.get(state, self.pi_init).copy()
        # light goal-biased exploration toward nearer targets (no shortest-path planning)
        if self.config.goal_bias > 0:
            good_actions = self._actions_toward_target(state)  # actions that reduce Manhattan distance
            if good_actions:
                bias = np.zeros(self.n_actions, dtype=float)
                bias[good_actions] = 1.0 / len(good_actions)
                probs = (1 - self.config.goal_bias) * probs + self.config.goal_bias * bias
        total = probs.sum()
        if total <= 0:
            probs = self.pi_init
            total = probs.sum()
        probs = probs / total
        return int(self.rng.choice(self.n_actions, p=probs))


        """updateFunction"""

    def update_trajectory(
        self,
        trajectory: Sequence[Tuple[Position, int]],
        start: Position,
        final: Position,
        reached_target: bool = False,
    ) -> None:
        """Apply reward/punishment rules to recorded (s,a) pairs."""
        if not trajectory:
            return
        L = len(trajectory)   # paper: trajectory length L (states)
        d = manhattan_distance(start, final)  # paper: Manhattan distance d between start and final
        d = max(d, 1e-6)
        u = L / d  # paper: fitness u = L / d
        if reached_target or u <= self.config.theta:
            r = 1 - (1 - u) * self.config.U  # paper: reward coefficient r
            for state, action in trajectory:
                counts = self._ensure_state_counts(state)
                if counts[action] >= r:
                    counts[action] += self.config.alpha
        elif u >= self.config.v:
            for state, action in trajectory:
                counts = self._ensure_state_counts(state)
                counts[action] = max(0.0, counts[action] - self.config.beta)
        else:
            return
        self.construct_policy()

    def train(self) -> None:
        """Run MAPF-EGT for the configured number of episodes."""
        for episode in range(self.config.total_episodes):
            observations = self.env.reset()  # sample initial states from iota
            start_states = [tuple(pos) for pos in observations]  # s0 per agent
            current_states = start_states.copy()  # current s per agent
            state_histories: List[List[Position]] = [[state] for state in start_states]  # path per agent
            trajectories: List[List[Tuple[Position, int]]] = [[] for _ in range(self.env.n_agents)]  # (s,a)
            reached = [False] * self.env.n_agents  # whether agent reached any target
            step = 0
            while step < self.config.max_steps and not all(reached):
                actions: List[int] = []
                for idx in range(self.env.n_agents):
                    if reached[idx]:
                        actions.append(self.n_actions - 1)
                        continue
                    action = self.sample_action(current_states[idx])  # sample from current policy
                    actions.append(action)
                    trajectories[idx].append((current_states[idx], action))  # record (s,a)
                next_obs, _, dones, info = self.env.step(actions)  # transition via Δ
                next_states = [tuple(pos) for pos in next_obs]  # s'
                for idx, next_state in enumerate(next_states):
                    if not reached[idx]:
                        state_histories[idx].append(next_state)  # record path for plotting
                for idx, done in enumerate(dones):
                    if not reached[idx] and done:
                        final_state = next_states[idx]  # terminal s_T
                        reached_goal = final_state in self.env.targets  # hit Φ
                        self.update_trajectory(trajectories[idx], start_states[idx], final_state, reached_target=reached_goal)
                        trajectories[idx] = []
                        reached[idx] = True
                current_states = next_states
                step += 1
            for idx in range(self.env.n_agents):
                if not reached[idx]:
                    final_state = current_states[idx]  # truncation terminal
                    reached_goal = final_state in self.env.targets  # whether Φ reached by truncation
                    self.update_trajectory(trajectories[idx], start_states[idx], final_state, reached_target=reached_goal)
            self.last_episode_paths = state_histories


def visualize_trajectories(
    env: MultiAgentGridEnv,
    agent_paths: Iterable[Sequence[Position]],
    save_path: str = "mapf_egt_paths.png",
    show: bool = False,
) -> None:
    """Plot agent trajectories with obstacles, targets, and start locations highlighted."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(-0.5, env.height - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("MAPF-EGT trajectories")

    if env.obstacles:
        ox, oy = zip(*env.obstacles)
        ax.scatter(ox, oy, c="red", marker="s", label="obstacle", s=40)
    if env.targets:
        tx, ty = zip(*env.targets)
        ax.scatter(tx, ty, c="green", marker="s", label="target", s=60)
    for path_idx, path in enumerate(agent_paths):
        if not path:
            continue
        xs, ys = zip(*path)
        ax.plot(xs, ys, c="royalblue", linewidth=1, label="trajectory" if path_idx == 0 else "")
        ax.scatter(xs[0], ys[0], c="yellow", edgecolor="black", s=80, marker="o", label="start" if path_idx == 0 else "")
        ax.scatter(xs[-1], ys[-1], c="blue", edgecolor="black", s=40, marker="x", label="current" if path_idx == 0 else "")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(save_path)
    if show:
        plt.show()


