"""Custom grid environment compatible with or without Gym installed."""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Set, Tuple

try:
    from gym import spaces
except ImportError:  # pragma: no cover
    spaces = None

import matplotlib.pyplot as plt
import numpy as np

Position = Tuple[int, int]  # paper: state s = (x, y)


class SimpleMultiDiscrete:
    def __init__(self, nvec: Sequence[int]) -> None:
        self.nvec = list(nvec)


class SimpleBox:
    def __init__(self, low: float, high: float, shape: Sequence[int], dtype: type) -> None:
        self.low = low
        self.high = high
        self.shape = tuple(shape)
        self.dtype = dtype


class SimpleDiscrete:
    def __init__(self, n: int) -> None:
        self.n = n


def manhattan_distance(a: Position, b: Position) -> int:
    """Return Manhattan distance between two grid coordinates."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def expected_min_distance_to_obstacles(
    state: Position, obstacles: Iterable[Position], width: int, height: int
) -> float:
    """Estimate the nearest obstacle/boundary distance from this position."""
    boundary_distance = min(state[0], width - 1 - state[0], state[1], height - 1 - state[1])  # to walls
    distances = [boundary_distance]
    for obstacle in obstacles:
        distances.append(manhattan_distance(state, obstacle))
    return float(min(distances)) if distances else float(boundary_distance)


def build_trajectory(states: Sequence[Position], actions: Sequence[int]) -> List[Tuple[Position, int]]:
    """Pair each state with the action that led out of it."""
    return list(zip(states, actions))


class MultiAgentGridEnv:
    """Minimal multi-agent gridworld that mirrors the paper's setup."""

    ACTIONS = ("up", "down", "left", "right", "stop")  #  action set A
    MOVE_MAP = {
        0: (0, -1),
        1: (0, 1),
        2: (-1, 0),
        3: (1, 0),
        4: (0, 0),
    }

    def __init__(
        self,
        width: int = 100,
        height: int = 100,
        n_agents: int = 10,  # paper: number of agents
        obstacle_density: float = 0.8,  # paper: obstacle ratio for O
        num_targets: int = 3,  # paper: size of target set Φ
        transition_success_prob: float = 0.9,  # paper: action success prob p
        delta_1: float = -0.04,  # paper: δ1 (non-target step)
        delta_2: float = -0.25,  # paper: δ2 (invalid move / collision)
        delta_3: float = 1.0,  # paper: δ3 (reach target)
        seed: Optional[int] = None, # 障碍/目标/初始位置的随机性。
    ) -> None:
        assert 0 <= obstacle_density < 1
        assert 0 <= transition_success_prob <= 1
        assert delta_2 < delta_1 < 0 < delta_3
        assert num_targets >= 1

        self.width = width
        self.height = height
        self.n_agents = n_agents
        self.obstacle_density = obstacle_density
        self.num_targets = num_targets
        self.transition_success_prob = transition_success_prob
        self.delta_1 = delta_1
        self.delta_2 = delta_2
        self.delta_3 = delta_3
        self.rng = np.random.default_rng(seed)
        self.obstacles: Set[Position] = set()  # paper: obstacle set O
        self.targets: Set[Position] = set()  # paper: target set Φ
        self.agent_positions: List[Position] = []
        self.agent_done: List[bool] = [False] * n_agents
        self.steps = 0
        self._reachable_free: Set[Position] = set()  # implementation: ensure starts can reach Φ
        self._build_static_sets()
        self._init_spaces()

    def _init_spaces(self) -> None:
        action_space_cls = spaces.MultiDiscrete if spaces is not None else SimpleMultiDiscrete
        box_cls = spaces.Box if spaces is not None else SimpleBox
        discrete_cls = spaces.Discrete if spaces is not None else SimpleDiscrete
        self._base_action_space = discrete_cls(len(self.ACTIONS))
        self.action_space = action_space_cls([len(self.ACTIONS)] * self.n_agents)
        self.observation_space = box_cls(
            low=0,
            high=max(self.width - 1, self.height - 1),
            shape=(self.n_agents, 2),
            dtype=np.int32,
        )

    def _build_static_sets(self) -> None:
        self.targets = self._generate_target_set()  # paper: generate Φ
        # regenerate obstacles until targets are reachable from at least one free start cell
        attempts = 0
        while True:
            attempts += 1
            self.obstacles = self._generate_obstacles()  # paper: generate O
            if self._targets_reachable():
                break
            if attempts > 100:
                # fall back to current obstacles to avoid infinite loop
                break

    def _generate_target_set(self) -> Set[Position]:
        target_set: Set[Position] = set()
        while len(target_set) < self.num_targets:
            candidate = (int(self.rng.integers(self.width)), int(self.rng.integers(self.height)))  # sample Φ
            target_set.add(candidate)
        return target_set

    def _generate_obstacles(self) -> Set[Position]:
        total_cells = self.width * self.height
        num_obstacles = int(total_cells * self.obstacle_density)  # |O|
        obstacle_set: Set[Position] = set()
        while len(obstacle_set) < num_obstacles:
            candidate = (int(self.rng.integers(self.width)), int(self.rng.integers(self.height)))  # sample O
            if candidate in self.targets:
                continue
            obstacle_set.add(candidate)
        return obstacle_set

    def _targets_reachable(self) -> bool:
        """Check whether free cells connected to targets exist; cache them for start sampling."""
        if not self.targets:
            self._reachable_free = set()
            return True
        free_cells = {
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in self.obstacles and (x, y) not in self.targets
        }
        if not free_cells:
            self._reachable_free = set()
            return False
        frontier = list(self.targets)  # multi-source BFS from Φ
        visited = set(frontier)
        while frontier:
            x, y = frontier.pop()
            for dx, dy in self.MOVE_MAP.values():
                nxt = (x + dx, y + dy)
                if nxt in visited:
                    continue
                if not self._is_within_bounds(nxt) or nxt in self.obstacles:
                    continue
                visited.add(nxt)
                frontier.append(nxt)
        reachable_free = {cell for cell in visited if cell not in self.targets and cell not in self.obstacles}  # starts that can reach Φ
        self._reachable_free = reachable_free
        return bool(reachable_free)

    def reset(self) -> List[Position]:
        self.agent_positions = []
        self.agent_done = [False] * self.n_agents
        self.steps = 0
        for _ in range(self.n_agents):
            self.agent_positions.append(self._sample_start_position())
        return list(self.agent_positions)

    def _sample_start_position(self) -> Position:
        # paper: initial state distribution ι (random legal start)
        candidates = list(self._reachable_free) if self._reachable_free else None
        while True:
            if candidates:
                candidate = candidates[int(self.rng.integers(len(candidates)))]
            else:
                candidate = (int(self.rng.integers(self.width)), int(self.rng.integers(self.height)))
            if candidate in self.obstacles or candidate in self.targets:
                continue
            if candidate in self.agent_positions:
                continue
            return candidate

    def step(self, actions: Sequence[int]) -> Tuple[List[Position], List[float], List[bool], dict]:
        assert len(actions) == self.n_agents, "Action must be provided for every agent."
        next_positions: List[Position] = [self.agent_positions[idx] for idx in range(self.n_agents)]  # default stay
        rewards: List[float] = [0.0] * self.n_agents  # per-agent reward
        dones: List[bool] = [self.agent_done[idx] for idx in range(self.n_agents)]  # per-agent done
        invalid_flags: List[bool] = [False] * self.n_agents  # invalid move flags
        collision_flags: List[bool] = [False] * self.n_agents  # agent-agent collision flags

        proposed: List[Position | None] = [None] * self.n_agents  # candidate s' per agent
        for idx, action_idx in enumerate(actions):
            if self.agent_done[idx]:
                continue
            allowed_action = self._sample_transition(action_idx)  # apply Δ
            candidate = self._move(self.agent_positions[idx], allowed_action)  # proposed s'
            if not self._is_permissible(candidate):
                rewards[idx] = self.delta_2  # paper: δ2 for invalid action
                invalid_flags[idx] = True
                continue
            proposed[idx] = candidate

        # 禁止多个agent同时占用单元格
        occupancy: Dict[Position, int] = {}
        for pos in proposed:
            if pos is None:
                continue
            occupancy[pos] = occupancy.get(pos, 0) + 1  # count claims per cell

        for idx in range(self.n_agents):
            if self.agent_done[idx]:
                continue
            if proposed[idx] is None:
                # already handled invalid move
                continue
            pos = proposed[idx]
            if occupancy.get(pos, 0) > 1:
                collision_flags[idx] = True
                rewards[idx] = self.delta_2  # paper: treat collision as invalid
                next_positions[idx] = self.agent_positions[idx]
                dones[idx] = False
            else:
                arrived = pos in self.targets  # reached Φ
                rewards[idx] = self.delta_3 if arrived else self.delta_1  # paper: δ3/δ1
                next_positions[idx] = pos
                dones[idx] = arrived
                self.agent_done[idx] = arrived
        self.agent_positions = next_positions
        self.steps += 1
        info = {
            "min_distances": [expected_min_distance_to_obstacles(pos, self.obstacles, self.width, self.height) for pos in next_positions],
            "invalid_flags": invalid_flags,
            "agent_collision_flags": collision_flags,
        }
        return next_positions, rewards, dones, info

    def _move(self, state: Position, action_idx: int) -> Position:
        dx, dy = self.MOVE_MAP.get(action_idx, (0, 0))
        return (state[0] + dx, state[1] + dy)

    def _is_within_bounds(self, state: Position) -> bool:
        return 0 <= state[0] < self.width and 0 <= state[1] < self.height

    def _is_permissible(self, state: Position) -> bool:
        return self._is_within_bounds(state) and state not in self.obstacles

    def permissible_actions(self, state: Position) -> Set[int]:
        #  permissible action set E(s)
        safe_actions: Set[int] = set()
        for action_idx in range(len(self.ACTIONS)):
            candidate = self._move(state, action_idx)
            if self._is_permissible(candidate) or action_idx == len(self.ACTIONS) - 1:
                safe_actions.add(action_idx)
        return safe_actions

    def _sample_transition(self, action_idx: int) -> int:
        #  transition model Δ with success prob p, else random action
        if self.rng.random() < self.transition_success_prob:
            return action_idx
        return int(self.rng.integers(len(self.ACTIONS)))

    def is_obstacle(self, state: Position) -> bool:
        return state in self.obstacles

    def is_target(self, state: Position) -> bool:
        return state in self.targets

    def render(self) -> None:
        grid = np.full((self.height, self.width), ".")
        for obs in self.obstacles:
            grid[obs[1], obs[0]] = "#"
        for target in self.targets:
            grid[target[1], target[0]] = "G"
        for idx, pos in enumerate(self.agent_positions):
            symbol = "X" if self.agent_done[idx] else "A"
            grid[pos[1], pos[0]] = symbol
        print("\n".join("".join(row) for row in grid[::-1]))

    def trajectory_stats(self, trajectory: Sequence[Position]) -> Tuple[int, float]:
        if not trajectory:
            return 0, float("inf")
        path_length = len(trajectory)
        distances = [expected_min_distance_to_obstacles(pos, self.obstacles, self.width, self.height) for pos in trajectory]
        return path_length, float(sum(distances) / len(distances))
