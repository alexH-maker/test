"""Batch runner for MAPF-EGT that can loop run_mapf_egt logic multiple times
while customizing agent数量和target数量。

Usage example:
    python batch_run_mapf_egt.py --runs 5 --agents 12 --targets 4 --episodes 200 --replay
"""

import argparse
import csv
import time
from pathlib import Path
from typing import List, Tuple, Optional

from mapf_egt import MAPFEGTConfig, MAPFEGTTrainer, MultiAgentGridEnv, visualize_trajectories
from run_mapf_egt import animate_recorded_paths, maybe_write_header, simplify_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch run MAPF-EGT multiple times.")
    p.add_argument("--runs", type=int, default=10, help="how many independent runs to execute")
    p.add_argument("--agents", type=int, default=10, help="number of agents per run")
    p.add_argument("--targets", type=int, default=3, help="number of targets per run")
    p.add_argument("--width", type=int, default=100, help="grid width")
    p.add_argument("--height", type=int, default=100, help="grid height")
    p.add_argument("--episodes", type=int, default=400, help="training episodes per run")
    p.add_argument("--max-steps", type=int, default=400, help="max steps per episode")
    p.add_argument("--goal-bias", type=float, default=0.75, help="exploit probability when sampling actions")
    p.add_argument("--obstacle-density", type=float, default=None, help="ratio of obstacle cells (None -> env default)")
    p.add_argument("--seed", type=int, default=42, help="base random seed; each run adds the run index")
    p.add_argument("--out-csv", type=Path, default=Path("mapf_egt_runs.csv"), help="output metrics csv")
    p.add_argument("--fig-dir", type=Path, default=Path("mapf_egt_figures"), help="directory to store figures")
    p.add_argument("--live-pause", type=float, default=0.02, help="delay between frames for live replay")
    p.add_argument("--replay", action="store_true", help="show live replay after each run")
    p.add_argument("--no-replay", action="store_false", dest="replay", help="disable live replay")
    p.set_defaults(replay=False)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> MAPFEGTConfig:
    """Use the tuned defaults from run_mapf_egt.py."""
    return MAPFEGTConfig(
        theta=0.25,   # more lenient good-threshold (0.20-0.25)
        v=2.0,        # tighter bad-threshold (1.8-2.0)
        U=1.5,        # reward coeff (>1, e.g., 1.2-1.5)
        alpha=1.2,    # stronger positive update (1.0-1.2)
        beta=0.2,     # softer penalty (0.2-0.5)
        eta=0.05,     # lower smoothing noise (0.02-0.05)
        goal_bias=args.goal_bias,
        total_episodes=args.episodes,
        max_steps=args.max_steps,
    )


def run_single(run_idx: int, args: argparse.Namespace, session_ts: str) -> Tuple[float, List[int], Path]:
    env_kwargs = dict(
        width=args.width,
        height=args.height,
        n_agents=args.agents,
        num_targets=args.targets,
        seed=args.seed + run_idx if args.seed is not None else None,
    )
    if args.obstacle_density is not None:
        env_kwargs["obstacle_density"] = args.obstacle_density
    env = MultiAgentGridEnv(**env_kwargs)
    config = build_config(args)
    trainer = MAPFEGTTrainer(env, config)

    t0 = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - t0  # exclude visualization

    lengths = [len(path) - 1 for path in trainer.last_episode_paths]
    cleaned_paths = [simplify_path(traj) for traj in trainer.last_episode_paths]

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = args.fig_dir / f"mapf_egt_traces_{session_ts}_run{run_idx}.png"
    visualize_trajectories(env, cleaned_paths, save_path=str(fig_path))

    if args.replay:
        print(f"[run {run_idx}] Opening live replay window...")
        animate_recorded_paths(env, trainer.last_episode_paths, pause=args.live_pause)

    return elapsed, lengths, fig_path


def main() -> None:
    args = parse_args()
    header = [
        "timestamp",
        "run_idx",
        "agents",
        "targets",
        "width",
        "height",
        "episodes",
        "max_steps",
        "goal_bias",
        "total_time_s",
        "lengths",
    ]
    maybe_write_header(args.out_csv, header)

    session_ts = time.strftime("%Y%m%d_%H%M%S")
    all_times: List[float] = []

    for run_idx in range(1, args.runs + 1):
        print(f"=== Run {run_idx}/{args.runs} (agents={args.agents}, targets={args.targets}) ===")
        elapsed, lengths, fig_path = run_single(run_idx, args, session_ts)
        all_times.append(elapsed)
        print(f"[run {run_idx}] Path lengths: {lengths}")
        print(f"[run {run_idx}] Total time (s): {round(elapsed, 4)}")
        print(f"[run {run_idx}] Saved figure: {fig_path}")

        with args.out_csv.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                run_idx,
                args.agents,
                args.targets,
                args.width,
                args.height,
                args.episodes,
                args.max_steps,
                args.goal_bias,
                round(elapsed, 4),
                "|".join(str(x) for x in lengths),
            ])

    avg_time = sum(all_times) / len(all_times)
    print(f"Finished {args.runs} runs. Avg time (s): {round(avg_time, 4)}")


if __name__ == "__main__":
    main()
