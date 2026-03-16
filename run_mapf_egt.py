"""Train MAPF-EGT, log metrics, save timestamped figures, and replay the actual episode trajectories live."""
import argparse
import csv
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from mapf_egt import (
    MAPFEGTConfig,
    MAPFEGTTrainer,
    MultiAgentGridEnv,
    visualize_trajectories,
)

INTERACTIVE_BACKENDS = ['TkAgg', 'Qt5Agg', 'QtAgg', 'WXAgg', 'GTK3Agg']


def ensure_interactive_backend() -> None:
    current = plt.get_backend().lower()
    if all(key not in current for key in ['agg', 'pdf', 'svg']):
        return
    for candidate in INTERACTIVE_BACKENDS:
        try:
            plt.switch_backend(candidate)
            return
        except Exception:
            continue
    warnings.warn('Could not switch to an interactive matplotlib backend; live replay may not display.', RuntimeWarning)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--agents', type=int, default=5, help='number of agents')#定义agent数量
    p.add_argument('--width', type=int, default=100)  # paper: grid width
    p.add_argument('--height', type=int, default=100)  # paper: grid height
    p.add_argument('--episodes', type=int, default=300)  # paper: total episodes
    p.add_argument('--max-steps', type=int, default=300)  # paper: max steps per episode T
    p.add_argument('--goal-bias', type=float, default=0.5)  # implementation: optional target bias
    p.add_argument('--obstacle-density', type=float, default=None)  # None -> use env default
    p.add_argument('--targets', type=int, default=3, help='number of target cells')  # paper: |Φ|
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out-csv', type=Path, default=Path('mapf_egt_runs.csv'))
    p.add_argument('--fig-dir', type=Path, default=Path('mapf_egt_figures'))
    p.add_argument('--live-pause', type=float, default=0.05, help='seconds between frames for live replay')
    p.add_argument('--no-replay', action='store_false', dest='replay', help='skip live replay (train time only)')
    p.set_defaults(replay=True)  # 默认开启回放，但不计入总时间
    return p.parse_args()


def maybe_write_header(path: Path, header: List[str]) -> None:
    if not path.exists():
        path.write_text(','.join(header) + '\n', encoding='utf-8')


def compact_path(traj: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Remove consecutive duplicates and leading stationary prefix."""
    if not traj:
        return []
    compact: List[Tuple[int, int]] = []
    for pos in traj:
        pos = tuple(pos)
        if not compact or compact[-1] != pos:
            compact.append(pos)
    # drop leading repeats (if agent stayed in place at start)
    while len(compact) >= 2 and compact[0] == compact[1]:
        compact.pop(0)
    return compact


def simplify_path(traj: List[Tuple[int, int]], loop_window: int = 12) -> List[Tuple[int, int]]:
    """
    Further prune short back-and-forth loops so early dithering is hidden in the replay/figure.
    loop_window controls the maximum loop length (in steps) that will be collapsed.
    """
    compact = compact_path(traj)
    if len(compact) <= 2:
        return compact

    path: List[Tuple[int, int]] = []
    last_index: Dict[Tuple[int, int], int] = {}
    for pt in compact:
        if pt in last_index:
            prev_idx = last_index[pt]
            loop_len = len(path) - prev_idx
            if loop_len <= loop_window:
                # Remove the small loop and keep the earlier occurrence.
                for dropped in path[prev_idx + 1:]:
                    last_index.pop(dropped, None)
                path = path[: prev_idx + 1]
                continue
        path.append(pt)

        last_index[pt] = len(path) - 1
    return path


def animate_recorded_paths(env: MultiAgentGridEnv, paths: List[List[Tuple[int, int]]], pause: float = 0.02) -> None:
    """Replay recorded trajectories quickly; consecutive duplicates are skipped."""
    ensure_interactive_backend()
    plt.ion()
    cleaned = [simplify_path(traj) for traj in paths]
    max_len = max(len(p) for p in cleaned)
    stride = max(1, max_len // 200)  # 最多200帧快速播放
    positions = [p[0] for p in cleaned]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(-0.5, env.height - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('MAPF-EGT live replay (recorded episode)')
    if env.obstacles:
        ox, oy = zip(*env.obstacles)
        ax.scatter(ox, oy, c='red', marker='s', s=20, label='obstacle')
    if env.targets:
        tx, ty = zip(*env.targets)
        ax.scatter(tx, ty, c='green', marker='s', s=40, label='target')
    scat = ax.scatter([p[0] for p in positions], [p[1] for p in positions], c='blue', marker='o', s=60, label='agent')
    lines = []
    for pos in positions:
        line, = ax.plot([pos[0]], [pos[1]], c='royalblue', linewidth=1)
        lines.append(line)
    ax.legend(loc='upper right')
    fig.canvas.draw_idle()
    plt.show(block=False)

    xs = [[pos[0]] for pos in positions]
    ys = [[pos[1]] for pos in positions]

    for step in range(1, max_len, stride):
        for idx, traj in enumerate(cleaned):
            j = min(step, len(traj) - 1)
            x, y = traj[j]
            xs[idx].append(x)
            ys[idx].append(y)
        scat.set_offsets([(xs[i][-1], ys[i][-1]) for i in range(len(xs))])
        for i, line in enumerate(lines):
            line.set_data(xs[i], ys[i])
        fig.canvas.draw_idle()
        plt.pause(pause)
    for _ in range(300):
        if not plt.fignum_exists(fig.number):
            break
        plt.pause(0.1)
    plt.close(fig)
    plt.ioff()


def main() -> None:
    args = parse_args()
    env_kwargs = dict(
        width=args.width,
        height=args.height,
        n_agents=args.agents,
        num_targets=args.targets,
        seed=args.seed,
    )
    if args.obstacle_density is not None:
        env_kwargs["obstacle_density"] = args.obstacle_density
    env = MultiAgentGridEnv(**env_kwargs)
    # swapped-in tuned config (visual time excluded from total_time)
    config = MAPFEGTConfig(
        theta=0.25,  # more lenient good-threshold (0.20-0.25)
        v=2.0,      # tighter bad-threshold (1.8-2.0)
        U=1.5,      # reward coeff (>1, e.g., 1.2-1.5)
        alpha=1,  # stronger positive update (1.0-1.2)
        beta=1,   # softer penalty (0.2-0.5)
        eta=0.05,   # lower smoothing noise (0.02-0.05)
        goal_bias=args.goal_bias,  # recommended 0.6-0.8 (default 0.7)
        total_episodes=args.episodes,
        max_steps=args.max_steps,
    )
    trainer = MAPFEGTTrainer(env, config)
    t0 = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - t0
    lengths = [len(path) - 1 for path in trainer.last_episode_paths]
    print('Path lengths per agent:', lengths)
    print('Total time (s):', round(elapsed, 4))

    header = ['timestamp', 'agents', 'width', 'height', 'episodes', 'max_steps', 'goal_bias', 'total_time_s', 'lengths']
    maybe_write_header(args.out_csv, header)
    with args.out_csv.open('a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            time.strftime('%Y-%m-%d %H:%M:%S'),
            args.agents,
            args.width,
            args.height,
            args.episodes,
            args.max_steps,
            args.goal_bias,
            round(elapsed, 4),
            '|'.join(str(x) for x in lengths),
        ])

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    fig_path = args.fig_dir / f'mapf_egt_traces_{ts}.png'
    cleaned_paths = [simplify_path(traj) for traj in trainer.last_episode_paths]
    visualize_trajectories(env, cleaned_paths, save_path=str(fig_path))
    print('Saved figure to', fig_path)

    if args.replay:
        print('Opening live replay window for recorded episode...')
        animate_recorded_paths(env, trainer.last_episode_paths, pause=args.live_pause)


if __name__ == '__main__':
    main()
