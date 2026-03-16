from pathlib import Path
path = Path('run_mapf_egt.py')
text = path.read_text()
old = "    p.add_argument(\"--obstacle-density\", type=float, default=0.08)\n    p.add_argument(\"--seed\", type=int, default=42)\n    p.add_argument(\"--out-csv\", type=Path, default=Path(\"mapf_egt_runs.csv\"))\n    p.add_argument(\"--figure\", type=Path, default=Path(\"mapf_egt_traces.png\"))\n    return p.parse_args()\n"
new = "    p.add_argument(\"--obstacle-density\", type=float, default=0.08)\n    p.add_argument(\"--seed\", type=int, default=42)\n    p.add_argument(\"--out-csv\", type=Path, default=Path(\"mapf_egt_runs.csv\"))\n    p.add_argument(\"--fig-dir\", type=Path, default=Path(\"mapf_egt_figures\"))\n    return p.parse_args()\n"
if old not in text:
    raise SystemExit('arg block not found')
text = text.replace(old, new, 1)
old_fig_call = "    visualize_trajectories(env, trainer.last_episode_paths, save_path=str(args.figure))\n"
replacement = "    args.fig_dir.mkdir(parents=True, exist_ok=True)\n    ts = time.strftime(\"%Y%m%d_%H%M%S\")\n    fig_path = args.fig_dir / f\"mapf_egt_traces_{ts}.png\"\n    visualize_trajectories(env, trainer.last_episode_paths, save_path=str(fig_path))\n    print(\"Saved figure to\", fig_path)\n"
if old_fig_call not in text:
    raise SystemExit('figure call not found')
text = text.replace(old_fig_call, replacement, 1)
path.write_text(text)
