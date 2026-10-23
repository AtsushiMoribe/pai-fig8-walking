"""
Plot the convoy run saved by fig8_multi_eval.py (trajectories, gaps, distance between robots, speed commands).

python plot_multi.py --out-dir outputs_multi
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["tab:blue", "tab:green", "tab:pink", "tab:purple", "tab:brown", "tab:cyan"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="outputs_multi")
    args = parser.parse_args()

    d = np.load(os.path.join(args.out_dir, "trajectory.npz"))
    with open(os.path.join(args.out_dir, "metrics.json")) as f:
        m = json.load(f)
    path, pos, t, n = d["path"], d["pos"], d["t"], m["num_robots"]
    names = ["leader"] + [f"follower {i}" for i in range(1, n)]

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(*np.vstack((path, path[:1])).T, "--", color="tab:orange", lw=2, label="reference")
    for i in range(n):
        ax.plot(pos[:, i, 0], pos[:, i, 1], color=COLORS[i % len(COLORS)], lw=1, alpha=0.8, label=names[i])
        ax.plot(*pos[0, i, :2], "o", color=COLORS[i % len(COLORS)])
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    title = f"{n} robots in a line: lap {m['lap_length_m']} m, {m['cruise_speed_mps']} m/s"
    if m.get("crossing_yield") and sum(m.get("stops_per_robot", [])):
        title += f"\n{sum(m['stops_per_robot'])} stops, min distance {m['min_distance_between_robots_m']} m"
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=n + 1, fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(3, 2, 2)
    gaps = d["progress"][:, :-1] - d["progress"][:, 1:]
    for i in range(n - 1):
        ax.plot(t, gaps[:, i], color=COLORS[(i + 1) % len(COLORS)], label=f"{names[i]} - {names[i + 1]}")
    ax.axhline(float(d["gap"]), color="k", ls="--", lw=1, label="target")
    ax.set_ylabel("gap along\npath [m]")
    ax.legend(fontsize=7, loc="lower right", ncol=3)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(3, 2, 4)
    ax.plot(t, d["min_dist"], color="tab:red", label="closest pair")
    ax.axhline(0.7, color="k", ls=":", lw=1, label="body length (0.7 m)")
    ax.set_ylabel("min distance\n[m]")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(3, 2, 6)
    for i in range(n):
        ax.plot(t, d["cmd"][:, i, 0], color=COLORS[i % len(COLORS)], lw=1, label=names[i])
        if "state" in d:
            # shade the time each robot is stopped (one thin band per robot under the axis)
            stopped = d["state"][:, i] == 2
            ax.fill_between(t, -0.04 * (i + 1), -0.04 * i, where=stopped, color=COLORS[i % len(COLORS)], step="mid")
    ax.set_ylabel("speed\ncommand [m/s]\n(bands: stopped)")
    ax.set_xlabel("time [s]")
    ax.legend(fontsize=7, loc="lower right", ncol=n)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(args.out_dir, "fig8_multi_result.png")
    fig.savefig(out, dpi=120)
    print("saved", out)


if __name__ == "__main__":
    main()
