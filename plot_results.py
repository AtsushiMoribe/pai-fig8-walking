"""
Plot the figure-eight run saved by fig8_eval.py (trajectory, commands, tracking error).

python plot_results.py --out-dir outputs
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    d = np.load(os.path.join(args.out_dir, "trajectory.npz"))
    with open(os.path.join(args.out_dir, "metrics.json")) as f:
        metrics = json.load(f)
    path, pos, t = d["path"], d["pos"], d["t"]

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(*np.vstack((path, path[:1])).T, "--", color="tab:orange", lw=2, label="reference (figure-eight)")
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=t, s=2, cmap="viridis", label="robot base")
    ax.plot(*pos[0, :2], "ko", label="start")
    fig.colorbar(sc, ax=ax, label="time [s]", shrink=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        f"Trajectory: {metrics['laps']} laps, cross-track RMS {metrics['cross_track_error_rms_m'] * 100:.1f} cm"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(3, 2, 2)
    ax.plot(t, d["lin_vel"][:, 0], color="tab:orange", alpha=0.6, label="actual vx")
    ax.plot(t, d["cmd"][:, 0], color="k", lw=1.5, label="command vx")
    ax.set_ylabel("[m/s]")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(3, 2, 4)
    ax.plot(t, d["ang_vel"][:, 2], color="tab:orange", alpha=0.6, label="actual yaw rate")
    ax.plot(t, d["cmd"][:, 2], color="k", lw=1.5, label="command yaw rate")
    ax.set_ylabel("[rad/s]")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(3, 2, 6)
    ax.plot(t, d["cte"] * 100, color="tab:red")
    ax.set_ylabel("cross-track\nerror [cm]")
    ax.set_xlabel("time [s]")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(args.out_dir, "fig8_result.png")
    fig.savefig(out, dpi=120)
    print("saved", out)


if __name__ == "__main__":
    main()
