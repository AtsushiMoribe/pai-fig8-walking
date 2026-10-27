"""
Figure-eight walking with a trained Go2 locomotion policy.

Hierarchical control:
  - high level : pure-pursuit path follower on a figure-eight (lemniscate of Bernoulli)
                 -> velocity command [vx, vy, yaw rate]
  - low level  : RL policy (PPO, trained by fig8_train.py) -> 12 joint position targets

The run is recorded headlessly by two cameras (overview + follow) and the trajectory is
saved with tracking metrics, so it works on Google Colab without a display.

# evaluation
python fig8_eval.py -e go2-fig8 --ckpt 500
"""

import argparse
import json
import math
import os
import pickle
from importlib import metadata

import numpy as np
import torch

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from genesis.utils.geom import quat_to_xyz

from fig8_env import Fig8Env


def figure_eight(a, n=4000):
    """Lemniscate of Bernoulli with half-width a [m], starting at (a, 0) heading +y."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    s = np.sin(t)
    d = 1.0 + s**2
    return np.stack((a * np.cos(t) / d, a * s * np.cos(t) / d), axis=-1)


class PurePursuit:
    """Pure-pursuit follower on a closed path that turns a pose into a velocity command."""

    def __init__(self, path, speed, lookahead, max_yaw_rate):
        self.path = path
        self.speed = speed
        self.lookahead = lookahead
        self.max_yaw_rate = max_yaw_rate
        seg = np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1)
        self.ds = seg.mean()
        self.length = seg.sum()
        self.idx = 0  # index of the nearest path point
        self.progress = 0  # number of path points passed (for lap counting)

    def __call__(self, pos_xy, yaw):
        n = len(self.path)
        # search the nearest point only a little ahead of the previous one, so the follower
        # does not jump to the other branch at the crossing of the eight
        window = (self.idx + np.arange(-20, 200)) % n
        d = np.linalg.norm(self.path[window] - pos_xy, axis=1)
        k = int(np.argmin(d))
        new_idx = int(window[k])
        self.progress += (new_idx - self.idx) % n if (new_idx - self.idx) % n < n // 2 else 0
        self.idx = new_idx
        cross_track_error = float(d[k])

        target = self.path[(self.idx + int(self.lookahead / self.ds)) % n]
        dx, dy = target - pos_xy
        # angle of the target in the robot frame
        alpha = math.atan2(dy, dx) - yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        # pure pursuit: curvature = 2 sin(alpha) / L
        yaw_rate = 2.0 * self.speed * math.sin(alpha) / self.lookahead
        yaw_rate = float(np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate))
        # slow down when the heading error is large
        vx = self.speed * max(0.3, math.cos(alpha))
        return np.array([vx, 0.0, yaw_rate]), cross_track_error

    @property
    def laps(self):
        return self.progress / len(self.path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="go2-fig8")
    parser.add_argument("--ckpt", type=int, default=500, help="Checkpoint iteration to load")
    parser.add_argument("--log-root", type=str, default="logs")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--size", type=float, default=3.0, help="Half-width of the figure-eight [m]")
    parser.add_argument("--speed", type=float, default=0.6, help="Walking speed command [m/s]")
    parser.add_argument("--lookahead", type=float, default=0.6, help="Pure-pursuit look-ahead distance [m]")
    parser.add_argument("--max-yaw-rate", type=float, default=1.0, help="Must stay inside the trained range")
    parser.add_argument("--laps", type=float, default=3.0, help="Number of laps to walk")
    parser.add_argument("--max-time", type=float, default=170.0, help="Time limit [s] (video must be <= 3 min)")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--res", type=int, nargs=2, default=(960, 540))
    parser.add_argument("--plain-scene", action="store_true", help="Genesis default checkerboard ground and dark sky")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")

    log_dir = os.path.join(args.log_root, args.exp_name)
    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    # start on the right tip of the eight, heading +y (tangent of the path)
    env_cfg["base_init_pos"] = [args.size, 0.0, 0.42]
    env_cfg["base_init_quat"] = [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
    env_cfg["episode_length_s"] = args.max_time + 10.0

    a = args.size
    cameras = []
    if not args.no_video:
        cameras = [
            # overview camera looking at the whole eight
            # (debug=True so that the reference path and trail drawn with draw_debug_* appear in the video)
            dict(res=tuple(args.res), pos=(0.0, -1.6 * a, 0.95 * a), lookat=(0.0, 0.05 * a, 0.3), fov=55, debug=True),
            # follow camera, moved with the robot
            dict(res=tuple(args.res), pos=(a + 1.8, -1.2, 0.9), lookat=(a, 0.0, 0.3), fov=50, debug=True),
        ]
    env = Fig8Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        cameras=cameras,
        scenery=not args.plain_scene,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"), map_location=str(gs.device))
    policy = runner.get_inference_policy(device=gs.device)

    path = figure_eight(a)
    follower = PurePursuit(path, args.speed, args.lookahead, args.max_yaw_rate)

    os.makedirs(args.out_dir, exist_ok=True)
    if cameras:
        # draw the reference path on the ground
        env.scene.draw_debug_trajectory(
            np.column_stack((np.concatenate((path[::20], path[:1])), np.full(len(path[::20]) + 1, 0.005))),
            radius=0.015,
            color=(1.0, 0.5, 0.0, 1.0),
        )
        env.cams[1].follow_entity(env.robot, smoothing=0.9)
        env.cams[0].start_recording(os.path.join(args.out_dir, "overview.mp4"), fps=args.fps)
        env.cams[1].start_recording(os.path.join(args.out_dir, "follow.mp4"), fps=args.fps)

    command = np.zeros(3)
    env.set_commands(command)
    obs = env.reset()
    log = {"t": [], "pos": [], "yaw": [], "cmd": [], "lin_vel": [], "ang_vel": [], "cte": []}
    n_falls = 0
    max_steps = int(args.max_time / env.dt)
    with torch.no_grad():
        for step in range(max_steps):
            pos = env.base_pos[0].cpu().numpy()
            yaw = float(quat_to_xyz(env.base_quat, rpy=True)[0, 2])
            command, cte = follower(pos[:2], yaw)
            env.set_commands(command)
            obs = env.get_observations()
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if bool(dones[0]) and step < max_steps - 1:
                n_falls += 1
                follower.idx = 0  # the robot is reset to the start point

            log["t"].append(step * env.dt)
            log["pos"].append(pos)
            log["yaw"].append(yaw)
            log["cmd"].append(command)
            log["lin_vel"].append(env.base_lin_vel[0].cpu().numpy())
            log["ang_vel"].append(env.base_ang_vel[0].cpu().numpy())
            log["cte"].append(cte)
            if cameras and step % 25 == 0:
                # leave a trail of the robot's actual path
                env.scene.draw_debug_sphere((pos[0], pos[1], 0.02), radius=0.03, color=(0.1, 0.4, 1.0, 1.0))
            if follower.laps >= args.laps:
                break

    if cameras:
        for cam in env.cams:
            cam.stop_recording()

    log = {k: np.asarray(v) for k, v in log.items()}
    np.savez(os.path.join(args.out_dir, "trajectory.npz"), path=path, **log)
    cte = log["cte"]
    metrics = {
        "time_s": float(log["t"][-1]),
        "laps": round(follower.laps, 2),
        "path_length_m": round(follower.length, 2),
        "mean_speed_mps": round(float(np.linalg.norm(log["lin_vel"][:, :2], axis=1).mean()), 3),
        "cross_track_error_rms_m": round(float(np.sqrt(np.mean(cte**2))), 3),
        "cross_track_error_max_m": round(float(cte.max()), 3),
        "yaw_rate_tracking_rmse": round(float(np.sqrt(np.mean((log["cmd"][:, 2] - log["ang_vel"][:, 2]) ** 2))), 3),
        "falls": n_falls,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
