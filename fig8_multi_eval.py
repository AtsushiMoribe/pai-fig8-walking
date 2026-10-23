"""
Three (or more) Go2 robots walking a figure-eight in a line (convoy).

- every robot runs the same RL locomotion policy trained by fig8_train.py (no retraining)
- all robots are in ONE scene, so robot-robot contacts are really simulated
- leader      : pure-pursuit on the figure-eight at the cruise speed
- followers   : pure-pursuit on the same path + speed control that keeps a fixed gap
                (measured along the path) to the robot in front
- safety      : a robot slows down / stops when another robot is close in front of it
- crossing    : (--yield, default on) a robot that would meet a robot of the other branch at the crossing
                of the eight stops at a stop line before it, and walks again once the crossing is clear
                (first come, first served)
- planner     : before the run, the figure-eight size (lap length) and the speed are chosen so that
                (1) the sharpest turn stays inside the trained yaw-rate range and
                (2) the head and the tail of the line never meet at the crossing of the eight

# evaluation (3 robots)
python fig8_multi_eval.py -e go2-fig8 --ckpt 500 --num-robots 3
# 5 robots on a figure-eight that is deliberately too short: they stop and wait at the crossing
python fig8_multi_eval.py -e go2-fig8 --ckpt 500 --num-robots 5 --gap 2.0 --no-plan --size 3.05 --speed 0.5
"""

import argparse
import json
import math
import os
import pickle
from importlib import metadata

import numpy as np
import torch
from tensordict import TensorDict

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat

from fig8_eval import figure_eight

LEMNISCATE_LENGTH_PER_A = 5.2441  # lap length of a lemniscate of Bernoulli = 5.2441 * half-width
TRAIL_COLORS = [
    (0.1, 0.4, 1.0, 1.0), (0.1, 0.8, 0.3, 1.0), (0.9, 0.2, 0.6, 1.0), (0.6, 0.3, 0.9, 1.0), (0.0, 0.8, 0.8, 1.0)
]
STATE_COLORS = {"walk": (0.2, 0.9, 0.2, 1.0), "slow": (1.0, 0.8, 0.0, 1.0), "stop": (1.0, 0.1, 0.1, 1.0)}


def crossing_conflicts(num_robots, gap, lap, clearance):
    """Pairs of robots (k apart in the line) that reach the crossing at the same time: separation ~ half a lap."""
    return [k for k in range(1, num_robots) if abs((k * gap) % lap - lap / 2) < clearance]


def plan(num_robots, gap, speed, size, max_yaw_rate, yaw_margin, crossing_clearance, keep_size):
    """Choose the figure-eight half-width a [m] and the cruise speed v [m/s].

    (1) turning : the tightest radius of the lemniscate is a/3, so the yaw rate there is 3v/a.
                  it must stay below yaw_margin * max_yaw_rate (room left for corrections).
    (2) crossing: the line occupies (n-1)*gap of the path. two robots meet at the crossing when their
                  distance along the path is about half a lap, so the whole line must fit in half a lap
                  minus a clearance: (n-1)*gap + clearance <= L/2.
    """
    yaw_ok = yaw_margin * max_yaw_rate
    a_turn = 3.0 * speed / yaw_ok
    a_cross = 2.0 * ((num_robots - 1) * gap + crossing_clearance) / LEMNISCATE_LENGTH_PER_A
    notes = []
    if keep_size:
        a = size
        if a < a_cross:
            raise SystemExit(
                f"--size {size} m is too small for {num_robots} robots with gap {gap} m "
                f"(the line would meet itself at the crossing; need >= {a_cross:.2f} m). Reduce --gap or the robots."
            )
        v_max = yaw_ok * a / 3.0
        if speed > v_max:
            notes.append(f"speed lowered {speed:.2f} -> {v_max:.2f} m/s so that the tightest turn stays within the trained yaw rate")
            speed = v_max
    else:
        a = max(size, a_turn, a_cross)
        if a > size:
            reason = "turning limit" if a == a_turn else "crossing clearance"
            notes.append(f"figure-eight enlarged {size:.2f} -> {a:.2f} m ({reason})")
    lap = LEMNISCATE_LENGTH_PER_A * a
    return a, speed, lap, notes


class Convoy:
    """N Go2 robots in one scene sharing one policy (same observation / action layout as Fig8Env)."""

    def __init__(self, num_robots, env_cfg, obs_cfg, starts, cameras=None):
        self.num_envs = num_robots  # the policy sees the robots as a batch
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg
        self.device = gs.device
        self.dt = 0.02
        self.max_episode_length = 10**9
        self.episode_length_buf = torch.zeros(num_robots, dtype=gs.tc_int, device=gs.device)
        self.obs_scales = obs_cfg["obs_scales"]

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False, max_collision_pairs=40 * num_robots, tolerance=1e-5
            ),
            show_viewer=False,
        )
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        self.robots = [
            self.scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=pos, quat=quat))
            for pos, quat in starts
        ]
        self.cams = [self.scene.add_camera(**kw) for kw in (cameras or [])]
        self.scene.build()

        r0 = self.robots[0]
        self.motors_dof_idx = torch.tensor(
            [r0.get_joint(name).dof_start for name in env_cfg["joint_names"]], dtype=gs.tc_int, device=gs.device
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)
        for r in self.robots:
            r.set_dofs_kp([env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
            r.set_dofs_kv([env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)
            r.set_dofs_position(
                torch.tensor([env_cfg["default_joint_angles"][n] for n in env_cfg["joint_names"]], device=gs.device),
                self.motors_dof_idx,
            )

        n = num_robots
        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in env_cfg["joint_names"]],
            dtype=gs.tc_float, device=gs.device,
        )
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)
        self.commands = torch.zeros((n, 3), dtype=gs.tc_float, device=gs.device)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            dtype=gs.tc_float, device=gs.device,
        )
        self.actions = torch.zeros((n, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)
        self._read_state()
        self._update_observation()

    def _read_state(self):
        pos = torch.cat([r.get_pos().reshape(1, 3) for r in self.robots])
        quat = torch.cat([r.get_quat().reshape(1, 4) for r in self.robots])
        inv = inv_quat(quat)
        self.base_pos, self.base_quat = pos, quat
        self.base_lin_vel = transform_by_quat(torch.cat([r.get_vel().reshape(1, 3) for r in self.robots]), inv)
        self.base_ang_vel = transform_by_quat(torch.cat([r.get_ang().reshape(1, 3) for r in self.robots]), inv)
        self.projected_gravity = transform_by_quat(self.global_gravity.expand(len(self.robots), 3), inv)
        self.dof_pos = torch.cat([r.get_dofs_position(self.motors_dof_idx).reshape(1, -1) for r in self.robots])
        self.dof_vel = torch.cat([r.get_dofs_velocity(self.motors_dof_idx).reshape(1, -1) for r in self.robots])

    def _update_observation(self):
        self.obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],
                self.projected_gravity,
                self.commands * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
                self.dof_vel * self.obs_scales["dof_vel"],
                self.actions,
            ),
            dim=-1,
        )

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def set_commands(self, commands):
        self.commands.copy_(torch.as_tensor(commands, dtype=gs.tc_float, device=gs.device))
        self._update_observation()

    def step(self, actions):
        self.actions = torch.clip(actions, -self.cfg["clip_actions"], self.cfg["clip_actions"])
        exec_actions = self.last_actions if self.cfg.get("simulate_action_latency", True) else self.actions
        target = exec_actions * self.cfg["action_scale"] + self.default_dof_pos
        for i, r in enumerate(self.robots):
            r.control_dofs_position(target[i, self.actions_dof_idx], slice(6, 18))
        self.scene.step()
        self._read_state()
        self._update_observation()
        self.last_actions.copy_(self.actions)
        return self.get_observations(), None, None, {}


class CrossingManager:
    """Right of way at the crossing of the figure-eight.

    The path goes through the crossing twice per lap (two "lanes"). A robot approaching the crossing stops at a
    stop line `stop_dist` before it when a robot of the other lane
    - is already in the crossing (past its stop line, or less than `exit_dist` beyond the crossing), or
    - is further back in the line (higher index) and would be in the crossing at the same time, i.e. the two
      distances to the crossing differ by less than the length of the crossing zone (+ margin).
    It walks again once that has been clear for `resume_steps` steps (hysteresis against stop-and-go).
    Why the robot further back goes first: when the head of the line meets its own tail at the crossing, the
    tail has nobody behind it, so the crossing is free right after it. If the tail waited instead, it would
    wait for the whole line to pass while the head comes round the loop behind it, and the line would lock up.
    While the head waits, the robots queued behind it close their gaps (down to about 1 m) so that the robots
    it waits for have room to clear the crossing.
    """

    def __init__(self, s_crossings, lap, slow_dist=1.0, stop_dist=1.1, exit_dist=1.0, margin=0.3, resume_steps=25):
        self.s_c, self.lap = np.asarray(s_crossings), lap
        self.slow_dist, self.stop_dist, self.exit_dist = slow_dist, stop_dist, exit_dist
        self.window = stop_dist + exit_dist + margin
        self.resume_steps = resume_steps
        self.clear_steps = None
        self.holding = None

    def __call__(self, s_pos):
        """s_pos: position of each robot along the path [m] (mod lap). Returns speed factors and whom each waits for."""
        n = len(s_pos)
        if self.clear_steps is None:
            self.clear_steps = np.full(n, self.resume_steps)
            self.holding = np.zeros(n, dtype=bool)
        ahead = (self.s_c[None, :] - s_pos[:, None]) % self.lap  # distance to each pass of the crossing
        behind = (s_pos[:, None] - self.s_c[None, :]) % self.lap
        lane = np.argmin(ahead, axis=1)  # the pass each robot comes to next
        d_next = ahead[np.arange(n), lane]
        factor, waiting_for = np.ones(n), [None] * n
        for i in range(n):
            if d_next[i] > self.stop_dist + self.slow_dist + 0.5 or d_next[i] < self.stop_dist - 0.05:
                self.clear_steps[i] = self.resume_steps  # far from the crossing, or already committed to it
                self.holding[i] = False
                continue
            other = 1 - lane[i]
            for j in range(n):
                if j == i:
                    continue
                inside = ahead[j, other] < self.stop_dist - 0.05 or behind[j, other] < self.exit_dist
                same_time = abs(ahead[j, other] - d_next[i]) < self.window
                if inside or (j > i and same_time):
                    waiting_for[i] = j
                    break
            self.clear_steps[i] = 0 if waiting_for[i] is not None else self.clear_steps[i] + 1
            if self.clear_steps[i] < self.resume_steps:
                f = (d_next[i] - self.stop_dist) / self.slow_dist
                self.holding[i] |= f < 0.25  # come to a clean stop at the stop line and hold there
                factor[i] = 0.0 if self.holding[i] else min(f, 1.0)
            else:
                self.holding[i] = False
        return factor, waiting_for


class PathTracker:
    """Progress of one robot along a closed path (unwrapped arc length) + pure-pursuit steering."""

    def __init__(self, path, s, start_idx, lookahead, max_yaw_rate):
        self.path, self.s, self.idx = path, s, start_idx
        self.lap = s[-1] + np.linalg.norm(path[0] - path[-1])
        self.lookahead, self.max_yaw_rate = lookahead, max_yaw_rate
        self.ds = self.lap / len(path)
        self.wraps = 0

    def update(self, pos_xy):
        n = len(self.path)
        window = (self.idx + np.arange(-20, 200)) % n
        d = np.linalg.norm(self.path[window] - pos_xy, axis=1)
        new_idx = int(window[int(np.argmin(d))])
        if new_idx < self.idx - n // 2:
            self.wraps += 1
        elif new_idx > self.idx + n // 2:
            self.wraps -= 1
        self.idx = new_idx
        return float(d.min())

    @property
    def progress(self):
        """Unwrapped arc length travelled along the path [m]."""
        return self.wraps * self.lap + self.s[self.idx]

    def steer(self, pos_xy, yaw, speed):
        target = self.path[(self.idx + int(self.lookahead / self.ds)) % len(self.path)]
        alpha = math.atan2(target[1] - pos_xy[1], target[0] - pos_xy[0]) - yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        yaw_rate = float(np.clip(2.0 * speed * math.sin(alpha) / self.lookahead, -self.max_yaw_rate, self.max_yaw_rate))
        return speed * max(0.3, math.cos(alpha)), yaw_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="go2-fig8")
    parser.add_argument("--ckpt", type=int, default=500)
    parser.add_argument("--log-root", type=str, default="logs")
    parser.add_argument("--out-dir", type=str, default="outputs_multi")
    parser.add_argument("--num-robots", type=int, default=3)
    parser.add_argument("--gap", type=float, default=1.5, help="Distance between neighbours along the path [m]")
    parser.add_argument("--speed", type=float, default=0.6, help="Cruise speed of the leader [m/s]")
    parser.add_argument("--size", type=float, default=3.0, help="Requested half-width of the figure-eight [m]")
    parser.add_argument("--keep-size", action="store_true", help="Keep --size and lower the speed instead")
    parser.add_argument("--no-plan", action="store_true", help="Use --size and --speed as given (ablation)")
    parser.add_argument("--max-yaw-rate", type=float, default=1.0, help="Yaw-rate range used in training [rad/s]")
    parser.add_argument("--yaw-margin", type=float, default=0.6, help="Fraction of the yaw-rate range used by the path")
    parser.add_argument("--crossing-clearance", type=float, default=1.5, help="Clearance at the crossing [m]")
    parser.add_argument("--safety-dist", type=float, default=1.0, help="Slow down when a robot is closer in front [m]")
    parser.add_argument("--no-yield", action="store_true", help="Disable the stop-and-wait at the crossing (ablation)")
    parser.add_argument("--no-safety", action="store_true", help="Also disable the slow-down near other robots (ablation)")
    parser.add_argument("--lookahead", type=float, default=0.6)
    parser.add_argument("--laps", type=float, default=3.0, help="Laps of the last robot")
    parser.add_argument("--max-time", type=float, default=170.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--res", type=int, nargs=2, default=(960, 540))
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    n = args.num_robots
    if args.no_plan:
        a, v, lap, notes = args.size, args.speed, LEMNISCATE_LENGTH_PER_A * args.size, ["planner disabled"]
    else:
        a, v, lap, notes = plan(
            n, args.gap, args.speed, args.size, args.max_yaw_rate, args.yaw_margin, args.crossing_clearance,
            args.keep_size,
        )
    print(f"[plan] robots={n} gap={args.gap} m -> half-width {a:.2f} m, lap {lap:.2f} m, speed {v:.2f} m/s")
    for note in notes:
        print("[plan]", note)
    conflicts = crossing_conflicts(n, args.gap, lap, 1.0)
    if conflicts:
        print(f"[plan] WARNING: robots {conflicts} apart in the line reach the crossing at the same time "
              f"(their distance {[k * args.gap for k in conflicts]} m ~ half a lap {lap / 2:.2f} m)")

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")
    log_dir = os.path.join(args.log_root, args.exp_name)
    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

    # path, arc length, and the start pose of each robot (lined up behind the leader along the path)
    path = figure_eight(a)
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))))
    lap_len = s[-1] + np.linalg.norm(path[0] - path[-1])
    starts, start_idx = [], []
    for i in range(n):
        k = int(np.searchsorted(s, (-i * args.gap) % lap_len)) % len(path)
        tangent = path[(k + 1) % len(path)] - path[k]
        yaw = math.atan2(tangent[1], tangent[0])
        starts.append(((path[k, 0], path[k, 1], 0.42), (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))))
        start_idx.append(k)

    cameras = []
    if not args.no_video:
        cameras = [
            dict(res=tuple(args.res), pos=(0.0, -1.3 * a, 1.2 * a), lookat=(0.0, -0.1 * a, 0.0), fov=55, debug=True),
            # with crossing management: a fixed close-up of the crossing. otherwise: follow the middle robot
            dict(res=tuple(args.res), pos=(2.4, -2.4, 1.7), lookat=(0.0, 0.0, 0.2), fov=50, debug=True)
            if not args.no_yield
            else dict(res=tuple(args.res), pos=(a + 2.6, -2.2, 1.5), lookat=(a, 0.0, 0.3), fov=55, debug=True),
        ]
    env = Convoy(n, env_cfg, obs_cfg, starts, cameras)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"), map_location=str(gs.device))
    policy = runner.get_inference_policy(device=gs.device)

    trackers = [PathTracker(path, s, start_idx[i], args.lookahead, args.max_yaw_rate) for i in range(n)]
    for i, t in enumerate(trackers):
        if i > 0 and t.s[t.idx] > 0:
            t.wraps = -1  # followers start behind the leader's start point: progress = -i * gap
    s0 = [t.progress for t in trackers]
    i_c = [len(path) // 4, 3 * len(path) // 4]  # the two passes through the crossing (origin)
    crossing = None if args.no_yield else CrossingManager(s[i_c], lap_len)
    markers = [None] * n
    events, state, stop_step = [], ["walk"] * n, [0] * n

    os.makedirs(args.out_dir, exist_ok=True)
    if cameras:
        env.scene.draw_debug_trajectory(
            np.column_stack((np.concatenate((path[::20], path[:1])), np.full(len(path[::20]) + 1, 0.005))),
            radius=0.015, color=(1.0, 0.5, 0.0, 1.0),
        )
        if args.no_yield:
            env.cams[1].follow_entity(env.robots[n // 2], smoothing=0.9)
        env.cams[0].start_recording(os.path.join(args.out_dir, "overview.mp4"), fps=args.fps)
        env.cams[1].start_recording(os.path.join(args.out_dir, "follow.mp4"), fps=args.fps)

    log = {k: [] for k in ("t", "pos", "cmd", "cte", "progress", "min_dist", "fallen", "state")}
    k_gap, v_max = 0.8, min(0.95, 1.5 * v)
    fallen = np.zeros(n, dtype=bool)
    with torch.no_grad():
        for step in range(int(args.max_time / env.dt)):
            pos = env.base_pos.cpu().numpy()
            yaw = quat_to_xyz(env.base_quat, rpy=True)[:, 2].cpu().numpy()
            cte = np.array([trackers[i].update(pos[i, :2]) for i in range(n)])
            prog = np.array([t.progress for t in trackers])
            fallen |= (env.projected_gravity[:, 2].cpu().numpy() > -0.7) | (pos[:, 2] < 0.15)

            cmd = np.zeros((n, 3))
            if crossing is not None:
                factor, waiting_for = crossing(prog % lap_len)
            else:
                factor, waiting_for = np.ones(n), [None] * n
            new_state = []
            for i in range(n):
                # leader: cruise speed. follower: keep the gap to the robot in front (along the path)
                if i == 0:
                    speed = v
                else:
                    gap_i = prog[i - 1] - prog[i]
                    speed = float(np.clip(v + k_gap * (gap_i - args.gap), 0.0, v_max))
                    speed = max(speed, 0.3) if gap_i > 1.2 else speed  # close up behind a stopped robot
                    speed *= float(np.clip((gap_i - 0.85) / 0.25, 0.0, 1.0))  # but keep about 1 m
                # right of way at the crossing: slow down to the stop line, wait, then walk again
                speed *= factor[i]
                # safety: slow down when any robot is close in front (within +-60 deg of the heading)
                for j in range(n):
                    if j == i:
                        continue
                    rel = pos[j, :2] - pos[i, :2]
                    dist = float(np.linalg.norm(rel))
                    ahead = math.cos(math.atan2(rel[1], rel[0]) - yaw[i]) > 0.5
                    if ahead and dist < args.safety_dist and not args.no_safety:
                        speed *= max(0.0, (dist - 0.6) / (args.safety_dist - 0.6))
                vx, wz = trackers[i].steer(pos[i, :2], yaw[i], speed)
                # hysteresis: once stopped, stay stopped for at least 0.5 s and until the speed is clearly up again
                stopped = speed < 0.05 or (state[i] == "stop" and (speed < 0.12 or step - stop_step[i] < 25))
                cmd[i] = (0.0, 0.0, 0.0) if stopped else (vx, 0.0, wz)
                new_state.append("stop" if stopped else ("slow" if speed < 0.7 * v else "walk"))
                if new_state[i] == "stop" and state[i] != "stop":
                    stop_step[i] = step
                if (new_state[i] == "stop") != (state[i] == "stop"):
                    if new_state[i] == "stop":
                        why = (f"yields to robot {waiting_for[i]} at the crossing" if waiting_for[i] is not None
                               else f"queues behind robot {i - 1}")
                    else:
                        why = "no risk any more"
                    events.append({"t": round(step * env.dt, 2), "robot": i,
                                   "event": "stop" if new_state[i] == "stop" else "resume", "reason": why})
                    print(f"[{step * env.dt:6.2f} s] robot {i} {events[-1]['event']:6s} ({why})")
            state = new_state
            env.set_commands(cmd)
            env.step(policy(env.get_observations()))

            d = [np.linalg.norm(pos[i, :2] - pos[j, :2]) for i in range(n) for j in range(i + 1, n)]
            log["t"].append(step * env.dt)
            log["pos"].append(pos)
            log["cmd"].append(cmd)
            log["cte"].append(cte)
            log["progress"].append(prog)
            log["min_dist"].append(min(d) if d else np.inf)
            log["fallen"].append(fallen.copy())
            log["state"].append([{"walk": 0, "slow": 1, "stop": 2}[x] for x in state])
            if cameras and step % 5 == 0:
                # status lamp above each robot: green = walking, yellow = slowing down, red = stopped
                for i in range(n):
                    if markers[i] is not None:
                        env.scene.clear_debug_object(markers[i])
                    markers[i] = env.scene.draw_debug_sphere(
                        (pos[i, 0], pos[i, 1], pos[i, 2] + 0.35), radius=0.07, color=STATE_COLORS[state[i]]
                    )
            if cameras and step % 25 == 0:
                for i in range(n):
                    env.scene.draw_debug_sphere(
                        (pos[i, 0], pos[i, 1], 0.02), radius=0.03, color=TRAIL_COLORS[i % len(TRAIL_COLORS)]
                    )
            if (prog[-1] - s0[-1]) / lap_len >= args.laps:
                break

    if cameras:
        for cam in env.cams:
            cam.stop_recording()

    log = {k: np.asarray(v_) for k, v_ in log.items()}
    np.savez(os.path.join(args.out_dir, "trajectory.npz"), path=path, gap=args.gap, **log)
    gaps = log["progress"][:, :-1] - log["progress"][:, 1:]
    metrics = {
        "num_robots": n,
        "half_width_m": round(a, 2),
        "lap_length_m": round(lap_len, 2),
        "cruise_speed_mps": round(v, 3),
        "target_gap_m": args.gap,
        "plan_notes": notes,
        "time_s": round(float(log["t"][-1]), 2),
        "laps_last_robot": round(float((log["progress"][-1, -1] - s0[-1]) / lap_len), 2),
        "min_distance_between_robots_m": round(float(log["min_dist"].min()), 3),
        "gap_error_rms_m": round(float(np.sqrt(np.mean((gaps - args.gap) ** 2))), 3),
        "cross_track_error_rms_m": [round(float(x), 3) for x in np.sqrt(np.mean(log["cte"] ** 2, axis=0))],
        "fallen": log["fallen"][-1].tolist(),
        "crossing_yield": crossing is not None,
        "stops_per_robot": [sum(e["robot"] == i and e["event"] == "stop" for e in events) for i in range(n)],
        "stopped_time_s": [round(float((log["state"][:, i] == 2).sum() * env.dt), 2) for i in range(n)],
        "min_distance_near_crossing_m": round(float(min(
            (np.linalg.norm(p[i, :2] - p[j, :2]) for p in log["pos"] for i in range(n) for j in range(i + 1, n)
             if min(np.linalg.norm(p[i, :2]), np.linalg.norm(p[j, :2])) < 1.5), default=np.inf)), 3),
    }
    with open(os.path.join(args.out_dir, "events.json"), "w") as f:
        json.dump(events, f, indent=2)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
