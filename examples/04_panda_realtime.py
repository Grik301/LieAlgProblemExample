"""Real-time MuJoCo simulation of the 7-DoF Franka Panda tracking SE(3) geodesics.

The end-effector target moves along screw-motion geodesics between a set of waypoint poses
(T(s) = T_a exp(s log(T_a^-1 T_b))). A resolved-rate controller on SE(3) turns the body-twist
error log(T^-1 T_des) into joint velocities through the damped pseudo-inverse of the body
Jacobian and feeds MuJoCo's position actuators; MuJoCo integrates the full rigid-body dynamics.

    python examples/04_panda_realtime.py              # interactive viewer, real-time
    python examples/04_panda_realtime.py --headless   # no window; logs + figures/panda_tracking.png
    python examples/04_panda_realtime.py --video out.mp4   # offscreen render (needs EGL/OSMesa + imageio)
"""
import argparse, time
import numpy as np
import mujoco
from liealg import se3, so3
from liealg.panda import Panda

ap = argparse.ArgumentParser()
ap.add_argument("--headless", action="store_true")
ap.add_argument("--duration", type=float, default=20.0, help="seconds (headless/video)")
ap.add_argument("--segment", type=float, default=3.0, help="seconds per geodesic segment")
ap.add_argument("--video", type=str, default=None)
args = ap.parse_args()

p = Panda()
p.reset()
T_home = p.ee()
R_down = T_home[:3, :3]
waypoints = [
    T_home,
    se3.make(R_down @ so3.exp([0.0, 0.0, 0.6]), T_home[:3, 3] + [0.05, 0.25, -0.15]),
    se3.make(R_down @ so3.exp([-0.5, 0.0, 0.0]), T_home[:3, 3] + [0.10, -0.25, -0.10]),
    se3.make(R_down @ so3.exp([0.0, -0.5, 0.0]), T_home[:3, 3] + [-0.15, 0.0, 0.20]),
]
for k, T in enumerate(waypoints):                      # make sure every waypoint is reachable
    _, err, ok = p.ik(T, iters=50, damping=1e-4)
    print(f"waypoint {k}: IK residual {err:.1e}")

def target_at(t):
    seg, s = divmod(t / args.segment, 1.0)
    a = waypoints[int(seg) % len(waypoints)]
    b = waypoints[(int(seg) + 1) % len(waypoints)]
    s = 0.5 - 0.5 * np.cos(np.pi * s)                  # smooth-start/stop time scaling
    return se3.interpolate(a, b, s)

log_t, log_e = [], []
ctrl_dt = p.model.opt.timestep * p.ctrl_substeps

def control_step():
    T_des = target_at(p.data.time)
    p.set_target_marker(T_des)
    e = p.resolved_rate(T_des, k=10.0, damping=5e-3)
    log_t.append(p.data.time); log_e.append(np.linalg.norm(e))

if args.video:
    import imageio
    renderer = mujoco.Renderer(p.model, 480, 640)
    cam = mujoco.MjvCamera(); cam.azimuth, cam.elevation, cam.distance, cam.lookat = 135, -20, 2.0, [0.3, 0, 0.4]
    frames = []
    while p.data.time < args.duration:
        control_step(); p.step()
        if len(log_t) % 3 == 0:                            # ~33 fps
            renderer.update_scene(p.data, camera=cam); frames.append(renderer.render())
    imageio.mimsave(args.video, frames, fps=33)
    print("wrote", args.video)
elif args.headless:
    wall = time.time()
    while p.data.time < args.duration:
        control_step(); p.step()
    print(f"simulated {p.data.time:.1f} s in {time.time() - wall:.2f} s wall-clock "
          f"({p.data.time / (time.time() - wall):.0f}x real time)")
else:
    import mujoco.viewer
    with mujoco.viewer.launch_passive(p.model, p.data) as viewer:
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 135, -20, 2.0
        viewer.cam.lookat[:] = [0.3, 0, 0.4]
        t0 = time.time()
        while viewer.is_running():
            control_step(); p.step()
            viewer.sync()
            lag = p.data.time - (time.time() - t0)            # real-time sync
            if lag > 0:
                time.sleep(lag)

log_t, log_e = np.array(log_t), np.array(log_e)
print(f"mean ||log(T^-1 T_des)|| after 1 s: {log_e[log_t > 1.0].mean():.4f}   max: {log_e[log_t > 1.0].max():.4f}")
try:
    from _plotstyle import plt, IND, ROSE
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(log_t, log_e, color=IND)
    for k in range(1, int(log_t[-1] / args.segment) + 1):
        ax.axvline(k * args.segment, color=ROSE, lw=0.8, ls=":")
    ax.set_xlabel("time [s]"); ax.set_ylabel(r"$\|\log(T^{-1}T_{des})\|$"); ax.set_title("Panda: SE(3) geodesic tracking error (MuJoCo)")
    fig.tight_layout(); fig.savefig("figures/panda_tracking.png", dpi=160); print("saved figures/panda_tracking.png")
except Exception as ex:
    print("plot skipped:", ex)
