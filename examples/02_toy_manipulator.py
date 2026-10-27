"""Toy 3-DoF arm: PoE kinematics, Lie-group IK, Newton-Euler dynamics, computed-torque tracking.

Output: figures/toy_arm.png
"""
import numpy as np
from _plotstyle import plt, IND, ROSE, NAVY, SLATE, GREEN
from liealg import se3, ToyArm3

arm = ToyArm3()
q = np.array([0.4, -0.5, 1.0])
T = arm.fk(q)
print("FK(q):\n", np.round(T, 3))

# ---- IK on SE(3): reach the position of a known pose (3 DoF -> position-only is exact) ----
p_des = arm.fk([0.9, -0.3, 0.8])[:3, 3]
q_ik, ok = arm.ik_position(p_des, q0=q)
print("IK position error:", np.linalg.norm(arm.fk(q_ik)[:3, 3] - p_des), "converged:", ok)

# ---- Dynamics checks ----
Mq = arm.mass_matrix(q)
print("M(q) symmetric:", np.allclose(Mq, Mq.T), " positive definite:", np.all(np.linalg.eigvalsh(Mq) > 0))
# energy drift with zero torque and no gravity should be ~0 (passivity check)
arm_ng = ToyArm3(g=0.0)
qq, dq = q.copy(), np.array([0.5, -0.3, 0.8])
E0 = 0.5 * dq @ arm_ng.mass_matrix(qq) @ dq
for _ in range(400):
    qq, dq = arm_ng.step(qq, dq, np.zeros(3), 2e-3)
E1 = 0.5 * dq @ arm_ng.mass_matrix(qq) @ dq
print(f"kinetic energy drift over 0.8 s (tau=0, g=0): {abs(E1 - E0):.2e} J")

# ---- Computed-torque tracking of a joint-space trajectory that follows an SE(3) geodesic ----
T0, T1 = arm.fk([0.0, -0.2, 0.6]), arm.fk([1.2, -0.8, 1.6])
ts = np.linspace(0, 3.0, 300)
q_traj = []
qg = np.array([0.0, -0.2, 0.6])
for s in ts / ts[-1]:
    Ts = se3.interpolate(T0, T1, s)                        # geodesic in SE(3)
    qg, _ = arm.ik_position(Ts[:3, 3], q0=qg)               # closest joint config (position)
    q_traj.append(qg.copy())
q_traj = np.array(q_traj)
dq_traj = np.gradient(q_traj, ts, axis=0)
ddq_traj = np.gradient(dq_traj, ts, axis=0)

dt = 1e-3
qs, dqs = q_traj[0].copy() + 0.2, np.zeros(3)             # start with an initial error
hist_q, hist_e, hist_p = [], [], []
for k, t in enumerate(np.arange(0, 3.0, dt)):
    i = min(int(t / 3.0 * 299), 299)
    qd = q_traj[i]
    tau = arm.computed_torque(qs, dqs, qd, dq_traj[i], ddq_traj[i])
    qs, dqs = arm.step(qs, dqs, tau, dt)
    if k % 10 == 0:
        hist_q.append(qs.copy()); hist_e.append(np.linalg.norm(qd - qs)); hist_p.append(arm.fk(qs)[:3, 3])
hist_p = np.array(hist_p)

fig = plt.figure(figsize=(11, 4.2))
ax = fig.add_subplot(1, 2, 1, projection="3d")
geo = np.array([se3.interpolate(T0, T1, s)[:3, 3] for s in np.linspace(0, 1, 60)])
ax.plot(*geo.T, color=ROSE, lw=2, label="SE(3) geodesic (desired)")
ax.plot(*hist_p.T, color=IND, lw=1.5, ls="--", label="computed-torque tracking")
for qq, c in [(q_traj[0], SLATE), (q_traj[-1], NAVY)]:
    pts = arm.joint_points(qq); ax.plot(*pts.T, "-o", color=c, lw=3, ms=4)
ax.set_title("Toy RRR arm"); ax.legend(fontsize=8, loc="upper left"); ax.set_box_aspect((1, 1, 0.7))
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(np.arange(len(hist_e)) * dt * 10, hist_e, color=IND)
ax2.set_xlabel("time [s]"); ax2.set_ylabel("||q_des - q||  [rad]"); ax2.set_title("Tracking error (RNEA computed-torque control)"); ax2.set_yscale("log")
fig.tight_layout(); fig.savefig("figures/toy_arm.png", dpi=160)
print("saved figures/toy_arm.png")
