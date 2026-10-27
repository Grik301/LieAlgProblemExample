"""The banana distribution on the Franka Panda.

Gaussian noise on the 7 joint angles is pushed through the forward kinematics. In Cartesian
coordinates the end-effector cloud is curved ('banana'); in exponential coordinates
xi = log(mu^-1 g) on se(3) it is well described by a Gaussian (Long, Wolfe, Mashner &
Chirikjian, RSS 2013). We also compare the sample covariance with the first-order
propagation Sigma = J_b Sigma_q J_b^T (the Lie-group linearisation).

Output: figures/banana_panda.png
"""
import numpy as np
from _plotstyle import plt, IND, ROSE, NAVY, SLATE, GREEN
from liealg import se3
from liealg.panda import Panda

rng = np.random.default_rng(0)
p = Panda()
q0 = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.8])
sigma_q = np.array([0.6, 0.12, 0.3, 0.12, 0.3, 0.12, 0.3])        # large, to make the banana visible
N = 4000
Q = q0 + sigma_q * rng.standard_normal((N, 7))
mu = p.fk_poe(q0)
poses = np.array([p.fk_poe(q) for q in Q])
P = poses[:, :3, 3]
XI = np.array([se3.log(se3.inverse(mu) @ T) for T in poses])      # exponential coordinates about mu

# first-order (Lie-group) covariance propagation
J = p.jac_body_poe(q0)
Sigma_lin = J @ np.diag(sigma_q**2) @ J.T
Sigma_emp = np.cov(XI.T)
print("||Sigma_emp - Sigma_lin|| / ||Sigma_emp|| =", np.linalg.norm(Sigma_emp - Sigma_lin) / np.linalg.norm(Sigma_emp))

def ellipse(ax, mean, cov, color, n=2, **kw):
    ev, evec = np.linalg.eigh(cov)
    t = np.linspace(0, 2 * np.pi, 100)
    circ = np.stack([np.cos(t), np.sin(t)]) * n
    pts = (evec @ np.diag(np.sqrt(ev)) @ circ).T + mean
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=2, **kw)

fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
ax = axs[0]
ax.scatter(P[:, 0], P[:, 1], s=2, color=IND, alpha=0.25, linewidths=0)
ellipse(ax, P[:, :2].mean(0), np.cov(P[:, :2].T), ROSE, ls="--")
ax.set_title("TCP position (x, y): curved cloud"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
ax = axs[1]
ax.scatter(P[:, 0], P[:, 2], s=2, color=IND, alpha=0.25, linewidths=0)
ellipse(ax, P[:, [0, 2]].mean(0), np.cov(P[:, [0, 2]].T), ROSE, ls="--")
ax.set_title("TCP position (x, z)"); ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]"); ax.set_aspect("equal")
ax = axs[2]
i, j = 3, 4            # translational part of the twist (v_x, v_y)
ax.scatter(XI[:, i], XI[:, j], s=2, color=IND, alpha=0.25, linewidths=0)
ellipse(ax, XI[:, [i, j]].mean(0), Sigma_emp[np.ix_([i, j], [i, j])], GREEN, label="sample 2σ")
ellipse(ax, np.zeros(2), Sigma_lin[np.ix_([i, j], [i, j])], NAVY, ls=":", label="J Σq Jᵀ  2σ")
ax.set_title(r"exponential coords $\log(\mu^{-1}g)$: Gaussian"); ax.set_xlabel("v_x"); ax.set_ylabel("v_y"); ax.set_aspect("equal"); ax.legend(fontsize=8)
fig.suptitle("Banana distribution on the Panda: Cartesian vs. Lie-algebra coordinates")
fig.tight_layout(); fig.savefig("figures/banana_panda.png", dpi=160)
print("saved figures/banana_panda.png")
