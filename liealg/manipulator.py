"""Toy 3-DoF (RRR) spatial manipulator with real kinematics *and* dynamics, built entirely on SE(3).

Geometry (home configuration, arm stretched along +x):
    joint 1: revolute about z at the base                    (yaw)
    joint 2: revolute about y at height L0                   (shoulder)
    joint 3: revolute about y at (L1, 0, L0)                 (elbow)
    end-effector at (L1 + L2, 0, L0)

Kinematics : product of exponentials (space form), space/body Jacobians, Lie-group IK.
Dynamics   : recursive Newton-Euler on SE(3) (Lynch & Park, Alg. 8.3) -> inverse dynamics,
             mass matrix, forward dynamics and a simple RK4 simulator.
"""
import numpy as np
from . import se3, so3


def rod_inertia(m, length, radius, axis):
    """Inertia of a solid cylinder of mass m about its COM, long axis = 'x'|'y'|'z'."""
    Ia = 0.5 * m * radius**2                         # about the long axis
    It = m * (length**2 / 12.0 + radius**2 / 4.0)    # transverse
    d = {"x": [Ia, It, It], "y": [It, Ia, It], "z": [It, It, Ia]}[axis]
    return np.diag(d)


class ToyArm3:
    def __init__(self, L0=0.4, L1=0.5, L2=0.4, masses=(2.0, 1.5, 1.0), radius=0.03, g=9.81):
        self.L = np.array([L0, L1, L2])
        self.n = 3
        self.g = g
        # --- kinematics: space-frame screw axes and home pose -------------------------------
        self.S = [se3.screw_axis([0, 0, 1], [0, 0, 0]),
                  se3.screw_axis([0, 1, 0], [0, 0, L0]),
                  se3.screw_axis([0, 1, 0], [L1, 0, L0])]
        self.M = se3.make(np.eye(3), [L1 + L2, 0, L0])
        # --- dynamics: link COM frames at home, spatial inertias ----------------------------
        self.Mlist = [se3.make(np.eye(3), [0, 0, L0 / 2]),
                      se3.make(np.eye(3), [L1 / 2, 0, L0]),
                      se3.make(np.eye(3), [L1 + L2 / 2, 0, L0]),
                      self.M]                                   # last = end-effector frame
        Ib = [rod_inertia(masses[0], L0, radius, "z"),
              rod_inertia(masses[1], L1, radius, "x"),
              rod_inertia(masses[2], L2, radius, "x")]
        self.G = []
        for m, I in zip(masses, Ib):
            Gi = np.zeros((6, 6)); Gi[:3, :3] = I; Gi[3:, 3:] = m * np.eye(3)
            self.G.append(Gi)
        self.masses = np.array(masses)
        # body-frame screw axes A_i = Ad_{M_i^-1} S_i
        self.A = [se3.adjoint(se3.inverse(self.Mlist[i])) @ self.S[i] for i in range(3)]
        # relative home transforms M_{i-1,i}
        self.Mrel = [self.Mlist[0]] + [se3.inverse(self.Mlist[i - 1]) @ self.Mlist[i] for i in range(1, 4)]

    # ---------------- kinematics ----------------
    def fk(self, q):
        return se3.poe(self.S, q, self.M)

    def link_poses(self, q):
        """World poses of the three link COM frames (useful for plotting / inertia checks)."""
        out, T = [], np.eye(4)
        for i in range(3):
            T = T @ se3.exp(self.S[i] * q[i])
            out.append(T @ self.Mlist[i])
        return out

    def joint_points(self, q):
        """3D points base -> joint2 -> joint3 -> EE, for drawing the stick figure."""
        T = np.eye(4); pts = [np.zeros(3)]
        T = T @ se3.exp(self.S[0] * q[0]); pts.append((T @ [0, 0, self.L[0], 1])[:3])
        T = T @ se3.exp(self.S[1] * q[1]); pts.append((T @ [self.L[1], 0, self.L[0], 1])[:3])
        T = T @ se3.exp(self.S[2] * q[2]); pts.append((T @ self.M)[:3, 3])
        return np.array(pts)

    def jac_space(self, q):
        return se3.space_jacobian(self.S, q)

    def jac_body(self, q):
        return se3.body_jacobian(self.S, q, self.M)

    def ik(self, T_des, q0=None, **kw):
        q0 = np.zeros(3) if q0 is None else q0
        # 3 DoF cannot reach an arbitrary 6-D pose: minimise the full twist error in the
        # least-squares sense (damped), which converges to the closest reachable pose.
        kw.setdefault("damping", 1e-3)
        return se3.ik_newton(self.fk, self.jac_body, T_des, q0, **kw)

    def ik_position(self, p_des, q0=None, iters=100, tol=1e-10):
        """Position-only IK (3 equations, 3 unknowns) with the linear part of the space Jacobian."""
        q = np.zeros(3) if q0 is None else np.array(q0, float)
        for _ in range(iters):
            p = self.fk(q)[:3, 3]
            e = p_des - p
            if np.linalg.norm(e) < tol:
                return q, True
            Js = self.jac_space(q)
            Jp = Js[3:] - so3.hat(p) @ Js[:3]     # velocity of the EE point = v + w x p
            q = q + np.linalg.lstsq(Jp, e, rcond=None)[0]
        return q, False

    # ---------------- dynamics (recursive Newton-Euler on SE(3)) ----------------
    def inverse_dynamics(self, q, dq, ddq, gravity=True, F_tip=np.zeros(6)):
        """tau = M(q) ddq + c(q,dq) + g(q) + J^T F_tip."""
        n = self.n
        V = [np.zeros(6)]
        Vd = [np.array([0, 0, 0, 0, 0, self.g if gravity else 0.0])]   # -(-g z) : base 'accelerates up'
        T_rel = []
        for i in range(n):
            Ti = se3.exp(-self.A[i] * q[i]) @ se3.inverse(self.Mrel[i])   # T_{i,i-1}
            T_rel.append(Ti)
            Ad = se3.adjoint(Ti)
            Vi = Ad @ V[-1] + self.A[i] * dq[i]
            Vdi = Ad @ Vd[-1] + se3.ad(Vi) @ self.A[i] * dq[i] + self.A[i] * ddq[i]
            V.append(Vi); Vd.append(Vdi)
        tau = np.zeros(n)
        F = F_tip.copy()
        T_next = se3.inverse(self.Mrel[n])            # T_{n+1,n}
        for i in reversed(range(n)):
            Vi, Vdi = V[i + 1], Vd[i + 1]
            F = se3.adjoint(T_next).T @ F + self.G[i] @ Vdi - se3.ad(Vi).T @ (self.G[i] @ Vi)
            tau[i] = F @ self.A[i]
            T_next = T_rel[i]
        return tau

    def mass_matrix(self, q):
        Mq = np.zeros((self.n, self.n))
        for i in range(self.n):
            e = np.zeros(self.n); e[i] = 1.0
            Mq[:, i] = self.inverse_dynamics(q, np.zeros(self.n), e, gravity=False)
        return Mq

    def bias(self, q, dq):
        """Coriolis + gravity torques h(q, dq)."""
        return self.inverse_dynamics(q, dq, np.zeros(self.n))

    def forward_dynamics(self, q, dq, tau):
        return np.linalg.solve(self.mass_matrix(q), tau - self.bias(q, dq))

    def step(self, q, dq, tau, dt):
        """One RK4 step of the forward dynamics."""
        def f(x):
            qq, dd = x[:3], x[3:]
            return np.concatenate([dd, self.forward_dynamics(qq, dd, tau)])
        x = np.concatenate([q, dq])
        k1 = f(x); k2 = f(x + 0.5 * dt * k1); k3 = f(x + 0.5 * dt * k2); k4 = f(x + dt * k3)
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x[:3], x[3:]

    def computed_torque(self, q, dq, q_des, dq_des=None, ddq_des=None, kp=100.0, kd=20.0):
        """Computed-torque (inverse-dynamics) control:
        tau = M(q)(ddq_des + kp (q_des - q) + kd (dq_des - dq)) + h(q, dq)."""
        dq_des = np.zeros(self.n) if dq_des is None else dq_des
        ddq_des = np.zeros(self.n) if ddq_des is None else ddq_des
        a = ddq_des + kp * (q_des - q) + kd * (dq_des - dq)
        return self.mass_matrix(q) @ a + self.bias(q, dq)
