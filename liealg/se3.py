"""SE(3): homogeneous transforms, twists, exp/log, adjoints and screw interpolation.

A twist is the 6-vector xi = (omega, v). hat(xi) is the 4x4 matrix [[ [omega], v ], [0, 0]].
"""
import numpy as np
from . import so3

EPS = 1e-9


def hat(xi):
    xi = np.asarray(xi, dtype=float)
    X = np.zeros((4, 4))
    X[:3, :3] = so3.hat(xi[:3])
    X[:3, 3] = xi[3:]
    return X


def vee(X):
    return np.concatenate([so3.vee(X[:3, :3]), X[:3, 3]])


def make(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def inverse(T):
    R, p = T[:3, :3], T[:3, 3]
    return make(R.T, -R.T @ p)


def exp(xi):
    """Exponential map se(3) -> SE(3) for twist xi = (omega, v) (angle folded into omega)."""
    xi = np.asarray(xi, dtype=float)
    w, v = xi[:3], xi[3:]
    th = np.linalg.norm(w)
    if th < EPS:
        return make(np.eye(3), v)
    k = w / th
    K = so3.hat(k)
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    V = np.eye(3) * th + (1 - np.cos(th)) * K + (th - np.sin(th)) * K @ K      # G(theta)
    return make(R, V @ (v / th))


def log(T):
    """Logarithmic map SE(3) -> se(3). Returns twist xi = (omega, v) with exp(xi) = T."""
    R, p = T[:3, :3], T[:3, 3]
    w = so3.log(R)
    th = np.linalg.norm(w)
    if th < EPS:
        return np.concatenate([np.zeros(3), p])
    K = so3.hat(w / th)
    Ginv = np.eye(3) / th - 0.5 * K + (1.0 / th - 0.5 / np.tan(th / 2.0)) * K @ K
    return np.concatenate([w, th * (Ginv @ p)])


def adjoint(T):
    """Adjoint Ad_T (6x6): maps twists between frames, xi_a = Ad_{T_ab} xi_b."""
    R, p = T[:3, :3], T[:3, 3]
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[3:, :3] = so3.hat(p) @ R
    A[3:, 3:] = R
    return A


def ad(xi):
    """Lie bracket matrix ad_xi (6x6): [xi1, xi2] = ad_{xi1} xi2."""
    w, v = xi[:3], xi[3:]
    A = np.zeros((6, 6))
    A[:3, :3] = so3.hat(w)
    A[3:, :3] = so3.hat(v)
    A[3:, 3:] = so3.hat(w)
    return A


def distance(T1, T2, w_rot=1.0, w_pos=1.0):
    """Left-invariant distance ||log(T1^-1 T2)|| with optional weighting of rotation / translation."""
    xi = log(inverse(T1) @ T2)
    return np.sqrt(w_rot * xi[:3] @ xi[:3] + w_pos * xi[3:] @ xi[3:])


def interpolate(T1, T2, s):
    """Screw-motion (geodesic) interpolation: T(s) = T1 exp(s log(T1^-1 T2))."""
    return T1 @ exp(s * log(inverse(T1) @ T2))


def screw_axis(omega, q, h=0.0):
    """Screw axis S = (omega, -omega x q + h*omega) for a unit rotation axis omega through point q."""
    omega = np.asarray(omega, float)
    q = np.asarray(q, float)
    return np.concatenate([omega, -np.cross(omega, q) + h * omega])


def poe(screws, q, M):
    """Product of exponentials (space form): T = exp([S1]q1) ... exp([Sn]qn) M."""
    T = np.eye(4)
    for S, qi in zip(screws, q):
        T = T @ exp(S * qi)
    return T @ M


def space_jacobian(screws, q):
    """Space Jacobian J_s(q); column i = Ad_{exp(S1 q1)...exp(S_{i-1} q_{i-1})} S_i."""
    n = len(screws)
    J = np.zeros((6, n))
    T = np.eye(4)
    for i in range(n):
        J[:, i] = adjoint(T) @ screws[i]
        T = T @ exp(screws[i] * q[i])
    return J


def body_jacobian(screws, q, M):
    """Body Jacobian J_b = Ad_{T^-1} J_s."""
    return adjoint(inverse(poe(screws, q, M))) @ space_jacobian(screws, q)


def ik_newton(fk, jac_body, T_des, q0, iters=100, tol=1e-9, damping=1e-6, step=1.0):
    """Lie-group Newton/Gauss-Newton IK: q <- q + J_b^+ log(T(q)^-1 T_des).

    fk(q) -> 4x4, jac_body(q) -> 6xn body Jacobian. Returns (q, err_norm, converged).
    """
    q = np.array(q0, dtype=float)
    err = log(inverse(fk(q)) @ T_des)
    for _ in range(iters):
        if np.linalg.norm(err) < tol:
            return q, np.linalg.norm(err), True
        J = jac_body(q)
        dq = np.linalg.solve(J.T @ J + damping * np.eye(J.shape[1]), J.T @ err)   # damped least squares
        q = q + step * dq
        err = log(inverse(fk(q)) @ T_des)
    return q, np.linalg.norm(err), np.linalg.norm(err) < tol
