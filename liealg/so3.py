"""SO(3): rotation matrices, the Lie algebra so(3), exp/log maps, geodesic distance and SLERP.

Conventions follow Lynch & Park, *Modern Robotics* (2017) and Solà et al. (2018).
"""
import numpy as np

EPS = 1e-9


def hat(w):
    """so(3) hat map: R^3 -> 3x3 skew-symmetric matrix [w]."""
    w = np.asarray(w, dtype=float)
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def vee(W):
    """Inverse of hat: 3x3 skew-symmetric -> R^3."""
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


def exp(w):
    """Exponential map so(3) -> SO(3) (Rodrigues' formula). `w` is a rotation vector (axis * angle)."""
    w = np.asarray(w, dtype=float)
    th = np.linalg.norm(w)
    if th < EPS:
        return np.eye(3) + hat(w)          # first-order approximation
    K = hat(w / th)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * K @ K


def log(R):
    """Logarithmic map SO(3) -> so(3). Returns the rotation vector w with ||w|| = theta in [0, pi]."""
    R = np.asarray(R, dtype=float)
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(c)
    if th < EPS:
        return np.zeros(3)
    if np.pi - th < 1e-6:                    # theta ~ pi: axis from the diagonal of (R + I)
        A = R + np.eye(3)
        k = np.argmax(np.diag(A))
        axis = A[:, k] / np.linalg.norm(A[:, k])
        return th * axis
    return th / (2.0 * np.sin(th)) * vee(R - R.T)


def distance(R1, R2):
    """Geodesic (bi-invariant) distance on SO(3): the angle of R1^T R2."""
    return np.linalg.norm(log(R1.T @ R2))


def slerp(R1, R2, s):
    """Geodesic interpolation R(s) = R1 exp(s log(R1^T R2)), s in [0, 1]."""
    return R1 @ exp(s * log(R1.T @ R2))


def project(M):
    """Project a near-rotation matrix onto SO(3) (polar decomposition via SVD)."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def random(rng=None):
    """Uniformly random rotation (via QR of a Gaussian matrix)."""
    rng = np.random.default_rng(rng)
    return project(rng.standard_normal((3, 3)))


def is_rotation(R, tol=1e-8):
    return np.allclose(R.T @ R, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1.0) < tol
