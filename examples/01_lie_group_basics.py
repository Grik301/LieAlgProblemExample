"""SO(3)/SE(3) basics: exp/log round trips, geodesic distance, SLERP, screw interpolation."""
import numpy as np
from liealg import so3, se3

rng = np.random.default_rng(0)
w = rng.standard_normal(3)
R = so3.exp(w)
print("R^T R = I :", np.allclose(R.T @ R, np.eye(3)), "  det R =", round(np.linalg.det(R), 6))
print("log(exp(w)) == w :", np.allclose(so3.log(R), w))

# Geodesic vs. Euclidean distance between rotations
R1, R2 = so3.random(rng), so3.random(rng)
print(f"geodesic distance (angle) = {so3.distance(R1, R2):.4f} rad,  Frobenius ||R1-R2|| = {np.linalg.norm(R1-R2):.4f}")

# SLERP is a geodesic: constant angular speed
for s in (0.0, 0.25, 0.5, 0.75, 1.0):
    print(f"  s={s:.2f}  dist(R1, R(s)) = {so3.distance(R1, so3.slerp(R1, R2, s)):.4f}")

# SE(3): a screw motion. Interpolating in the Lie algebra gives a helix, not a straight line.
T1 = se3.make(np.eye(3), [0, 0, 0])
T2 = se3.make(so3.exp([0, 0, np.pi / 2]), [1.0, 0.0, 0.5])
xi = se3.log(se3.inverse(T1) @ T2)
print("twist (w, v) =", np.round(xi, 3), "  pitch =", round(xi[:3] @ xi[3:] / (xi[:3] @ xi[:3]), 3))
pts = np.array([se3.interpolate(T1, T2, s)[:3, 3] for s in np.linspace(0, 1, 5)])
print("screw path samples:\n", np.round(pts, 3))
