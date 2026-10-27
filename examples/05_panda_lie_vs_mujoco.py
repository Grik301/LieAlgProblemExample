"""Cross-check the Lie-group model of the Panda against MuJoCo.

1. PoE forward kinematics (screw axes extracted from the MJCF) vs. MuJoCo kinematics.
2. Body Jacobian from PoE vs. MuJoCo's mj_jac.
3. Lie-group Newton IK: quadratic convergence on random reachable poses.
"""
import numpy as np
from liealg import se3
from liealg.panda import Panda

p = Panda(); rng = np.random.default_rng(0)
print("screw axes S_i = (w, -w x q) extracted at q=0:")
for i, S in enumerate(p.S):
    print(f"  S{i+1} =", np.round(S, 4))
print("home pose M:\n", np.round(p.M, 4))
e_fk = e_J = 0.0
for _ in range(20):
    q = rng.uniform(p.q_min, p.q_max)
    e_fk = max(e_fk, np.abs(p.fk(q) - p.fk_poe(q)).max())
    e_J = max(e_J, np.abs(p.jac_body(q) - p.jac_body_poe(q)).max())
print(f"max |FK_PoE - FK_MuJoCo| = {e_fk:.1e}   max |Jb_PoE - Jb_MuJoCo| = {e_J:.1e}")
qt = p.q_home + rng.uniform(-0.4, 0.4, 7)
T = p.fk_poe(qt)
q = p.q_home.copy()
print("IK iterations (error norm):")
for it in range(6):
    e = se3.log(se3.inverse(p.fk_poe(q)) @ T)
    print(f"  {it}: {np.linalg.norm(e):.2e}")
    J = p.jac_body_poe(q)
    q = q + np.linalg.lstsq(J, e, rcond=None)[0]
