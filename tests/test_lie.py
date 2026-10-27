import numpy as np, pytest
from liealg import so3, se3, ToyArm3

rng = np.random.default_rng(42)

def test_so3_exp_log_roundtrip():
    for _ in range(20):
        w = rng.standard_normal(3); w *= min(1.0, 3.0 / np.linalg.norm(w))
        R = so3.exp(w)
        assert so3.is_rotation(R)
        assert np.allclose(so3.log(R), w, atol=1e-8)

def test_so3_log_near_pi():
    R = so3.exp(np.array([np.pi - 1e-9, 0, 0]))
    assert abs(np.linalg.norm(so3.log(R)) - np.pi) < 1e-6

def test_se3_exp_log_roundtrip_and_inverse():
    for _ in range(20):
        xi = rng.standard_normal(6); xi[:3] *= min(1.0, 3.0 / np.linalg.norm(xi[:3]))
        T = se3.exp(xi)
        assert np.allclose(se3.log(T), xi, atol=1e-8)
        assert np.allclose(se3.inverse(T) @ T, np.eye(4), atol=1e-12)

def test_adjoint_identity():
    T = se3.exp(rng.standard_normal(6)); xi = rng.standard_normal(6)
    lhs = T @ se3.hat(xi) @ se3.inverse(T)
    assert np.allclose(se3.vee(lhs), se3.adjoint(T) @ xi)

def test_interpolation_is_geodesic():
    T1, T2 = se3.exp(rng.standard_normal(6)), se3.exp(rng.standard_normal(6))
    d = se3.distance(T1, T2)
    for s in (0.25, 0.5, 0.75):
        assert abs(se3.distance(T1, se3.interpolate(T1, T2, s)) - s * d) < 1e-8

def test_toy_arm_jacobian_numeric():
    arm = ToyArm3(); q = rng.uniform(-1, 1, 3); h = 1e-6
    Jb = arm.jac_body(q); Jn = np.zeros((6, 3))
    for i in range(3):
        e = np.zeros(3); e[i] = h
        Jn[:, i] = se3.vee(se3.inverse(arm.fk(q)) @ (arm.fk(q + e) - arm.fk(q - e)) / (2 * h))
    assert np.allclose(Jb, Jn, atol=1e-6)

def test_toy_arm_dynamics_consistency():
    arm = ToyArm3(); q = rng.uniform(-1, 1, 3); dq = rng.standard_normal(3); tau = rng.standard_normal(3)
    M = arm.mass_matrix(q)
    assert np.allclose(M, M.T) and np.all(np.linalg.eigvalsh(M) > 0)
    ddq = arm.forward_dynamics(q, dq, tau)
    assert np.allclose(arm.inverse_dynamics(q, dq, ddq), tau, atol=1e-9)

def test_toy_arm_energy_conservation():
    arm = ToyArm3(g=0.0); q = np.array([0.3, -0.4, 0.9]); dq = np.array([0.5, -0.3, 0.8])
    E0 = 0.5 * dq @ arm.mass_matrix(q) @ dq
    for _ in range(200):
        q, dq = arm.step(q, dq, np.zeros(3), 2e-3)
    assert abs(0.5 * dq @ arm.mass_matrix(q) @ dq - E0) < 1e-6

mujoco = pytest.importorskip("mujoco")
from liealg.panda import Panda

@pytest.fixture(scope="module")
def panda():
    return Panda()

def test_panda_poe_matches_mujoco(panda):
    for _ in range(10):
        q = rng.uniform(panda.q_min, panda.q_max)
        assert np.allclose(panda.fk(q), panda.fk_poe(q), atol=1e-10)
        assert np.allclose(panda.jac_body(q), panda.jac_body_poe(q), atol=1e-10)

def test_panda_ik_converges(panda):
    T = panda.fk_poe(panda.q_home + rng.uniform(-0.3, 0.3, 7))
    q, err, ok = panda.ik(T, damping=1e-6)
    assert ok and err < 1e-8
