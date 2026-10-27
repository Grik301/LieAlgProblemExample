"""Franka Emika Panda (7 DoF) in MuJoCo, seen through Lie-group glasses.

* The kinematic chain is *extracted* from the MJCF at q = 0 as space-frame screw axes, so the
  same product-of-exponentials / Jacobian / IK code as the toy arm works on the real robot.
* MuJoCo provides the physics (contacts, actuator dynamics, implicit integration); this module
  provides the geometry: FK/IK on SE(3), body Jacobians, and a resolved-rate task-space controller
  that drives MuJoCo's position actuators along an SE(3) geodesic.
"""
import os
import numpy as np
import mujoco
from . import se3, so3

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
SCENE_XML = os.path.join(MODEL_DIR, "panda_scene.xml")
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
EE_BODY = "hand"
# Flange-to-TCP offset of the Panda hand (between the finger pads).
TCP_OFFSET = se3.make(np.eye(3), [0.0, 0.0, 0.1034])


class Panda:
    def __init__(self, xml=SCENE_XML):
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self._scratch = mujoco.MjData(self.model)       # for kinematics queries that must not touch the sim
        self.jids = [self.model.joint(n).id for n in ARM_JOINTS]
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in self.jids])
        self.vadr = np.array([self.model.jnt_dofadr[j] for j in self.jids])
        self.ee_body = self.model.body(EE_BODY).id
        self.n = 7
        self.q_home = self.model.key("home").qpos[self.qadr].copy()
        self.q_min = self.model.jnt_range[self.jids, 0]
        self.q_max = self.model.jnt_range[self.jids, 1]
        self.target_mocap = self.model.body("target").mocapid[0]
        self.S, self.M = self._extract_screws()
        self._q_ref = None

    # ---------------- MuJoCo kinematics ----------------
    def _ee_pose(self, d):
        R = d.xmat[self.ee_body].reshape(3, 3)
        return se3.make(R, d.xpos[self.ee_body]) @ TCP_OFFSET

    def fk(self, q):
        """TCP pose from MuJoCo forward kinematics (does not disturb the running simulation)."""
        d = self._scratch
        d.qpos[:] = self.data.qpos
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        return self._ee_pose(d)

    def jac_body(self, q):
        """Body Jacobian (twist = (w, v) expressed in the TCP frame), 6x7, from MuJoCo mj_jac."""
        d = self._scratch
        d.qpos[:] = self.data.qpos
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        T = self._ee_pose(d)
        jacp = np.zeros((3, self.model.nv)); jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, d, jacp, jacr, T[:3, 3], self.ee_body)   # world-frame, at the TCP point
        R = T[:3, :3]
        return np.vstack([R.T @ jacr[:, self.vadr], R.T @ jacp[:, self.vadr]])

    # ---------------- product of exponentials extracted from the model ----------------
    def _extract_screws(self):
        d = self._scratch
        d.qpos[:] = 0.0
        d.qpos[self.qadr] = 0.0
        mujoco.mj_kinematics(self.model, d)
        screws = []
        for j in self.jids:
            axis_w = d.xmat[self.model.jnt_bodyid[j]].reshape(3, 3) @ self.model.jnt_axis[j]
            point_w = d.xanchor[j]
            screws.append(se3.screw_axis(axis_w, point_w))
        M = self._ee_pose(d)
        return screws, M

    def fk_poe(self, q):
        """Pure-Python PoE forward kinematics; must agree with MuJoCo (see tests)."""
        return se3.poe(self.S, q, self.M)

    def jac_body_poe(self, q):
        return se3.body_jacobian(self.S, q, self.M)

    def ik(self, T_des, q0=None, **kw):
        q0 = self.q_home if q0 is None else q0
        q, err, ok = se3.ik_newton(self.fk_poe, self.jac_body_poe, T_des, q0, **kw)
        return np.clip(q, self.q_min, self.q_max), err, ok

    # ---------------- simulation helpers ----------------
    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self._q_ref = None
        mujoco.mj_forward(self.model, self.data)

    @property
    def q(self):
        return self.data.qpos[self.qadr].copy()

    @property
    def dq(self):
        return self.data.qvel[self.vadr].copy()

    def ee(self):
        return self._ee_pose(self.data)

    def set_target_marker(self, T):
        self.data.mocap_pos[self.target_mocap] = T[:3, 3]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, T[:3, :3].reshape(-1))
        self.data.mocap_quat[self.target_mocap] = quat

    def command_q(self, q_des):
        self.data.ctrl[:7] = np.clip(q_des, self.q_min, self.q_max)

    def resolved_rate(self, T_des, k=8.0, damping=1e-3, dq_max=2.0):
        """Resolved-rate task-space control on SE(3).

        The error is the body twist e = log(T^-1 T_des) measured on the *simulated* robot; the
        joint velocity dq = J_b^+ (k e) (damped least squares, speed-limited) is integrated into a
        reference position q_ref which is sent to MuJoCo's position actuators. Integrating the
        reference (rather than commanding q + dq*dt) avoids the steady-state lag of the servo.
        """
        if self._q_ref is None:
            self._q_ref = self.q
        T = self.ee()
        e = se3.log(se3.inverse(T) @ T_des)
        J = self.jac_body(self.q)
        dq = np.linalg.solve(J.T @ J + damping * np.eye(7), J.T @ (k * e))
        nrm = np.linalg.norm(dq)
        if nrm > dq_max:
            dq *= dq_max / nrm
        self._q_ref = np.clip(self._q_ref + dq * self.model.opt.timestep * self.ctrl_substeps,
                              self.q_min, self.q_max)
        self.command_q(self._q_ref)
        return e

    ctrl_substeps = 5          # control runs every 5 physics steps (2 ms * 5 = 10 ms)

    def step(self, n=None):
        for _ in range(n or self.ctrl_substeps):
            mujoco.mj_step(self.model, self.data)
