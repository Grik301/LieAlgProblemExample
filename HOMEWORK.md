# Homework assignment — from Lie-group theory to real-robot execution

**Course:** Advanced Robotics Lab · *Motion Planning on Lie Groups* (PBL)
**Format:** group project, 3–4 students per group
**Robot:** Franka Emika Panda (7 DoF) — first in MuJoCo (this repository), then on the real arm in the lab
**Weight:** 100 % of the PBL module grade (see rubric below)

The goal is to take every concept from the lecture — SO(3)/SE(3), the exponential/logarithm maps,
Jacobians, the banana distribution, geodesic interpolation and sampling-based planning — and make the
**real** Panda execute a motion that you planned entirely on the Lie group. The code in this repository
is your starting point; you are expected to extend it, not to rewrite it from scratch.

---

## 1. The task

Your robot starts at the `home` configuration. On the table there are a **start block** and a
**goal block** whose poses `T_start, T_goal ∈ SE(3)` are given to you (from the lab's motion-capture
system or measured by hand), and an **obstacle** (a box) between them.

Your group must

1. **Plan** a collision-free end-effector path from `T_start` to `T_goal` on SE(3);
2. **Quantify** the pose uncertainty at the goal that results from joint-encoder/actuation noise;
3. **Execute** the plan on the real Panda and measure how far the achieved pose is from `T_goal`
   using the Lie-group distance `‖log(T_goal⁻¹ T_achieved)‖`.

---

## 2. Phases (mirrors the four PBL phases in the slides)

### Phase 1 — Mathematics (week 1)

* Derive the Panda's space-frame screw axes `S₁…S₇` and home pose `M` **by hand** from the DH /
  URDF parameters, then verify them against `Panda._extract_screws()` (`examples/05_panda_lie_vs_mujoco.py`).
* Implement, in your own words, `exp`, `log` on SE(3) and the body Jacobian; compare numerically
  with `liealg.se3`. Explain in the report why `log(T⁻¹ T_des)` is the right pose error (left-invariance)
  and what goes wrong with Euler angles near the wrist singularity.
* **Deliverable:** 2-page derivation (PDF) + a notebook/script with the numerical checks.

### Phase 2 — Stochastics (week 2)

* Model each joint as `qᵢ = q̄ᵢ + εᵢ, εᵢ ~ N(0, σᵢ²)` with σᵢ measured on the real robot
  (read the encoder at rest and while holding a pose; `franka_msgs/FrankaState` or `robot_state.q`).
* Propagate the uncertainty to the end-effector in two ways: Monte-Carlo through the forward
  kinematics (as in `examples/03_banana_distribution.py`) and first-order `Σ = J_b Σ_q J_bᵀ`.
  Show the "banana" in Cartesian coordinates and its Gaussian image in exponential coordinates
  for at least two configurations (one near a singularity).
* **Deliverable:** figures + a short analysis of when the first-order propagation breaks down.

### Phase 3 — Algorithms (week 3)

* Implement a sampling-based planner (RRT or RRT*) **in joint space** whose goal test, node
  distance and steering use SE(3) quantities: distance `‖log(T₁⁻¹T₂)‖` (weighted rotation/translation),
  steering by screw interpolation `T₁ exp(s log(T₁⁻¹T₂))`, goal reached when the Lie-group error is
  below a threshold. Collision checking against the obstacle box can be done with MuJoCo
  (`mj_collision` / contact count) or with simple capsule checks.
* Post-process the path into a smooth, time-parameterised trajectory (e.g. SLERP for orientation,
  minimum-jerk in the Lie algebra) that respects the Panda's joint-velocity limits.
* Validate in simulation with `examples/04_panda_realtime.py` as a template (real-time viewer,
  log the tracking error).
* **Deliverable:** planner code, a video of the MuJoCo execution, tracking-error plot.

### Phase 4 — Real-robot execution (lab week)

* Transfer the trajectory to the real Panda using one of the allowed toolchains (section 3).
* Record `q(t)` and the end-effector pose; compute the final error `‖log(T_goal⁻¹ T_achieved)‖`
  and compare it with the uncertainty predicted in Phase 2 (does the achieved pose fall inside your
  2σ ellipsoid?).
* **Deliverable:** lab video, logged data (`.csv`/`.npz`), final report, 10-minute group presentation.

---

## 3. Allowed toolchains for the real robot

You may use **any one** of the following to communicate with the Panda. Mixing is allowed
(e.g. plan in Python, execute in C++) as long as the interface between the two is documented.

| Toolchain | Language | Typical use | Notes |
|---|---|---|---|
| **ROS 2** (`franka_ros2`, `ros2_control`, MoveIt 2 optional) | **Python** (rclpy) | publish `JointTrajectory` / use the joint-position or Cartesian-pose controller; read `FrankaState` for encoders | Recommended for groups who did Phase 3 in Python; you can reuse `liealg` directly. Run at ≤ 100 Hz over ROS topics; do **not** try to close a 1 kHz loop in Python. |
| **Franka Control Interface (FCI) / libfranka** (the "Franka UI" + C++ path) | **C++** | `franka::Robot` with `JointPositions` / `CartesianPose` motion generators at 1 kHz | Best real-time performance. Export your trajectory from Python as a `.csv` and stream it from C++. Use the `franka::limitRate` helpers. |
| **MATLAB** (Robotics System Toolbox + ROS Toolbox, or the Franka MATLAB interface) | **MATLAB** | plan/verify with `se3`, `rotm2axang`, `expm/logm`; execute via ROS 2 messages or the Franka MATLAB toolbox | Re-implement `exp`/`log` yourself (MATLAB's `expm`/`logm` are generic and slow); submit the `.m` files. |

Regardless of the toolchain:

* **Simulation first.** No code runs on the real robot before it has been run in MuJoCo with the
  same trajectory file and the tracking error is < 0.02 (norm of the log-error) along the whole path.
* **Joint limits and speed limits** must be enforced in *your* code (`Panda.q_min/q_max`;
  velocity ≤ 50 % of the Panda limits for the first real run), not just by the Franka safety layer.
* The Panda's **safety settings** (collision thresholds, workspace limits) are set by the TA in Desk
  and must not be changed. Always keep the enabling device in hand; one group member is the
  designated "E-stop person" during every run.
* The gripper is not used in this assignment (it can stay in the `home` keyframe state).

---

## 4. Deliverables and submission

Submit **one pull request per group** to this repository (branch `group-<name>`), containing

```
groups/<group-name>/
├── README.md            # who did what, how to run, toolchain used
├── report.pdf           # ≤ 8 pages: theory (Phase 1), uncertainty (Phase 2), planner (Phase 3), lab results (Phase 4)
├── derivation/          # Phase 1 notebook / script
├── planner/             # Phase 3 code (Python / C++ / MATLAB)
├── execution/           # ROS 2 nodes, libfranka program or MATLAB scripts used on the real robot
├── data/                # logged q(t), poses, computed errors (.csv / .npz)
└── media/               # MuJoCo video + lab video (links are fine if > 50 MB)
```

Deadline: **Friday of the lab week, 23:59** (the date announced on the course page). The 10-minute
presentation takes place the following Monday.

---

## 5. Grading rubric (same weights as in the lecture)

| Criterion | Weight | What we look at |
|---|---|---|
| **Mathematical rigor** | 40 % | correct SE(3) implementations and derivations, screw axes verified, Lie-group error used consistently, singularities discussed |
| **Stochastic modelling** | 30 % | measured σ, Monte-Carlo vs first-order propagation, banana vs Gaussian in exponential coordinates, honest comparison with the achieved real-robot error |
| **Algorithm efficiency** | 30 % | planner runs in real time on the lab PC, trajectory feasible and smooth, successful collision-free execution on the real Panda, tracking error reported |

Bonus (up to +10 %): RRT* instead of RRT with the Lie-group cost, or uncertainty-aware planning
(the planner keeps the 2σ ellipsoid of the end-effector away from the obstacle).

---

## 6. Hints

* `Panda.ik()` is Newton's method on SE(3) — it converges quadratically but only from a reasonable
  initial guess; warm-start it with the previous waypoint's solution.
* The real Panda's `hand` frame and the TCP in `liealg/panda.py` (`TCP_OFFSET`, 10.34 cm) correspond
  to the `panda_hand_tcp` frame in `franka_description`. Check this once with a ruler.
* When something looks wrong on the real robot, compare the **same** joint vector in MuJoCo
  (`Panda.fk(q)`) and in your real-robot FK before touching the controller.
* Questions: open a GitHub issue in this repository with the label `homework`.
