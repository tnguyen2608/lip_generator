"""
Batch generation script for two-step motion dataset with random initial poses.

Generates a single NPZ file containing trajectories with two consecutive steps,
starting from randomized initial poses:
- Each trajectory has: wait_before + step1 + wait_mid + step2 + wait_after

Dataset fields:
- q: (n, nq) - joint positions with base
- qd: (n, nv) - joint velocities
- T_blf: (n, 4) - body frame to left foot frame transform (x, y, z, yaw)
- T_brf: (n, 4) - body frame to right foot frame transform (x, y, z, yaw)
- T_stsw: (n, 4) - stance foot to swing foot transform (x, y, z, yaw)
- p_wcom: (n, 3) - CoM position in world frame
- T_wbase: (n, 7) - base transform in world frame (x, y, z, qw, qx, qy, qz)
- v_b: (n, 6) - base velocity in base frame (linear xyz, angular xyz)
- cmd_footstep: (n, 4) - [x, y, sin(yaw), cos(yaw)] in stance foot frame
- cmd_stance: (n, 1) - 0=left stance, 1=right stance
- cmd_countdown: (n, 1) - countdown timer: 0 during wait, 1->0 during step
- traj: (k,) - starting indices of each trajectory
- traj_dt: float - time step between frames
"""

import numpy as np
import pinocchio
import crocoddyl
import gc
from step import SimpleBipedGaitProblem
try:
    import meshcat
    import meshcat.transformations as tf
    MESHCAT_AVAILABLE = True
except ImportError:
    MESHCAT_AVAILABLE = False

# Configuration
OUTPUT_FILE = "stepping_dataset_11.19_rand2step.npz"
NUM_KNOTS = 30
TIME_STEP = 0.02
STEP_KNOTS = NUM_KNOTS
SUPPORT_KNOTS = 2
WITHDISPLAY = True

# Step generation parameters
STEP_HEIGHT = 0.35  # Step height in meters
WAIT_TIME_RANGE = (0.2, 0.4)  # Waiting period before/after steps (seconds)
MID_WAIT_TIME_RANGE = (0.3, 0.6)  # Waiting period between two steps (seconds)

# Random initial pose ranges (relative to half-sitting)
LEFT_RAND_RANGE = {
    "x": (-0.1, 0.2),  # Forward/backward range (meters)
    "y": (-0.02, 0.02),  # Lateral range (meters)
    "yaw": (-0.2, 0.2),
}
RIGHT_RAND_RANGE = {
    "x": (-0.1, 0.2),  # Forward/backward range (meters)
    "y": (-0.02, 0.02),  # Lateral range (meters)
    "yaw": (-0.2, 0.2),
}

# First step swinging ranges
LEFT_SWING_RANGE = {
    "x": (-0.15, 0.15),  # Forward/backward range (meters)
    "y": (0.2, 0.25),  # Lateral range (meters)
    "yaw": (-0.2, 0.2),
}
RIGHT_SWING_RANGE = {
    "x": (-0.15, 0.15),  # Forward/backward range (meters)
    "y": (-0.25, -0.2),  # Lateral range (meters)
    "yaw": (-0.2, 0.2),
}

# Second step ranges (relative displacements)
LEFT_SECOND_RANGE = {
    "x": (-0.15, 0.15),
    "y": (0.15, 0.25),
    "yaw": (-0.1, 0.1),
}
RIGHT_SECOND_RANGE = {
    "x": (-0.15, 0.15),
    "y": (-0.25, -0.15),
    "yaw": (-0.1, 0.1),
}

# Grid sampling parameters
GRID_X_STEPS = 5  # Number of steps in x direction for first step
GRID_Y_STEPS = 3  # Number of steps in y direction for first step
GRID_YAW_STEPS = 3  # Number of steps in yaw direction for first step
GRID_X_STEPS_2 = 3  # Number of steps in x direction for second step
GRID_Y_STEPS_2 = 2  # Number of steps in y direction for second step
GRID_YAW_STEPS_2 = 2  # Number of steps in yaw direction for second step

# Random initial pose sampling
RAND_INIT_X_SAMPLES = 2
RAND_INIT_Y_SAMPLES = 2
RAND_INIT_YAW_SAMPLES = 2

# Solver parameters
MAX_ITERATIONS = 100
SOLVER_THRESHOLD = 1e-7


def load_robot():
    """Load the robot model and set up initial configuration."""
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, "model", "T1_7dof_arms_with_gripper.urdf")
    package_dir = os.path.join(script_dir, "model")

    robot = pinocchio.RobotWrapper.BuildFromURDF(
        urdf_path,
        package_dirs=[package_dir],
        root_joint=pinocchio.JointModelFreeFlyer(),
    )

    half_sitting = np.array(
        [
            0,
            0,
            0.665,  # base position
            0,
            0,
            0,
            1,  # base orientation (quaternion)
            0,
            0,  # torso joints
            0.2,
            -1.35,
            0,
            -0.5,
            0.0,
            0.0,
            0.0,  # left arm
            0.2,
            1.35,
            0,
            0.5,
            0.0,
            0.0,
            0.0,  # right arm
            0,  # head
            -0.1,
            0,
            0,
            0.2,
            -0.1,
            0,  # left leg
            -0.1,
            0,
            0,
            0.2,
            -0.1,
            0,  # right leg
        ]
    )
    robot.model.referenceConfigurations["half_sitting"] = half_sitting

    return robot


def rotation_matrix_to_yaw(R):
    """Extract yaw angle from rotation matrix."""
    return np.arctan2(R[1, 0], R[0, 0])


def transform_to_stance_frame(target_pos, stance_pos, stance_R, target_yaw):
    """
    Transform target position from world frame to stance foot frame.

    Args:
        target_pos: [x, y, z] in world frame
        stance_pos: [x, y, z] stance foot position in world frame
        stance_R: 3x3 rotation matrix of stance foot

    Returns:
        [x, y, z, yaw] in stance foot frame
    """
    # Transform position to stance frame
    p_world = target_pos - stance_pos
    p_stance = stance_R.T @ p_world

    return np.array([p_stance[0], p_stance[1], 0.0, target_yaw])


def generate_grid_samples_with_random_init(
    lfPos0, rfPos0, left_rand_range, right_rand_range,
    left_swing_range, right_swing_range, left_second_range, right_second_range,
    grid_x_steps, grid_y_steps, grid_yaw_steps,
    grid_x_steps_2, grid_y_steps_2, grid_yaw_steps_2,
    rand_x_samples, rand_y_samples, rand_yaw_samples
):
    """
    Generate grid of two-step stepping trajectories with random initial poses.

    For each random starting pose, generate first step targets from swing_range.
    Second step targets are displacements applied after the first step.
    """
    samples = []

    # Generate random initial foot positions
    lf_init_x = np.linspace(left_rand_range["x"][0], left_rand_range["x"][1], max(2, rand_x_samples))
    lf_init_y = np.linspace(left_rand_range["y"][0], left_rand_range["y"][1], max(2, rand_y_samples))
    lf_init_yaw = np.linspace(left_rand_range["yaw"][0], left_rand_range["yaw"][1], max(2, rand_yaw_samples))

    rf_init_x = np.linspace(right_rand_range["x"][0], right_rand_range["x"][1], max(2, rand_x_samples))
    rf_init_y = np.linspace(right_rand_range["y"][0], right_rand_range["y"][1], max(2, rand_y_samples))
    rf_init_yaw = np.linspace(right_rand_range["yaw"][0], right_rand_range["yaw"][1], max(2, rand_yaw_samples))

    # First step swing ranges
    x_values_right = np.linspace(right_swing_range["x"][0], right_swing_range["x"][1], grid_x_steps)
    y_values_right = np.linspace(right_swing_range["y"][0], right_swing_range["y"][1], grid_y_steps)
    yaw_values_right = np.linspace(right_swing_range["yaw"][0], right_swing_range["yaw"][1], grid_yaw_steps)

    x_values_left = np.linspace(left_swing_range["x"][0], left_swing_range["x"][1], grid_x_steps)
    y_values_left = np.linspace(left_swing_range["y"][0], left_swing_range["y"][1], grid_y_steps)
    yaw_values_left = np.linspace(left_swing_range["yaw"][0], left_swing_range["yaw"][1], grid_yaw_steps)

    # Second step ranges
    dx_values_left_second = np.linspace(left_second_range["x"][0], left_second_range["x"][1], grid_x_steps_2)
    dy_values_left_second = np.linspace(left_second_range["y"][0], left_second_range["y"][1], grid_y_steps_2)
    yaw_values_left_second = np.linspace(left_second_range["yaw"][0], left_second_range["yaw"][1], grid_yaw_steps_2)

    dx_values_right_second = np.linspace(right_second_range["x"][0], right_second_range["x"][1], grid_x_steps_2)
    dy_values_right_second = np.linspace(right_second_range["y"][0], right_second_range["y"][1], grid_y_steps_2)
    yaw_values_right_second = np.linspace(right_second_range["yaw"][0], right_second_range["yaw"][1], grid_yaw_steps_2)

    # Sequence 1: Random init -> Right swing -> Left swing
    for lf_dx_init in lf_init_x:
        for lf_dy_init in lf_init_y:
            for lf_yaw_init in lf_init_yaw:
                # Compute random left foot starting position
                left_init = lfPos0.copy()
                left_init[0] += lf_dx_init
                left_init[1] += lf_dy_init

                # Apply yaw rotation to right foot relative to left foot
                rel_rf = rfPos0 - lfPos0
                cos_yaw = np.cos(lf_yaw_init)
                sin_yaw = np.sin(lf_yaw_init)
                rotated_rf = np.array([cos_yaw * rel_rf[0] - sin_yaw * rel_rf[1],
                                      sin_yaw * rel_rf[0] + cos_yaw * rel_rf[1], rel_rf[2]])
                right_init = left_init + rotated_rf

                # Generate first step targets (right foot swings)
                for dx1 in x_values_right:
                    for dy1 in y_values_right:
                        for yaw1 in yaw_values_right:
                            rf_step1 = left_init.copy()
                            rf_step1[0] += dx1
                            rf_step1[1] += dy1

                            # Generate second step displacements (left foot swings)
                            for dx2 in dx_values_left_second:
                                for dy2 in dy_values_left_second:
                                    for yaw2 in yaw_values_left_second:
                                        samples.append(
                                            {
                                                "step1": {
                                                    "left_target": left_init.copy(),
                                                    "right_target": rf_step1.copy(),
                                                    "stance_foot": "left",
                                                    "swing_foot": "right",
                                                    "target_yaw": lf_yaw_init + yaw1,
                                                },
                                                "step2_displacement": {
                                                    "dx": dx2,
                                                    "dy": dy2,
                                                    "stance_foot": "right",
                                                    "swing_foot": "left",
                                                    "target_yaw": yaw2,
                                                },
                                            }
                                        )

    # Sequence 2: Random init -> Left swing -> Right swing
    for rf_dx_init in rf_init_x:
        for rf_dy_init in rf_init_y:
            for rf_yaw_init in rf_init_yaw:
                # Compute random right foot starting position
                right_init = rfPos0.copy()
                right_init[0] += rf_dx_init
                right_init[1] += rf_dy_init

                # Apply yaw rotation to left foot relative to right foot
                rel_lf = lfPos0 - rfPos0
                cos_yaw = np.cos(rf_yaw_init)
                sin_yaw = np.sin(rf_yaw_init)
                rotated_lf = np.array([cos_yaw * rel_lf[0] - sin_yaw * rel_lf[1],
                                      sin_yaw * rel_lf[0] + cos_yaw * rel_lf[1], rel_lf[2]])
                left_init = right_init + rotated_lf

                # Generate first step targets (left foot swings)
                for dx1 in x_values_left:
                    for dy1 in y_values_left:
                        for yaw1 in yaw_values_left:
                            lf_step1 = right_init.copy()
                            lf_step1[0] += dx1
                            lf_step1[1] += dy1

                            # Generate second step displacements (right foot swings)
                            for dx2 in dx_values_right_second:
                                for dy2 in dy_values_right_second:
                                    for yaw2 in yaw_values_right_second:
                                        samples.append(
                                            {
                                                "step1": {
                                                    "left_target": lf_step1.copy(),
                                                    "right_target": right_init.copy(),
                                                    "stance_foot": "right",
                                                    "swing_foot": "left",
                                                    "target_yaw": rf_yaw_init + yaw1,
                                                },
                                                "step2_displacement": {
                                                    "dx": dx2,
                                                    "dy": dy2,
                                                    "stance_foot": "left",
                                                    "swing_foot": "right",
                                                    "target_yaw": yaw2,
                                                },
                                            }
                                        )

    return samples


def solve_stepping_problem(gait, x0, left_target, right_target, target_yaw=0.0, verbose=False):
    """Solve a stepping problem."""
    try:
        # CRITICAL: Make a copy to avoid shared reference issues
        x0_copy = x0.copy()
        problem = gait.createSingleStepProblem(x0_copy, left_target, right_target, TIME_STEP, STEP_KNOTS, SUPPORT_KNOTS, STEP_HEIGHT, target_yaw)
        solver = crocoddyl.SolverIntro(problem)
        solver.th_stop = SOLVER_THRESHOLD

        if verbose:
            solver.setCallbacks([crocoddyl.CallbackVerbose()])

        # CRITICAL: Use list comprehension with copy() to avoid shared references
        xs = [x0_copy.copy() for _ in range(solver.problem.T + 1)]
        us = solver.problem.quasiStatic([x0_copy.copy() for _ in range(solver.problem.T)])
        solver.solve(xs, us, MAX_ITERATIONS, False)

        success = solver.stop < SOLVER_THRESHOLD
        return solver, success
    except Exception as e:
        print(f"Error solving problem: {e}")
        return None, False


def generate_waiting_frames(robot, gait, x0, num_frames, left_target, right_target, yaw_target):
    """Generate waiting frames where the robot stands still."""
    nq = robot.model.nq
    nv = robot.model.nv

    q_data = np.tile(x0[:nq], (num_frames, 1))
    qd_data = np.zeros((num_frames, nv))
    T_blf_data = np.zeros((num_frames, 4))
    T_brf_data = np.zeros((num_frames, 4))
    T_stsw_data = np.zeros((num_frames, 4))
    p_wcom_data = np.zeros((num_frames, 3))
    T_wbase_data = np.zeros((num_frames, 7))
    v_b_data = np.zeros((num_frames, 6))
    cmd_footstep_data = np.zeros((num_frames, 4))
    cmd_stance_data = np.zeros((num_frames, 1))
    cmd_countdown_data = np.zeros((num_frames, 1))

    rdata = robot.model.createData()
    q = x0[:nq]

    pinocchio.forwardKinematics(robot.model, rdata, q)
    pinocchio.updateFramePlacements(robot.model, rdata)

    lf_init = rdata.oMf[gait.lfId].translation.copy()
    rf_init = rdata.oMf[gait.rfId].translation.copy()

    left_movement = np.linalg.norm(left_target - lf_init)
    right_movement = np.linalg.norm(right_target - rf_init)

    if left_movement > right_movement:
        stance_is_left = 0
        swing_target = lf_init
        stance_foot_id = gait.rfId
        swing_foot_id = gait.lfId
    else:
        stance_is_left = 1
        swing_target = rf_init
        stance_foot_id = gait.lfId
        swing_foot_id = gait.rfId

    body_pos = q[:3]
    body_quat = q[3:7]
    body_R = pinocchio.Quaternion(body_quat[3], body_quat[0], body_quat[1], body_quat[2]).toRotationMatrix()

    lf_world_pos = rdata.oMf[gait.lfId].translation
    rf_world_pos = rdata.oMf[gait.rfId].translation
    lf_world_R = rdata.oMf[gait.lfId].rotation
    rf_world_R = rdata.oMf[gait.rfId].rotation

    lf_body_pos = body_R.T @ (lf_world_pos - body_pos)
    rf_body_pos = body_R.T @ (rf_world_pos - body_pos)

    lf_body_R = body_R.T @ lf_world_R
    rf_body_R = body_R.T @ rf_world_R
    lf_body_yaw = rotation_matrix_to_yaw(lf_body_R)
    rf_body_yaw = rotation_matrix_to_yaw(rf_body_R)

    com_world = pinocchio.centerOfMass(robot.model, rdata, q)

    stance_pos = rdata.oMf[stance_foot_id].translation
    stance_R = rdata.oMf[stance_foot_id].rotation
    swing_pos = rdata.oMf[swing_foot_id].translation
    swing_R = rdata.oMf[swing_foot_id].rotation

    swing_stance_pos = stance_R.T @ (swing_pos - stance_pos)
    swing_stance_R = stance_R.T @ swing_R
    swing_stance_yaw = rotation_matrix_to_yaw(swing_stance_R)

    cmd_footstep = transform_to_stance_frame(swing_target, stance_pos, stance_R, swing_stance_yaw)

    for i in range(num_frames):
        T_blf_data[i] = [lf_body_pos[0], lf_body_pos[1], lf_body_pos[2], lf_body_yaw]
        T_brf_data[i] = [rf_body_pos[0], rf_body_pos[1], rf_body_pos[2], rf_body_yaw]
        T_stsw_data[i] = [swing_stance_pos[0], swing_stance_pos[1], swing_stance_pos[2], swing_stance_yaw]
        p_wcom_data[i] = com_world
        T_wbase_data[i] = [body_pos[0], body_pos[1], body_pos[2], body_quat[3], body_quat[0], body_quat[1], body_quat[2]]
        cmd_stance_data[i, 0] = 0 if stance_is_left else 1
        cmd_footstep_data[i] = cmd_footstep

    return {
        "q": q_data,
        "qd": qd_data,
        "T_blf": T_blf_data,
        "T_brf": T_brf_data,
        "T_stsw": T_stsw_data,
        "p_wcom": p_wcom_data,
        "T_wbase": T_wbase_data,
        "v_b": v_b_data,
        "cmd_footstep": cmd_footstep_data,
        "cmd_stance": cmd_stance_data,
        "cmd_countdown": cmd_countdown_data,
    }


def extract_trajectory_data(robot, solver, gait, left_target, right_target, yaw_target):
    """Extract trajectory data from solver."""
    nq = robot.model.nq
    nv = robot.model.nv
    T = len(solver.xs)

    q_data = np.zeros((T, nq))
    qd_data = np.zeros((T, nv))
    T_blf_data = np.zeros((T, 4))
    T_brf_data = np.zeros((T, 4))
    T_stsw_data = np.zeros((T, 4))
    p_wcom_data = np.zeros((T, 3))
    T_wbase_data = np.zeros((T, 7))
    v_b_data = np.zeros((T, 6))
    cmd_footstep_data = np.zeros((T, 4))
    cmd_stance_data = np.zeros((T, 1))
    cmd_countdown_data = np.zeros((T, 1))

    rdata = robot.model.createData()
    pinocchio.forwardKinematics(robot.model, rdata, solver.xs[0][:nq])
    pinocchio.updateFramePlacements(robot.model, rdata)

    lf_init = rdata.oMf[gait.lfId].translation.copy()
    rf_init = rdata.oMf[gait.rfId].translation.copy()

    left_movement = np.linalg.norm(left_target - lf_init)
    right_movement = np.linalg.norm(right_target - rf_init)

    if left_movement > right_movement:
        stance_is_left = 0
        swing_target = left_target
        stance_foot_id = gait.rfId
        swing_foot_id = gait.lfId
    else:
        stance_is_left = 1
        swing_target = right_target
        stance_foot_id = gait.lfId
        swing_foot_id = gait.rfId

    for t in range(T):
        q = solver.xs[t][:nq]
        qd = solver.xs[t][nq:]

        q_data[t] = q
        qd_data[t] = qd

        pinocchio.forwardKinematics(robot.model, rdata, q)
        pinocchio.updateFramePlacements(robot.model, rdata)

        body_pos = q[:3]
        body_quat = q[3:7]
        body_R = pinocchio.Quaternion(body_quat[3], body_quat[0], body_quat[1], body_quat[2]).toRotationMatrix()

        lf_world_pos = rdata.oMf[gait.lfId].translation
        rf_world_pos = rdata.oMf[gait.rfId].translation
        lf_world_R = rdata.oMf[gait.lfId].rotation
        rf_world_R = rdata.oMf[gait.rfId].rotation

        lf_body_pos = body_R.T @ (lf_world_pos - body_pos)
        rf_body_pos = body_R.T @ (rf_world_pos - body_pos)

        lf_body_R = body_R.T @ lf_world_R
        rf_body_R = body_R.T @ rf_world_R
        lf_body_yaw = rotation_matrix_to_yaw(lf_body_R)
        rf_body_yaw = rotation_matrix_to_yaw(rf_body_R)

        v_world_linear = qd[:3]
        v_world_angular = qd[3:6]
        v_b_linear = body_R.T @ v_world_linear
        v_b_angular = body_R.T @ v_world_angular
        v_b_data[t] = np.concatenate([v_b_linear, v_b_angular])

        T_blf_data[t] = np.array([lf_body_pos[0], lf_body_pos[1], lf_body_pos[2], lf_body_yaw])
        T_brf_data[t] = np.array([rf_body_pos[0], rf_body_pos[1], rf_body_pos[2], rf_body_yaw])

        com_world = pinocchio.centerOfMass(robot.model, rdata, q)
        p_wcom_data[t] = com_world

        T_wbase_data[t] = np.array([body_pos[0], body_pos[1], body_pos[2], body_quat[3], body_quat[0], body_quat[1], body_quat[2]])

        stance_pos = rdata.oMf[stance_foot_id].translation
        stance_R = rdata.oMf[stance_foot_id].rotation
        swing_pos = rdata.oMf[swing_foot_id].translation
        swing_R = rdata.oMf[swing_foot_id].rotation

        swing_stance_pos = stance_R.T @ (swing_pos - stance_pos)
        swing_stance_R = stance_R.T @ swing_R
        swing_stance_yaw = rotation_matrix_to_yaw(swing_stance_R)
        T_stsw_data[t] = np.array([swing_stance_pos[0], swing_stance_pos[1], swing_stance_pos[2], swing_stance_yaw])

        cmd_footstep_data[t] = transform_to_stance_frame(swing_target, stance_pos, stance_R, yaw_target)

        cmd_stance_data[t, 0] = 0 if stance_is_left else 1

        progress = t / (T - 1) if T > 1 else 0
        cmd_countdown_data[t, 0] = 1 - progress

    return {
        "q": q_data,
        "qd": qd_data,
        "T_blf": T_blf_data,
        "T_brf": T_brf_data,
        "T_stsw": T_stsw_data,
        "p_wcom": p_wcom_data,
        "T_wbase": T_wbase_data,
        "v_b": v_b_data,
        "cmd_footstep": cmd_footstep_data,
        "cmd_stance": cmd_stance_data,
        "cmd_countdown": cmd_countdown_data,
    }


def create_display(robot):
    """Create a display instance for visualization."""
    if WITHDISPLAY:
        try:
            display = crocoddyl.MeshcatDisplay(robot)
            display.rate = -1
            display.freq = 1
            print("[Meshcat] Visualization enabled - open browser at http://localhost:7000")
            return display
        except Exception as e:
            print(f"[Warning] Failed to create display: {e}")
            return None
    return None


def visualize_trajectory(display, solver):
    """Visualize a single trajectory using existing display."""
    if display is not None:
        display.displayFromSolver(solver)


def main():
    """Main batch generation loop."""
    print("=" * 80)
    print("Two-Step Motion Dataset Generator with Random Initial Poses")
    print("=" * 80)

    # Load robot
    print("\n[1/4] Loading robot model...")
    robot = load_robot()

    # Initial state (half-sitting)
    q0 = robot.model.referenceConfigurations["half_sitting"].copy()
    v0 = np.zeros(robot.model.nv)
    x0 = np.concatenate([q0, v0])

    # Get initial foot positions
    rightFoot = "right_foot_link"
    leftFoot = "left_foot_link"

    rdata = robot.model.createData()
    pinocchio.forwardKinematics(robot.model, rdata, q0)
    pinocchio.updateFramePlacements(robot.model, rdata)

    rfId = robot.model.getFrameId(rightFoot)
    lfId = robot.model.getFrameId(leftFoot)
    rfPos0 = rdata.oMf[rfId].translation.copy()
    lfPos0 = rdata.oMf[lfId].translation.copy()

    print(f"Initial right foot position: {rfPos0}")
    print(f"Initial left foot position: {lfPos0}")

    # Initialize gait problem
    print("\n[2/4] Initializing gait problem...")
    gait = SimpleBipedGaitProblem(robot.model, rightFoot, leftFoot, fwddyn=False)

    # Create display
    display = create_display(robot)
    if display is not None:
        print("Visualization enabled")

    # Generate grid samples with random initialization
    print("\n[3/4] Generating grid samples with random initial poses...")
    grid_samples = generate_grid_samples_with_random_init(
        lfPos0, rfPos0,
        LEFT_RAND_RANGE, RIGHT_RAND_RANGE,
        LEFT_SWING_RANGE, RIGHT_SWING_RANGE,
        LEFT_SECOND_RANGE, RIGHT_SECOND_RANGE,
        GRID_X_STEPS, GRID_Y_STEPS, GRID_YAW_STEPS,
        GRID_X_STEPS_2, GRID_Y_STEPS_2, GRID_YAW_STEPS_2,
        RAND_INIT_X_SAMPLES, RAND_INIT_Y_SAMPLES, RAND_INIT_YAW_SAMPLES
    )
    print(f"Total grid samples: {len(grid_samples)}")

    # Accumulate all trajectory data
    all_q = []
    all_qd = []
    all_T_blf = []
    all_T_brf = []
    all_T_stsw = []
    all_p_wcom = []
    all_T_wbase = []
    all_v_b = []
    all_cmd_footstep = []
    all_cmd_stance = []
    all_cmd_countdown = []
    traj_starts = [0]

    successful_samples = 0

    for i, sample in enumerate(grid_samples):
        print(f"\n--- Sample {i + 1}/{len(grid_samples)} ---")

        step1 = sample["step1"]
        step2_disp = sample["step2_displacement"]

        print(f"Step 1: {step1['swing_foot']} swings (stance: {step1['stance_foot']})")
        print(f"  Left foot: {lfPos0[:2]} -> {step1['left_target'][:2]}")
        print(f"  Right foot: {rfPos0[:2]} -> {step1['right_target'][:2]}")

        # Generate random waiting time before stepping
        wait_time_before = np.random.uniform(WAIT_TIME_RANGE[0], WAIT_TIME_RANGE[1])
        wait_frames_before = int(wait_time_before / TIME_STEP)

        # Generate waiting frames before first step
        wait_data_before = generate_waiting_frames(
            robot, gait, x0, wait_frames_before, step1["left_target"], step1["right_target"], step1["target_yaw"]
        )

        # Solve first step
        solver1, success1 = solve_stepping_problem(gait, x0, step1["left_target"], step1["right_target"], step1["target_yaw"], verbose=False)

        if not success1 or solver1 is None:
            print("✗ Step 1 failed to converge")
            continue

        visualize_trajectory(display, solver1)

        # Extract first step data
        step1_data = extract_trajectory_data(robot, solver1, gait, step1["left_target"], step1["right_target"], step1["target_yaw"])

        # Compute actual foot positions after step 1
        # CRITICAL: Copy the state to avoid shared reference issues between solvers
        state_after_step1 = solver1.xs[-1].copy()
        q_after_step1 = state_after_step1[: robot.model.nq]

        rdata_step1 = robot.model.createData()
        pinocchio.forwardKinematics(robot.model, rdata_step1, q_after_step1)
        pinocchio.updateFramePlacements(robot.model, rdata_step1)

        lf_after_step1 = rdata_step1.oMf[gait.lfId].translation.copy()
        rf_after_step1 = rdata_step1.oMf[gait.rfId].translation.copy()

        # Compute step 2 targets based on actual positions after step 1
        if step2_disp["swing_foot"] == "left":
            lf_step2_target = rf_after_step1.copy()
            lf_step2_target[0] += step2_disp["dx"]
            lf_step2_target[1] += step2_disp["dy"]
            rf_step2_target = rf_after_step1.copy()
        else:
            rf_step2_target = lf_after_step1.copy()
            rf_step2_target[0] += step2_disp["dx"]
            rf_step2_target[1] += step2_disp["dy"]
            lf_step2_target = lf_after_step1.copy()

        print(f"Step 2: {step2_disp['swing_foot']} swings (stance: {step2_disp['stance_foot']})")
        print(f"  Left foot: {lf_after_step1[:2]} -> {lf_step2_target[:2]}")
        print(f"  Right foot: {rf_after_step1[:2]} -> {rf_step2_target[:2]}")

        # Generate middle waiting time
        wait_time_mid = np.random.uniform(MID_WAIT_TIME_RANGE[0], MID_WAIT_TIME_RANGE[1])
        wait_frames_mid = int(wait_time_mid / TIME_STEP)

        # Generate waiting frames between steps
        wait_data_mid = generate_waiting_frames(
            robot, gait, state_after_step1, wait_frames_mid, lf_step2_target, rf_step2_target, step2_disp["target_yaw"]
        )

        # Solve second step
        solver2, success2 = solve_stepping_problem(gait, state_after_step1, lf_step2_target, rf_step2_target, step2_disp["target_yaw"], verbose=False)

        if not success2 or solver2 is None:
            print("✗ Step 2 failed to converge")
            continue

        visualize_trajectory(display, solver2)

        # Extract second step data
        step2_data = extract_trajectory_data(robot, solver2, gait, lf_step2_target, rf_step2_target, step2_disp["target_yaw"])

        # Generate random waiting time after stepping
        wait_time_after = np.random.uniform(WAIT_TIME_RANGE[0], WAIT_TIME_RANGE[1])
        wait_frames_after = int(wait_time_after / TIME_STEP)

        # Generate waiting frames after second step
        final_state = solver2.xs[-1]
        wait_data_after = generate_waiting_frames(
            robot, gait, final_state, wait_frames_after, lf_step2_target, rf_step2_target, step2_disp["target_yaw"]
        )

        # Concatenate: waiting_before + step1 + waiting_mid + step2 + waiting_after
        all_q.append(wait_data_before["q"])
        all_q.append(step1_data["q"])
        all_q.append(wait_data_mid["q"])
        all_q.append(step2_data["q"])
        all_q.append(wait_data_after["q"])

        all_qd.append(wait_data_before["qd"])
        all_qd.append(step1_data["qd"])
        all_qd.append(wait_data_mid["qd"])
        all_qd.append(step2_data["qd"])
        all_qd.append(wait_data_after["qd"])

        all_T_blf.append(wait_data_before["T_blf"])
        all_T_blf.append(step1_data["T_blf"])
        all_T_blf.append(wait_data_mid["T_blf"])
        all_T_blf.append(step2_data["T_blf"])
        all_T_blf.append(wait_data_after["T_blf"])

        all_T_brf.append(wait_data_before["T_brf"])
        all_T_brf.append(step1_data["T_brf"])
        all_T_brf.append(wait_data_mid["T_brf"])
        all_T_brf.append(step2_data["T_brf"])
        all_T_brf.append(wait_data_after["T_brf"])

        all_T_stsw.append(wait_data_before["T_stsw"])
        all_T_stsw.append(step1_data["T_stsw"])
        all_T_stsw.append(wait_data_mid["T_stsw"])
        all_T_stsw.append(step2_data["T_stsw"])
        all_T_stsw.append(wait_data_after["T_stsw"])

        all_p_wcom.append(wait_data_before["p_wcom"])
        all_p_wcom.append(step1_data["p_wcom"])
        all_p_wcom.append(wait_data_mid["p_wcom"])
        all_p_wcom.append(step2_data["p_wcom"])
        all_p_wcom.append(wait_data_after["p_wcom"])

        all_T_wbase.append(wait_data_before["T_wbase"])
        all_T_wbase.append(step1_data["T_wbase"])
        all_T_wbase.append(wait_data_mid["T_wbase"])
        all_T_wbase.append(step2_data["T_wbase"])
        all_T_wbase.append(wait_data_after["T_wbase"])

        all_v_b.append(wait_data_before["v_b"])
        all_v_b.append(step1_data["v_b"])
        all_v_b.append(wait_data_mid["v_b"])
        all_v_b.append(step2_data["v_b"])
        all_v_b.append(wait_data_after["v_b"])

        all_cmd_footstep.append(wait_data_before["cmd_footstep"])
        all_cmd_footstep.append(step1_data["cmd_footstep"])
        all_cmd_footstep.append(wait_data_mid["cmd_footstep"])
        all_cmd_footstep.append(step2_data["cmd_footstep"])
        all_cmd_footstep.append(wait_data_after["cmd_footstep"])

        all_cmd_stance.append(wait_data_before["cmd_stance"])
        all_cmd_stance.append(step1_data["cmd_stance"])
        all_cmd_stance.append(wait_data_mid["cmd_stance"])
        all_cmd_stance.append(step2_data["cmd_stance"])
        all_cmd_stance.append(wait_data_after["cmd_stance"])

        all_cmd_countdown.append(wait_data_before["cmd_countdown"])
        all_cmd_countdown.append(step1_data["cmd_countdown"])
        all_cmd_countdown.append(wait_data_mid["cmd_countdown"])
        all_cmd_countdown.append(step2_data["cmd_countdown"])
        all_cmd_countdown.append(wait_data_after["cmd_countdown"])

        # Record next trajectory start index
        current_length = sum(len(q) for q in all_q)
        traj_starts.append(current_length)

        successful_samples += 1
        print(
            f"✓ Success! Wait before: {wait_frames_before}, Step1: {len(step1_data['q'])}, "
            + f"Wait mid: {wait_frames_mid}, Step2: {len(step2_data['q'])}, Wait after: {wait_frames_after}"
        )

        # CRITICAL: Explicit cleanup and garbage collection to prevent memory accumulation
        del solver1, solver2, rdata_step1, step1_data, step2_data, wait_data_before, wait_data_mid, wait_data_after
        gc.collect()

    # Remove last element (it's one past the end)
    traj_starts = traj_starts[:-1]

    # Concatenate all trajectories
    print("\n[4/4] Saving dataset...")
    q = np.vstack(all_q)
    qd = np.vstack(all_qd)
    T_blf = np.vstack(all_T_blf)
    T_brf = np.vstack(all_T_brf)
    T_stsw = np.vstack(all_T_stsw)
    p_wcom = np.vstack(all_p_wcom)
    T_wbase = np.vstack(all_T_wbase)
    v_b = np.vstack(all_v_b)
    cmd_footstep = np.vstack(all_cmd_footstep)
    cmd_stance = np.vstack(all_cmd_stance)
    cmd_countdown = np.vstack(all_cmd_countdown)
    traj = np.array(traj_starts, dtype=np.int32)

    # Save single NPZ file
    np.savez_compressed(
        OUTPUT_FILE,
        q=q,
        qd=qd,
        T_blf=T_blf,
        T_brf=T_brf,
        T_stsw=T_stsw,
        p_wcom=p_wcom,
        T_wbase=T_wbase,
        v_b=v_b,
        cmd_footstep=cmd_footstep,
        cmd_stance=cmd_stance,
        cmd_countdown=cmd_countdown,
        traj=traj,
        traj_dt=TIME_STEP,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("Dataset Generation Complete!")
    print("=" * 80)
    print(f"Successful trajectories: {successful_samples}/{len(grid_samples)}")
    print(f"Total timesteps: {len(q)}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Random init samples: {RAND_INIT_X_SAMPLES}x{RAND_INIT_Y_SAMPLES}x{RAND_INIT_YAW_SAMPLES}")
    print(f"Step 1 grid: {GRID_X_STEPS}x{GRID_Y_STEPS}x{GRID_YAW_STEPS}")
    print(f"Step 2 grid: {GRID_X_STEPS_2}x{GRID_Y_STEPS_2}x{GRID_YAW_STEPS_2}")
    print(f"Step height: {STEP_HEIGHT:.2f} m")
    print(f"Wait time range (before & after): {WAIT_TIME_RANGE[0]:.2f} - {WAIT_TIME_RANGE[1]:.2f} s")
    print(f"Mid wait time range (between steps): {MID_WAIT_TIME_RANGE[0]:.2f} - {MID_WAIT_TIME_RANGE[1]:.2f} s")
    print("\nDataset contents:")
    print(f"  q:             {q.shape}")
    print(f"  qd:            {qd.shape}")
    print(f"  T_blf:         {T_blf.shape} (x, y, z, yaw)")
    print(f"  T_brf:         {T_brf.shape} (x, y, z, yaw)")
    print(f"  T_stsw:        {T_stsw.shape} (x, y, z, yaw)")
    print(f"  p_wcom:        {p_wcom.shape} (x, y, z)")
    print(f"  T_wbase:       {T_wbase.shape} (x, y, z, qw, qx, qy, qz)")
    print(f"  v_b:           {v_b.shape} (linear xyz, angular xyz)")
    print(f"  cmd_footstep:  {cmd_footstep.shape} (x, y, z, yaw)")
    print(f"  cmd_stance:    {cmd_stance.shape} (0 lf stance, 1 rf stance)")
    print(f"  cmd_countdown: {cmd_countdown.shape} (0 during wait, 1->0 during step)")
    print(f"  traj:          {traj.shape} (trajectory start indices)")
    print(f"  traj_dt:       {TIME_STEP:.6f} (time step)")
    print("=" * 80)


if __name__ == "__main__":
    main()
