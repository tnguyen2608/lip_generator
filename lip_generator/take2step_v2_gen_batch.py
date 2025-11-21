"""
Batch generation script for two-step motion dataset.

Generates a single NPZ file containing trajectories with two consecutive steps:
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
import matplotlib.pyplot as plt
from step import SimpleBipedGaitProblem
import psutil
import gc
try:
    import meshcat
    import meshcat.transformations as tf
    MESHCAT_AVAILABLE = True
except ImportError:
    MESHCAT_AVAILABLE = False

# Configuration
OUTPUT_FILE = "stepping_dataset.npz"
NUM_KNOTS = 25
TIME_STEP = 0.02
STEP_KNOTS = NUM_KNOTS
SUPPORT_KNOTS = 10
WITHDISPLAY = False

# Step generation parameters
STEP_HEIGHT = 0.15  # Step height in meters
WAIT_TIME_RANGE = (0.5, 0.7)  # Waiting period before step (seconds)
MID_WAIT_TIME_RANGE = (0.3, 0.6)  # Waiting period between two steps (seconds)
# Grid sampling parameters
GRID_X_STEPS = 5  # Number of steps in x direction
GRID_Y_STEPS = 3  # Number of steps in y direction
GRID_YAW_STEPS = 3  # Number of steps in yaw direction
XY_STEP_UNIT = 0.1
Y_OFFSET = 0.15
YAW_STEP_UNIT = 0.1

# Solver parameters
MAX_ITERATIONS = 100  # Reduced to avoid divergence
SOLVER_THRESHOLD = 1e-6  # Relaxed threshold to exit early


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert to MB


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
            -0.2,
            0,
            0,
            0.4,
            -0.25,
            0,  # left leg
            -0.2,
            0,
            0,
            0.4,
            -0.25,
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


def generate_grid_samples(
    lfPos0, rfPos0, grid_x_steps, grid_y_steps, grid_yaw_steps, xy_step_unit, y_offset, yaw_step_unit
):
    samples = []
    x_values_right = np.arange(-grid_x_steps/2, grid_x_steps/2) * xy_step_unit 
    y_values_right = -np.arange(grid_y_steps) * xy_step_unit - y_offset
    yaw_values_right = np.arange(-grid_yaw_steps/2, grid_yaw_steps/2) * yaw_step_unit

    #left swing, respected to right stance
    x_values_left = np.arange(-grid_x_steps/2, grid_x_steps/2) * xy_step_unit 
    y_values_left = np.arange(grid_y_steps) * xy_step_unit + y_offset
    yaw_values_left = np.arange(-grid_yaw_steps/2, grid_yaw_steps/2) * yaw_step_unit
    # First sequence: Right foot swings, then left foot swings
    for dx1 in x_values_right:
        for dy1 in y_values_right:
            for yaw1 in yaw_values_right:
                # Step 1: Right foot swings
                rf_step1 = lfPos0.copy()
                rf_step1[0] += dx1
                rf_step1[1] += dy1

                # Step 2: Left foot swings (displacement will be applied after step 1)
                for dx2 in x_values_left:
                    for dy2 in y_values_left:
                        for yaw2 in yaw_values_left:
                            samples.append(
                                {
                                    "step1": {
                                        "left_target": lfPos0.copy(),
                                        "right_target": rf_step1.copy(),
                                        "stance_foot": "left",
                                        "swing_foot": "right",
                                        "target_yaw": yaw1,
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

    # Second sequence: Left foot swings, then right foot swings
    for dx1 in x_values_left:
        for dy1 in y_values_left:
            for yaw1 in yaw_values_left:
                # Step 1: Left foot swings
                lf_step1 = rfPos0.copy()
                lf_step1[0] += dx1
                lf_step1[1] += dy1

                # Step 2: Right foot swings (displacement will be applied after step 1)
                for dx2 in x_values_right:
                    for dy2 in y_values_right:
                        for yaw2 in yaw_values_right:
                            samples.append(
                                {
                                    "step1": {
                                        "left_target": lf_step1.copy(),
                                        "right_target": rfPos0.copy(),
                                        "stance_foot": "right",
                                        "swing_foot": "left",
                                        "target_yaw": yaw1,
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
    """Solve a stepping problem with extra safety checks."""
    try:
        # Validate initial state
        nq = gait.rmodel.nq
        if len(x0) != gait.state.nx:
            return None, False

        q0 = x0[:nq]
        if np.any(np.isnan(q0)) or np.any(np.isinf(q0)):
            return None, False

        # Validate targets are reasonable (not NaN, not too far)
        left_target = np.array(left_target)
        right_target = np.array(right_target)

        if np.any(np.isnan(left_target)) or np.any(np.isnan(right_target)):
            return None, False

        problem = gait.createSingleStepProblem(x0, left_target, right_target, TIME_STEP, STEP_KNOTS, SUPPORT_KNOTS, STEP_HEIGHT, target_yaw)

        solver = crocoddyl.SolverIntro(problem)
        solver.th_stop = SOLVER_THRESHOLD

        # Only add verbose callback if requested
        if verbose:
            solver.setCallbacks([crocoddyl.CallbackVerbose()])

        # Create independent copies of x0 for each time step to avoid memory issues
        xs = [x0.copy() for _ in range(solver.problem.T + 1)]
        us = solver.problem.quasiStatic([x0.copy() for _ in range(solver.problem.T)])

        # Solve the problem
        solver.solve(xs, us, MAX_ITERATIONS, False)

        success = solver.stop < SOLVER_THRESHOLD
        return solver, success
    except (RuntimeError, ValueError, ZeroDivisionError):
        return None, False
    except Exception:
        return None, False


def generate_waiting_frames(robot, gait, x0, num_frames, left_target, right_target, yaw_target):
    """
    Generate waiting frames where the robot stands still.

    Returns:
        dict with keys: q, qd, T_blf, T_brf, T_stsw, p_wcom, T_wbase, v_b, cmd_footstep, cmd_stance, cmd_countdown
    """
    nq = robot.model.nq
    nv = robot.model.nv

    q_data = np.tile(x0[:nq], (num_frames, 1))
    qd_data = np.zeros((num_frames, nv))
    T_blf_data = np.zeros((num_frames, 4))
    T_brf_data = np.zeros((num_frames, 4))
    T_stsw_data = np.zeros((num_frames, 4))
    p_wcom_data = np.zeros((num_frames, 3))
    T_wbase_data = np.zeros((num_frames, 7))
    # Base velocity in base frame (linear, angular)
    v_b_data = np.zeros((num_frames, 6))
    cmd_footstep_data = np.zeros((num_frames, 4))
    cmd_stance_data = np.zeros((num_frames, 1))
    cmd_countdown_data = np.zeros((num_frames, 1))  # All zeros during waiting

    # Compute kinematics for the static pose
    rdata = robot.model.createData()
    q = x0[:nq]

    # Determine stance foot
    pinocchio.forwardKinematics(robot.model, rdata, q)
    pinocchio.updateFramePlacements(robot.model, rdata)

    lf_init = rdata.oMf[gait.lfId].translation.copy()
    rf_init = rdata.oMf[gait.rfId].translation.copy()

    left_movement = np.linalg.norm(left_target - lf_init)
    right_movement = np.linalg.norm(right_target - rf_init)

    if left_movement > right_movement:
        stance_is_left = 0  # Right foot is stance
        swing_target = lf_init
        stance_foot_id = gait.rfId
        swing_foot_id = gait.lfId
    else:
        stance_is_left = 1  # Left foot is stance
        swing_target = rf_init
        stance_foot_id = gait.lfId
        swing_foot_id = gait.rfId

    # Get body transformation
    body_pos = q[:3]
    body_quat = q[3:7]
    body_R = pinocchio.Quaternion(body_quat[3], body_quat[0], body_quat[1], body_quat[2]).toRotationMatrix()

    # Get foot transforms
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

    # CoM
    com_world = pinocchio.centerOfMass(robot.model, rdata, q)

    # Stance and swing foot poses
    stance_pos = rdata.oMf[stance_foot_id].translation
    stance_R = rdata.oMf[stance_foot_id].rotation
    swing_pos = rdata.oMf[swing_foot_id].translation
    swing_R = rdata.oMf[swing_foot_id].rotation

    swing_stance_pos = stance_R.T @ (swing_pos - stance_pos)
    swing_stance_R = stance_R.T @ swing_R
    swing_stance_yaw = rotation_matrix_to_yaw(swing_stance_R)

    # Footstep command
    cmd_footstep = transform_to_stance_frame(swing_target, stance_pos, stance_R, swing_stance_yaw)

    # Fill all frames with the same data
    for i in range(num_frames):
        T_blf_data[i] = [lf_body_pos[0], lf_body_pos[1], lf_body_pos[2], lf_body_yaw]
        T_brf_data[i] = [rf_body_pos[0], rf_body_pos[1], rf_body_pos[2], rf_body_yaw]
        T_stsw_data[i] = [swing_stance_pos[0], swing_stance_pos[1], swing_stance_pos[2], swing_stance_yaw]
        p_wcom_data[i] = com_world
        T_wbase_data[i] = [body_pos[0], body_pos[1], body_pos[2], body_quat[3], body_quat[0], body_quat[1], body_quat[2]]
        cmd_footstep_data[i] = cmd_footstep
        cmd_stance_data[i, 0] = 0 if stance_is_left else 1

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
    """
    Extract trajectory data in required format.

    Returns:
        dict with keys: q, qd, T_blf, T_brf, T_stsw, p_wcom, T_wbase, v_b, cmd_footstep, cmd_stance, cmd_countdown
    """
    T = len(solver.xs)
    nq = robot.model.nq
    nv = robot.model.nv

    q_data = np.zeros((T, nq))
    qd_data = np.zeros((T, nv))
    T_blf_data = np.zeros((T, 4))  # (x, y, z, yaw)
    T_brf_data = np.zeros((T, 4))  # (x, y, z, yaw)
    T_stsw_data = np.zeros((T, 4))  # (x, y, z, yaw) stance to swing
    p_wcom_data = np.zeros((T, 3))
    T_wbase_data = np.zeros((T, 7))  # (x, y, z, qw, qx, qy, qz)
    # Base velocity in base frame (linear, angular)
    v_b_data = np.zeros((T, 6))
    cmd_footstep_data = np.zeros((T, 4))
    cmd_stance_data = np.zeros((T, 1))
    cmd_countdown_data = np.zeros((T, 1))

    # Determine which foot is moving (stance foot is the one NOT moving)
    rdata = robot.model.createData()
    pinocchio.forwardKinematics(robot.model, rdata, solver.xs[0][:nq])
    pinocchio.updateFramePlacements(robot.model, rdata)

    lf_init = rdata.oMf[gait.lfId].translation.copy()
    rf_init = rdata.oMf[gait.rfId].translation.copy()

    left_movement = np.linalg.norm(left_target - lf_init)
    right_movement = np.linalg.norm(right_target - rf_init)

    # Stance foot: 0 = left, 1 = right
    # If left foot moves more, left is swing, right is stance
    if left_movement > right_movement:
        stance_is_left = 0  # Right foot is stance
        swing_target = left_target
        stance_foot_id = gait.rfId
        swing_foot_id = gait.lfId
    else:
        stance_is_left = 1  # Left foot is stance
        swing_target = right_target
        stance_foot_id = gait.lfId
        swing_foot_id = gait.rfId

    for t in range(T):
        q = solver.xs[t][:nq]
        qd = solver.xs[t][nq:]

        # Store q and qd
        q_data[t] = q
        qd_data[t] = qd

        # Compute forward kinematics
        pinocchio.forwardKinematics(robot.model, rdata, q)
        pinocchio.updateFramePlacements(robot.model, rdata)

        # Get body (base) transformation
        body_pos = q[:3]
        body_quat = q[3:7]  # [x, y, z, w] in pinocchio
        # Convert quaternion to rotation matrix
        body_R = pinocchio.Quaternion(body_quat[3], body_quat[0], body_quat[1], body_quat[2]).toRotationMatrix()

        # Get foot transforms in world frame
        lf_world_pos = rdata.oMf[gait.lfId].translation
        rf_world_pos = rdata.oMf[gait.rfId].translation
        lf_world_R = rdata.oMf[gait.lfId].rotation
        rf_world_R = rdata.oMf[gait.rfId].rotation

        # Transform to body frame: p_body_to_foot = R_body^T @ (p_foot - p_body)
        lf_body_pos = body_R.T @ (lf_world_pos - body_pos)
        rf_body_pos = body_R.T @ (rf_world_pos - body_pos)

        # Get yaw angles from rotation matrices
        lf_body_R = body_R.T @ lf_world_R
        rf_body_R = body_R.T @ rf_world_R
        lf_body_yaw = rotation_matrix_to_yaw(lf_body_R)
        rf_body_yaw = rotation_matrix_to_yaw(rf_body_R)

        # Compute base velocity in base frame
        # qd[:3] is linear velocity in world frame, qd[3:6] is angular velocity in world frame
        v_world_linear = qd[:3]
        v_world_angular = qd[3:6]
        v_b_linear = body_R.T @ v_world_linear
        v_b_angular = body_R.T @ v_world_angular
        v_b_data[t] = np.concatenate([v_b_linear, v_b_angular])

        # Store body to foot transforms (x, y, z, yaw)
        T_blf_data[t] = np.array([lf_body_pos[0], lf_body_pos[1], lf_body_pos[2], lf_body_yaw])
        T_brf_data[t] = np.array([rf_body_pos[0], rf_body_pos[1], rf_body_pos[2], rf_body_yaw])

        # Compute CoM position in world frame
        com_world = pinocchio.centerOfMass(robot.model, rdata, q)
        p_wcom_data[t] = com_world

        # Store base transform in world frame (x, y, z, qw, qx, qy, qz)
        T_wbase_data[t] = np.array([body_pos[0], body_pos[1], body_pos[2], body_quat[3], body_quat[0], body_quat[1], body_quat[2]])

        # Get stance and swing foot poses
        stance_pos = rdata.oMf[stance_foot_id].translation
        stance_R = rdata.oMf[stance_foot_id].rotation
        swing_pos = rdata.oMf[swing_foot_id].translation
        swing_R = rdata.oMf[swing_foot_id].rotation

        # Compute stance to swing transform (x, y, z, yaw)
        swing_stance_pos = stance_R.T @ (swing_pos - stance_pos)
        swing_stance_R = stance_R.T @ swing_R
        swing_stance_yaw = rotation_matrix_to_yaw(swing_stance_R)
        T_stsw_data[t] = np.array([swing_stance_pos[0], swing_stance_pos[1], swing_stance_pos[2], swing_stance_yaw])

        # Compute cmd_footstep in stance frame
        cmd_footstep_data[t] = transform_to_stance_frame(swing_target, stance_pos, stance_R, yaw_target)

        # Stance indicator
        cmd_stance_data[t, 0] = 0 if stance_is_left else 1

        # Countdown: goes from 1 -> 0 uniformly throughout the step
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


def extract_com_from_trajectory(robot, q_trajectory, start_idx, end_idx):
    """Extract COM position from trajectory segment."""
    com_trajectory = []
    rdata = robot.model.createData()

    for i in range(start_idx, end_idx):
        q = q_trajectory[i]
        pinocchio.forwardKinematics(robot.model, rdata, q)
        com = pinocchio.centerOfMass(robot.model, rdata, q)
        com_trajectory.append(com)

    return np.array(com_trajectory)


def extract_feet_from_trajectory(robot, q_trajectory, start_idx, end_idx, foot_frame):
    """Extract foot position from trajectory segment."""
    foot_trajectory = []
    rdata = robot.model.createData()
    foot_id = robot.model.getFrameId(foot_frame)

    for i in range(start_idx, end_idx):
        q = q_trajectory[i]
        pinocchio.forwardKinematics(robot.model, rdata, q)
        pinocchio.updateFramePlacements(robot.model, rdata)
        foot_pos = rdata.oMf[foot_id].translation.copy()
        foot_trajectory.append(foot_pos)

    return np.array(foot_trajectory)


def plot_com_trajectory(robot, q_trajectory, traj_start, traj_end, traj_idx):
    """Plot COM trajectory for a single trajectory segment."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"COM Trajectory Analysis - Trajectory {traj_idx}", fontsize=14, fontweight='bold')

    com_traj = extract_com_from_trajectory(robot, q_trajectory, traj_start, traj_end)
    lf_traj = extract_feet_from_trajectory(robot, q_trajectory, traj_start, traj_end, "left_foot_link")
    rf_traj = extract_feet_from_trajectory(robot, q_trajectory, traj_start, traj_end, "right_foot_link")

    time_steps = np.arange(len(com_traj))

    # XY plane view (top-down)
    ax = axes[0, 0]
    ax.plot(com_traj[:, 0], com_traj[:, 1], 'b-', linewidth=2, label='COM')
    ax.plot(lf_traj[:, 0], lf_traj[:, 1], 'r--', linewidth=2, label='Left Foot')
    ax.plot(rf_traj[:, 0], rf_traj[:, 1], 'g--', linewidth=2, label='Right Foot')
    ax.plot(com_traj[0, 0], com_traj[0, 1], 'bo', markersize=8, label='Start')
    ax.plot(com_traj[-1, 0], com_traj[-1, 1], 'bs', markersize=8, label='End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Top-Down View (XY Plane)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # X position over time
    ax = axes[0, 1]
    ax.plot(time_steps, com_traj[:, 0], 'b-', linewidth=2, label='COM X')
    ax.plot(time_steps, lf_traj[:, 0], 'r--', linewidth=1.5, alpha=0.7, label='LF X')
    ax.plot(time_steps, rf_traj[:, 0], 'g--', linewidth=1.5, alpha=0.7, label='RF X')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('X Position (m)')
    ax.set_title('Forward/Backward Motion')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Y position over time (Lateral Lean - CRITICAL)
    ax = axes[1, 0]
    ax.plot(time_steps, com_traj[:, 1], 'b-', linewidth=2.5, label='COM Y')
    ax.plot(time_steps, lf_traj[:, 1], 'r--', linewidth=1.5, alpha=0.7, label='LF Y')
    ax.plot(time_steps, rf_traj[:, 1], 'g--', linewidth=1.5, alpha=0.7, label='RF Y')
    ax.fill_between(time_steps, lf_traj[:, 1], rf_traj[:, 1], alpha=0.1, color='gray')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Lateral Lean (CRITICAL)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Z position over time (Height)
    ax = axes[1, 1]
    ax.plot(time_steps, com_traj[:, 2], 'b-', linewidth=2, label='COM Z')
    ax.plot(time_steps, lf_traj[:, 2], 'r--', linewidth=1.5, alpha=0.7, label='LF Z')
    ax.plot(time_steps, rf_traj[:, 2], 'g--', linewidth=1.5, alpha=0.7, label='RF Z')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Height (m)')
    ax.set_title('Vertical Motion')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Main batch generation loop."""
    print("=" * 80)
    print("Stepping Motion Dataset Generator")
    print("=" * 80)

    # Load robot
    print("\n[1/4] Loading robot model...")
    robot = load_robot()

    # Initial state
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

    # Create display (if visualization enabled)
    display = create_display(robot)
    if display is not None:
        print("Visualization enabled")

    # Generate grid samples
    print("\n[3/4] Generating grid samples...")
    grid_samples = generate_grid_samples(
        lfPos0, rfPos0, GRID_X_STEPS, GRID_Y_STEPS, GRID_YAW_STEPS, XY_STEP_UNIT, Y_OFFSET, YAW_STEP_UNIT
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
    traj_starts = [0]  # First trajectory starts at index 0

    successful_samples = 0
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")

    for i, sample in enumerate(grid_samples):
        # Recreate gait EVERY iteration to completely prevent state accumulation
        del gait
        gait = SimpleBipedGaitProblem(robot.model, rightFoot, leftFoot, fwddyn=False)

        # Track memory for every sample
        current_memory = get_memory_usage()
        memory_delta = current_memory - initial_memory
        print(f"\n[Memory] Sample {i}: {current_memory:.2f} MB (Δ{memory_delta:+.2f} MB)")

        print(f"--- Sample {i + 1}/{len(grid_samples)} ---")

        step1 = sample["step1"]
        step2_disp = sample["step2_displacement"]

        print(f"Step 1: {step1['swing_foot']} swings (stance: {step1['stance_foot']})")
        print(f"  Left foot: {lfPos0[:2]} -> {step1['left_target'][:2]}")
        print(f"  Right foot: {rfPos0[:2]} -> {step1['right_target'][:2]}")

        # Generate random waiting time and frames (before stepping)
        wait_time_before = np.random.uniform(WAIT_TIME_RANGE[0], WAIT_TIME_RANGE[1])
        wait_frames_before = int(wait_time_before / TIME_STEP)

        # Generate waiting frames before first step
        try:
            wait_data_before = generate_waiting_frames(
                robot, gait, x0, wait_frames_before, step1["left_target"], step1["right_target"], step1["target_yaw"]
            )
        except Exception as e:
            print(f"✗ Failed to generate wait_data_before: {e}")
            continue

        # Solve first step
        solver1, success1 = solve_stepping_problem(gait, x0, step1["left_target"], step1["right_target"], step1["target_yaw"], verbose=False)

        if not success1 or solver1 is None:
            # Track memory on failure
            if (i + 1) % 100 == 0:
                print(f"  [Memory on failure] {get_memory_usage():.2f} MB")
            continue

        # Visualize first step
        visualize_trajectory(display, solver1)

        # Extract first step data
        try:
            step1_data = extract_trajectory_data(robot, solver1, gait, step1["left_target"], step1["right_target"], step1["target_yaw"])
        except Exception as e:
            print(f"✗ Failed to extract step1 data: {e}")
            del solver1
            continue

        # Compute actual foot positions after step 1
        state_after_step1 = solver1.xs[-1].copy()  # Copy to avoid reference to solver
        q_after_step1 = state_after_step1[: robot.model.nq]

        # Clean up solver1 to free memory
        del solver1

        rdata_step1 = robot.model.createData()
        pinocchio.forwardKinematics(robot.model, rdata_step1, q_after_step1)
        pinocchio.updateFramePlacements(robot.model, rdata_step1)

        lf_after_step1 = rdata_step1.oMf[gait.lfId].translation.copy()
        rf_after_step1 = rdata_step1.oMf[gait.rfId].translation.copy()

        # Compute step 2 targets based on actual positions after step 1
        if step2_disp["swing_foot"] == "left":
            # Left foot swings, right foot is stance
            lf_step2_target = rf_after_step1.copy()
            lf_step2_target[0] += step2_disp["dx"]
            lf_step2_target[1] += step2_disp["dy"]
            rf_step2_target = rf_after_step1.copy()  # Stance foot stays
        else:
            # Right foot swings, left foot is stance
            rf_step2_target = lf_after_step1.copy()
            rf_step2_target[0] += step2_disp["dx"]
            rf_step2_target[1] += step2_disp["dy"]
            lf_step2_target = lf_after_step1.copy()  # Stance foot stays

        print(f"Step 2: {step2_disp['swing_foot']} swings (stance: {step2_disp['stance_foot']})")
        print(f"  Left foot: {lf_after_step1[:2]} -> {lf_step2_target[:2]}")
        print(f"  Right foot: {rf_after_step1[:2]} -> {rf_step2_target[:2]}")

        # Generate middle waiting time and frames (between steps)
        wait_time_mid = np.random.uniform(MID_WAIT_TIME_RANGE[0], MID_WAIT_TIME_RANGE[1])
        wait_frames_mid = int(wait_time_mid / TIME_STEP)

        # Generate waiting frames between steps (using final state from step 1)
        try:
            wait_data_mid = generate_waiting_frames(
                robot, gait, state_after_step1, wait_frames_mid, lf_step2_target, rf_step2_target, step2_disp["target_yaw"]
            )
        except Exception as e:
            print(f"✗ Failed to generate wait_data_mid: {e}")
            continue

        # Solve second step (starting from final state of step 1)
        solver2, success2 = solve_stepping_problem(gait, state_after_step1, lf_step2_target, rf_step2_target, step2_disp["target_yaw"], verbose=False)

        if not success2 or solver2 is None:
            continue

        # Visualize second step
        visualize_trajectory(display, solver2)

        # Extract second step data
        try:
            step2_data = extract_trajectory_data(robot, solver2, gait, lf_step2_target, rf_step2_target, step2_disp["target_yaw"])
        except Exception as e:
            print(f"✗ Failed to extract step2 data: {e}")
            del solver2
            continue

        # Generate random waiting time and frames (after second step)
        wait_time_after = np.random.uniform(WAIT_TIME_RANGE[0], WAIT_TIME_RANGE[1])
        wait_frames_after = int(wait_time_after / TIME_STEP)

        # Generate waiting frames after second step (using final state from step 2)
        final_state = solver2.xs[-1].copy()  # Copy to avoid reference to solver
        try:
            wait_data_after = generate_waiting_frames(
                robot, gait, final_state, wait_frames_after, lf_step2_target, rf_step2_target, step2_disp["target_yaw"]
            )
        except Exception as e:
            print(f"✗ Failed to generate wait_data_after: {e}")
            del solver2
            continue

        # Clean up solver2 to free memory
        del solver2

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

        # Force garbage collection EVERY successful sample to free memory aggressively
        gc.collect()

        # Report memory every 50 samples
        if successful_samples % 50 == 0:
            current_memory = get_memory_usage()
            memory_delta = current_memory - initial_memory
            print(f"  [Memory after GC] {current_memory:.2f} MB (Δ{memory_delta:+.2f} MB)")

        # Optionally plot this trajectory (every Nth sample to avoid too many plots)
        if successful_samples % 10 == 0:  # Plot every 10th successful sample
            print(f"  Plotting trajectory {successful_samples}...")
            # Combine all data for this trajectory
            traj_q = np.vstack([
                wait_data_before["q"], step1_data["q"], wait_data_mid["q"],
                step2_data["q"], wait_data_after["q"]
            ])
            # Extract COM and feet
            com_traj = extract_com_from_trajectory(robot, traj_q, 0, len(traj_q))
            lf_traj = extract_feet_from_trajectory(robot, traj_q, 0, len(traj_q), "left_foot_link")
            rf_traj = extract_feet_from_trajectory(robot, traj_q, 0, len(traj_q), "right_foot_link")
        
            # Plot
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"COM Trajectory - Sample {i+1}, Successful #{successful_samples}", fontsize=14, fontweight='bold')
            time_steps = np.arange(len(com_traj))
        
            # XY plane
            ax = axes[0, 0]
            ax.plot(com_traj[:, 0], com_traj[:, 1], 'b-', linewidth=2, label='COM')
            ax.plot(lf_traj[:, 0], lf_traj[:, 1], 'r--', linewidth=2, label='Left Foot')
            ax.plot(rf_traj[:, 0], rf_traj[:, 1], 'g--', linewidth=2, label='Right Foot')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title('Top-Down View')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
        
            # Y position (lateral lean)
            ax = axes[1, 0]
            ax.plot(time_steps, com_traj[:, 1], 'b-', linewidth=2.5, label='COM Y')
            ax.plot(time_steps, lf_traj[:, 1], 'r--', linewidth=1.5, alpha=0.7, label='LF Y')
            ax.plot(time_steps, rf_traj[:, 1], 'g--', linewidth=1.5, alpha=0.7, label='RF Y')
            ax.fill_between(time_steps, lf_traj[:, 1], rf_traj[:, 1], alpha=0.1, color='gray')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Y Position (m)')
            ax.set_title('Lateral Lean (CRITICAL)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
            # Z position (height)
            ax = axes[1, 1]
            ax.plot(time_steps, com_traj[:, 2], 'b-', linewidth=2, label='COM Z')
            ax.plot(time_steps, lf_traj[:, 2], 'r--', linewidth=1.5, alpha=0.7, label='LF Z')
            ax.plot(time_steps, rf_traj[:, 2], 'g--', linewidth=1.5, alpha=0.7, label='RF Z')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Height (m)')
            ax.set_title('Vertical Motion')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
            # X position
            ax = axes[0, 1]
            ax.plot(time_steps, com_traj[:, 0], 'b-', linewidth=2, label='COM X')
            ax.plot(time_steps, lf_traj[:, 0], 'r--', linewidth=1.5, alpha=0.7, label='LF X')
            ax.plot(time_steps, rf_traj[:, 0], 'g--', linewidth=1.5, alpha=0.7, label='RF X')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('X Position (m)')
            ax.set_title('Forward/Backward Motion')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
            plt.tight_layout()
            plot_file = f"com_trajectory_sample_{i+1:04d}_success_{successful_samples:04d}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"  Saved: {plot_file}")
            plt.close(fig)

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
    print(f"Grid configuration: {GRID_X_STEPS}x{GRID_Y_STEPS}x{GRID_YAW_STEPS} (x × y × yaw)")
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

    # Plot random trajectory COM
    if successful_samples > 0:
        print("\nPlotting random trajectory COM...")
        random_traj_idx = np.random.randint(0, len(traj))
        traj_start = traj[random_traj_idx]
        traj_end = traj[random_traj_idx + 1] if random_traj_idx + 1 < len(traj) else len(q)
    
        fig = plot_com_trajectory(robot, q, traj_start, traj_end, random_traj_idx)
        plot_filename = f"com_trajectory_random_traj_{random_traj_idx}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Saved random trajectory plot: {plot_filename}")
        plt.close(fig)


if __name__ == "__main__":
    main()
