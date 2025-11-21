"""
Batch generation script for stepping motion dataset.

Generates a single NPZ file containing:
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
from step import SimpleBipedGaitProblem

# Configuration
OUTPUT_FILE = "stepping_dataset.npz"
NUM_KNOTS = 30
TIME_STEP = 0.02
STEP_KNOTS = NUM_KNOTS
SUPPORT_KNOTS = 2
WITHDISPLAY = True

# Step generation parameters
STEP_HEIGHT = 0.35  # Step height in meters
WAIT_TIME_RANGE = (0.2, 0.4)  # Waiting period before step (seconds)
LEFT_RAND_RANGE = {
    "x": (-0.2, 0.2),  # Forward/backward range (meters)
    "y": (-0.02, 0.02),  # Lateral range (meters)
    "yaw": (-0.2, 0.2),
}
RIGHT_RAND_RANGE = {
    "x": (-0.2, 0.2),  # Forward/backward range (meters)
    "y": (-0.02, 0.02),  # Lateral range (meters)
    "yaw": (-0.2, 0.2),
}
LEFT_SWING_RANGE = {
    "x": (-0.2, 0.2),  # Forward/backward range (meters)
    "y": (0.2, 0.25),  # Lateral range (meters)
    "yaw": (-0.2, 0.2),
}
RIGHT_SWING_RANGE = {
    "x": (-0.2, 0.2),  # Forward/backward range (meters)
    "y": (-0.25, -0.2),  # Lateral range (meters)
    "yaw": (-0.2, 0.2),
}

# Grid sampling parameters
GRID_X_STEPS = 5  # Number of steps in x direction
GRID_Y_STEPS = 3  # Number of steps in y direction
GRID_YAW_STEPS = 3  # Number of steps in yaw direction

# Solver parameters
MAX_ITERATIONS = 100
SOLVER_THRESHOLD = 1e-7


def load_robot():
    """Load the robot model and set up initial configuration."""
    robot = pinocchio.RobotWrapper.BuildFromURDF(
        "model/T1_7dof_arms_with_gripper.urdf",
        package_dirs=["model"],
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


def generate_grid_samples(
    lfPos0, rfPos0, left_rand_range, right_rand_range, left_swing_range, right_swing_range, grid_x_steps, grid_y_steps, grid_yaw_steps
):
    """
    Generate a grid of stepping trajectories from randomized starting poses to target end poses.

    For each starting pose (generated from rand_range), generate trajectories to multiple
    end poses (generated from swing_range).

    Args:
        lfPos0: Initial left foot position (half-sitting)
        rfPos0: Initial right foot position (half-sitting)
        left_rand_range: Range dict for left foot starting position variation {'x', 'y', 'yaw'}
        right_rand_range: Range dict for right foot starting position variation {'x', 'y', 'yaw'}
        left_swing_range: Range dict for left foot target positions (relative to start) {'x', 'y', 'yaw'}
        right_swing_range: Range dict for right foot target positions (relative to start) {'x', 'y', 'yaw'}
        grid_x_steps: Number of grid points in x direction
        grid_y_steps: Number of grid points in y direction
        grid_yaw_steps: Number of grid points in yaw direction

    Returns:
        List of dicts with 'left_start', 'right_start', 'left_target', 'right_target', 'stance_foot', 'swing_foot', 'target_yaw'
    """
    samples = []

    # Generate grid of starting poses for left foot
    lf_start_x = np.linspace(left_rand_range["x"][0], left_rand_range["x"][1], max(2, grid_x_steps // 3))
    lf_start_y = np.linspace(left_rand_range["y"][0], left_rand_range["y"][1], max(2, grid_y_steps // 3))
    lf_start_yaw = np.linspace(left_rand_range["yaw"][0], left_rand_range["yaw"][1], max(2, grid_yaw_steps // 2))

    # Generate grid of starting poses for right foot
    rf_start_x = np.linspace(right_rand_range["x"][0], right_rand_range["x"][1], max(2, grid_x_steps // 3))
    rf_start_y = np.linspace(right_rand_range["y"][0], right_rand_range["y"][1], max(2, grid_y_steps // 3))
    rf_start_yaw = np.linspace(right_rand_range["yaw"][0], right_rand_range["yaw"][1], max(2, grid_yaw_steps // 2))
    
    # Generate target swing ranges
    x_values_swing = np.linspace(right_swing_range["x"][0], right_swing_range["x"][1], grid_x_steps)
    y_values_swing = np.linspace(right_swing_range["y"][0], right_swing_range["y"][1], grid_y_steps)
    yaw_values_swing = np.linspace(right_swing_range["yaw"][0], right_swing_range["yaw"][1], grid_yaw_steps)

    # print("init",lfPos0, rfPos0)
    # For each left foot starting position (right foot swings)
    print(len(lf_start_x), len(lf_start_y), len(lf_start_yaw))
    for lf_dx_start in lf_start_x:
        for lf_dy_start in lf_start_y:
            for lf_yaw_start in lf_start_yaw:
                # Compute starting positions with yaw offset
                left_start = lfPos0.copy()
                left_start[0] += lf_dx_start
                left_start[1] += lf_dy_start

                # Apply yaw rotation to right foot starting position relative to left foot
                # Rotate rfPos0 relative to lfPos0 by lf_yaw_start
                rel_rf = rfPos0 - lfPos0  # Right foot relative to left foot
                cos_yaw = np.cos(lf_yaw_start)
                sin_yaw = np.sin(lf_yaw_start)
                rotated_rf = np.array([cos_yaw * rel_rf[0] - sin_yaw * rel_rf[1], sin_yaw * rel_rf[0] + cos_yaw * rel_rf[1], rel_rf[2]])
                right_start = left_start + rotated_rf
                print("left_start", left_start, "right_start", right_start)

                # For each right foot target position (relative to left start)
                for dx_swing in x_values_swing:
                    for dy_swing in y_values_swing:
                        for yaw_swing in yaw_values_swing:
                            # Rotate swing target by starting yaw
                            swing_rel = np.array([dx_swing, dy_swing, 0.0])
                            rotated_swing = np.array(
                                [cos_yaw * swing_rel[0] - sin_yaw * swing_rel[1], sin_yaw * swing_rel[0] + cos_yaw * swing_rel[1], swing_rel[2]]
                            )
                            right_target = left_start + rotated_swing

                            samples.append(
                                {
                                    "left_start": left_start.copy(),
                                    "right_start": right_start.copy(),
                                    "left_target": left_start.copy(),  # Stance foot doesn't move
                                    "right_target": right_target,
                                    "stance_foot": "left",
                                    "swing_foot": "right",
                                    "target_yaw": lf_yaw_start + yaw_swing,
                                }
                            )

    # For each right foot starting position (left foot swings)
    x_values_swing_left = np.linspace(left_swing_range["x"][0], left_swing_range["x"][1], grid_x_steps)
    y_values_swing_left = np.linspace(left_swing_range["y"][0], left_swing_range["y"][1], grid_y_steps)
    yaw_values_swing_left = np.linspace(left_swing_range["yaw"][0], left_swing_range["yaw"][1], grid_yaw_steps)

    for rf_dx_start in rf_start_x:
        for rf_dy_start in rf_start_y:
            for rf_yaw_start in rf_start_yaw:
                # Compute starting positions with yaw offset
                right_start = rfPos0.copy()
                right_start[0] += rf_dx_start
                right_start[1] += rf_dy_start
                if(right_start[1]>0):
                    print("right_start[1] > 0")
                    print(rfPos0.copy()[1])
                    print(rf_dy_start)
                # Apply yaw rotation to left foot starting position relative to right foot
                # Rotate lfPos0 relative to rfPos0 by rf_yaw_start
                rel_lf = lfPos0 - rfPos0  # Left foot relative to right foot
                cos_yaw = np.cos(rf_yaw_start)
                sin_yaw = np.sin(rf_yaw_start)
                rotated_lf = np.array([cos_yaw * rel_lf[0] - sin_yaw * rel_lf[1], sin_yaw * rel_lf[0] + cos_yaw * rel_lf[1], rel_lf[2]])
                left_start = right_start + rotated_lf

                # For each left foot target position (relative to right start)
                for dx_swing in x_values_swing_left:
                    for dy_swing in y_values_swing_left:
                        for yaw_swing in yaw_values_swing_left:
                            # Rotate swing target by starting yaw
                            swing_rel = np.array([dx_swing, dy_swing, 0.0])
                            rotated_swing = np.array(
                                [cos_yaw * swing_rel[0] - sin_yaw * swing_rel[1], sin_yaw * swing_rel[0] + cos_yaw * swing_rel[1], swing_rel[2]]
                            )
                            left_target = right_start + rotated_swing

                            samples.append(
                                {
                                    "left_start": left_start.copy(),
                                    "right_start": right_start.copy(),
                                    "left_target": left_target,
                                    "right_target": right_start.copy(),  # Stance foot doesn't move
                                    "stance_foot": "right",
                                    "swing_foot": "left",
                                    "target_yaw": rf_yaw_start + yaw_swing,
                                }
                            )

    return samples


def solve_stepping_problem(gait, x0, left_target, right_target, target_yaw=0.0, verbose=False):
    """Solve a stepping problem."""
    try:
        problem = gait.createSingleStepProblem(x0, left_target, right_target, TIME_STEP, STEP_KNOTS, SUPPORT_KNOTS, STEP_HEIGHT, target_yaw)
        solver = crocoddyl.SolverIntro(problem)
        solver.th_stop = SOLVER_THRESHOLD

        # Only add verbose callback if requested
        if verbose:
            solver.setCallbacks([crocoddyl.CallbackVerbose()])

        xs = [x0] * (solver.problem.T + 1)
        us = solver.problem.quasiStatic([x0] * solver.problem.T)
        solver.solve(xs, us, MAX_ITERATIONS, False)

        success = solver.stop < SOLVER_THRESHOLD
        return solver, success
    except Exception as e:
        print(f"Error solving problem: {e}")
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
        T_stsw_const = np.array([swing_stance_pos[0], swing_stance_pos[1], swing_stance_pos[2], swing_stance_yaw])

        # Compute cmd_footstep in stance frame
        cmd_footstep_data[t] = transform_to_stance_frame(swing_target, stance_pos, stance_R, yaw_target)

        # Stance indicator
        cmd_stance_data[t, 0] = 0 if stance_is_left else 1

        # Countdown: goes from 1 -> 0 uniformly throughout the step
        progress = t / (T - 1) if T > 1 else 0
        cmd_countdown_data[t, 0] = 1 - progress
    T_stsw_data[:] = T_stsw_const
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
        display = crocoddyl.MeshcatDisplay(robot)
        display.rate = -1
        display.freq = 1

        # Add world coordinate frame (XYZ axes) using Pinocchio
        try:
            viz = display.robot.viz
            # Display frame at world origin with 0.5m scale
            pinocchio.visualize.meshcat_visualizer.displayFrame(
                viz, "world_frame", pinocchio.SE3.Identity(), 0.5
            )
            print("World coordinate frame (XYZ) displayed at origin:")
            print("  X-axis: Red")
            print("  Y-axis: Green")
            print("  Z-axis: Blue")
        except Exception as e:
            print(f"Could not add world frame: {e}")

        return display
    return None


def visualize_trajectory(display, solver):
    """Visualize a single trajectory using existing display."""
    if display is not None:
        # Use the display method that takes the full state trajectory
        display.displayFromSolver(solver)


def main():
    """Main batch generation loop."""
    print("=" * 80)
    print("Stepping Motion Dataset Generator")
    print("=" * 80)

    # Load robot
    print("\n[1/4] Loading robot model...")
    robot = load_robot()
    robotpre = load_robot()

    # Initial state (reference for randomization)
    q0 = robot.model.referenceConfigurations["half_sitting"].copy()
    v0 = np.zeros(robot.model.nv)

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
    gaitpre = SimpleBipedGaitProblem(robotpre.model, rightFoot, leftFoot, fwddyn=False)

    # Create display (if visualization enabled)
    display = create_display(robot)
    if display is not None:
        print("Visualization enabled")

    # Generate grid samples
    print("\n[3/4] Generating grid samples...")
    grid_samples = generate_grid_samples(
        lfPos0, rfPos0, LEFT_RAND_RANGE, RIGHT_RAND_RANGE, LEFT_SWING_RANGE, RIGHT_SWING_RANGE, GRID_X_STEPS, GRID_Y_STEPS, GRID_YAW_STEPS
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
    rng = np.random.default_rng(42)
    rng.shuffle(grid_samples)  # Shuffle samples for randomness
    for i, sample in enumerate(grid_samples):
        print(f"\n--- Sample {i + 1}/{len(grid_samples)} ---")

        left_start = sample["left_start"]
        right_start = sample["right_start"]
        left_target = sample["left_target"]
        right_target = sample["right_target"]
        stance_foot = sample["stance_foot"]
        swing_foot = sample["swing_foot"]
        target_yaw = sample["target_yaw"]

        # Compute foot command (displacement from start to target)
        if swing_foot == "left":
            foot_cmd_xy = left_target[:2] - left_start[:2]
        else:
            foot_cmd_xy = right_target[:2] - right_start[:2]

        print(f"Stance foot: {stance_foot}")
        print(f"Foot command: [{foot_cmd_xy[0]:.3f}, {foot_cmd_xy[1]:.3f}], yaw: {target_yaw:.3f} rad")
        print(f"  Left foot: {left_start[:2]} -> {left_target[:2]}")
        print(f"  Right foot: {right_start[:2]} -> {right_target[:2]}")

        # Generate random waiting time and frames (before stepping)
        wait_time_before = np.random.uniform(WAIT_TIME_RANGE[0], WAIT_TIME_RANGE[1])
        wait_frames_before = int(wait_time_before / TIME_STEP)

        # Randomize initial joint configuration (add small noise to half_sitting)
        q0_random = q0.copy()
        # Add noise to joint positions (not the base)
        q0_random[7:] += np.random.uniform(-0.05, 0.05, size=len(q0_random[7:]))
        x0_random = np.concatenate([q0_random, v0])

        # Solve stepping trajectory to starting positions first (presolver)
        presolver, presuccess = solve_stepping_problem(gaitpre, x0_random, left_start, right_start, 0.0, verbose=False)
        if not presuccess or presolver is None:
            print("✗ Failed to converge to starting position")
            continue
        xstart = presolver.xs[-1].copy()
        wait_data_before = generate_waiting_frames(robot, gait, xstart, wait_frames_before, left_target, right_target, target_yaw)
        solver, success = solve_stepping_problem(gait, xstart, left_target, right_target, target_yaw, verbose=False)

        if success and solver is not None:
            # Visualize the trajectory
            visualize_trajectory(display, solver)

            # Extract stepping data
            step_data = extract_trajectory_data(robot, solver, gait, left_target, right_target, target_yaw)

            # Generate random waiting time and frames (after stepping)
            wait_time_after = np.random.uniform(WAIT_TIME_RANGE[0], WAIT_TIME_RANGE[1])
            wait_frames_after = int(wait_time_after / TIME_STEP)

            # Generate waiting frames after stepping (using final state from step)
            final_state = solver.xs[-1]
            wait_data_after = generate_waiting_frames(robot, gait, final_state, wait_frames_after, left_target, right_target, target_yaw)

            # Concatenate: waiting_before + stepping + waiting_after
            all_q.append(wait_data_before["q"])
            all_q.append(step_data["q"])
            all_q.append(wait_data_after["q"])
            all_qd.append(wait_data_before["qd"])
            all_qd.append(step_data["qd"])
            all_qd.append(wait_data_after["qd"])
            all_T_blf.append(wait_data_before["T_blf"])
            all_T_blf.append(step_data["T_blf"])
            all_T_blf.append(wait_data_after["T_blf"])
            all_T_brf.append(wait_data_before["T_brf"])
            all_T_brf.append(step_data["T_brf"])
            all_T_brf.append(wait_data_after["T_brf"])
            all_T_stsw.append(wait_data_before["T_stsw"])
            all_T_stsw.append(step_data["T_stsw"])
            all_T_stsw.append(wait_data_after["T_stsw"])
            all_p_wcom.append(wait_data_before["p_wcom"])
            all_p_wcom.append(step_data["p_wcom"])
            all_p_wcom.append(wait_data_after["p_wcom"])
            all_T_wbase.append(wait_data_before["T_wbase"])
            all_T_wbase.append(step_data["T_wbase"])
            all_T_wbase.append(wait_data_after["T_wbase"])
            all_v_b.append(wait_data_before["v_b"])
            all_v_b.append(step_data["v_b"])
            all_v_b.append(wait_data_after["v_b"])
            all_cmd_footstep.append(wait_data_before["cmd_footstep"])
            all_cmd_footstep.append(step_data["cmd_footstep"])
            all_cmd_footstep.append(wait_data_after["cmd_footstep"])
            all_cmd_stance.append(wait_data_before["cmd_stance"])
            all_cmd_stance.append(step_data["cmd_stance"])
            all_cmd_stance.append(wait_data_after["cmd_stance"])
            all_cmd_countdown.append(wait_data_before["cmd_countdown"])
            all_cmd_countdown.append(step_data["cmd_countdown"])
            all_cmd_countdown.append(wait_data_after["cmd_countdown"])
            print(wait_data_before["T_wbase"])
            # Record next trajectory start index
            current_length = sum(len(q) for q in all_q)
            traj_starts.append(current_length)

            successful_samples += 1
            # print("T_stsw: ", wait_data_before["T_stsw"],step_data["T_stsw"],wait_data_after["T_stsw"])
            # print("cmd_footstep: ", wait_data_before["cmd_footstep"],step_data["cmd_footstep"],wait_data_after["cmd_footstep"])
            print(f"✓ Success! Wait before: {wait_frames_before}, Step: {len(step_data['q'])}, Wait after: {wait_frames_after}")
        else:
            print("✗ Failed to converge")

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
