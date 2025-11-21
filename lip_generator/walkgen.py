import os
import signal
import sys
import time

import numpy as np
import pinocchio

import crocoddyl
from crocoddyl.utils.biped import SimpleBipedGaitProblem, plotSolution

WITHDISPLAY = True
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

robot = pinocchio.RobotWrapper.BuildFromURDF(
    "model/T1_7dof_arms_with_gripper.urdf",
    package_dirs=["model"],
    root_joint=pinocchio.JointModelFreeFlyer()
)
half_sitting = np.array([
    0, 0, 0.665,  # base position
    0, 0, 0, 1,   # base orientation (quaternion)
    0, 0,         # torso joints
    0.2, -1.35, 0, -0.5, 0.0, 0.0, 0.0,  # left arm
    0.2, 1.35, 0, 0.5, 0.0, 0.0, 0.0,    # right arm
    0,            # head
    -0.2, 0, 0, 0.4, -0.25, 0,  # left leg
    -0.2, 0, 0, 0.4, -0.25, 0   # right leg
])
robot.model.referenceConfigurations["half_sitting"] = half_sitting

# Defining the initial state of the robot
q0 = robot.model.referenceConfigurations["half_sitting"].copy()
v0 = pinocchio.utils.zero(robot.model.nv)
x0 = np.concatenate([q0, v0])
print(x0.shape)


# Setting up the 3d walking problem
rightFoot = "left_foot_link"
leftFoot = "right_foot_link"
gait = SimpleBipedGaitProblem(
    robot.model, rightFoot, leftFoot, fwddyn=False
)

# Setting up all tasks
NUM_KNOTS = 30
GAITPHASES = [
    {
        "walking": {
            "stepLength": 0.3,
            "stepHeight": 0.15,
            "timeStep": 1.0 / 1.2 / NUM_KNOTS,
            "stepKnots": NUM_KNOTS,
            "supportKnots": 2,
        }
    },
]

solver = [None] * len(GAITPHASES)
for i, phase in enumerate(GAITPHASES):
    for key, value in phase.items():
        if key == "walking":
            # Creating a walking problem
            solver[i] = crocoddyl.SolverIntro(
                gait.createWalkingProblem(
                    x0,
                    value["stepLength"],
                    value["stepHeight"],
                    value["timeStep"],
                    value["stepKnots"],
                    value["supportKnots"],
                )
            )
        elif key == "jumping":
            # Creating a jumping problem
            solver[i] = crocoddyl.SolverIntro(
                gait.createJumpingProblem(
                    x0,
                    value["jumpHeight"],
                    value["jumpLength"],
                    value["timeStep"],
                    value["groundKnots"],
                    value["flyingKnots"],
                )
            )
        solver[i].th_stop = 1e-7

    # Added the callback functions
    print("*** SOLVE " + key + " ***")
    if WITHPLOT:
        solver[i].setCallbacks(
            [
                crocoddyl.CallbackVerbose(),
                crocoddyl.CallbackLogger(),
            ]
        )
    else:
        solver[i].setCallbacks([crocoddyl.CallbackVerbose()])

    # Solving the problem with the solver
    xs = [x0] * (solver[i].problem.T + 1)
    us = solver[i].problem.quasiStatic([x0] * solver[i].problem.T)
    solver[i].solve(xs, us, 100, False)

    # Defining the final state as initial one for the next phase
    x0 = solver[i].xs[-1]


# Export trajectories to npz
q_trajectory = []
root_pose_trajectory = []
phase_trajectory = []

for i, phase_dict in enumerate(GAITPHASES):
    phase_name = next(iter(phase_dict.keys()))
    for xs in solver[i].xs:
        q = xs[:robot.model.nq]

        # extract root pose (xyz + quaternion)
        root_pos = q[:3]  # xyz
        root_quat = q[3:7]  # xyzw in pinocchio
        # convert pinocchio quaternion (x,y,z,w) to xyz_wxyz format
        root_pose = np.concatenate([root_pos, [root_quat[3]], root_quat[:3]])

        # extract joint positions (without base)
        q_joints = q[7:]

        q_trajectory.append(q_joints)
        root_pose_trajectory.append(root_pose)
        phase_trajectory.append(i)

q_trajectory = np.array(q_trajectory)
root_pose_trajectory = np.array(root_pose_trajectory)
phase_trajectory = np.linspace(0.0, 1.0, num=q_trajectory.shape[0])

np.savez(
    "trajectory.npz",
    q=q_trajectory,
    phase=phase_trajectory,
    root_pose=root_pose_trajectory
)
print(
    f"Exported trajectory: q{q_trajectory.shape}, phase{phase_trajectory.shape}, root_pose{root_pose_trajectory.shape}"
)

# Display the entire motion
if WITHDISPLAY:
    try:
        import gepetto

        gepetto.corbaserver.Client()
        cameraTF = [3.0, 3.68, 0.84, 0.2, 0.62, 0.72, 0.22]
        display = crocoddyl.GepettoDisplay(robot, 4, 4, cameraTF)
    except Exception:
        display = crocoddyl.MeshcatDisplay(robot)
    display.rate = -1
    display.freq = 1
    while True:
        for i, phase in enumerate(GAITPHASES):
            display.displayFromSolver(solver[i])
        time.sleep(1.0)
