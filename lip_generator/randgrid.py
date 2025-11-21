import numpy as np
import pinocchio
import crocoddyl
#Left/Right foot random range when to Right/Left foot stance
LEFT_RAND_RANGE = {
    "x": (-0.3, 0.4),  # Forward/backward range (meters)
    "y": (0.23, 0.45),  # Lateral range (meters)  0.25 to avoid self-collision
    "yaw": (-0.2, 0.2),
}
RIGHT_RAND_RANGE = {
    "x": (-0.3, 0.4),  # Forward/backward range (meters)
    "y": (-0.45, -0.23),  # Lateral range (meters) - 0.25 to avoid self-collision
    "yaw": (-0.2, 0.2),
}
LEFT_SWING_RANGE = {
    "x": (-0.3, 0.4),  # Forward/backward range (meters)
    "y": (-0.02, 0.2),  # Lateral range (meters)
    "yaw": (-0.2, 0.2),
}
RIGHT_SWING_RANGE = {
    "x": (-0.3, 0.4),  # Forward/backward range (meters)
    "y": (-0.2, 0.02),  # Lateral range (meters)
    "yaw": (-0.2, 0.2),
}

class RandomGridGenerator:
    def __init__(self, robot_model, right_foot, left_foot):
        self.robot_model = robot_model
        self.right_foot = right_foot
        self.left_foot = left_foot
    def generate_random_step(self):
        #Randoml pick left and right foot positions within defined ranges
        left_foot_pos = np.array([
            np.random.uniform(*LEFT_RAND_RANGE["x"]),
            np.random.uniform(*LEFT_RAND_RANGE["y"]),
            np.random.uniform(*LEFT_RAND_RANGE["yaw"]),
        ])

        right_foot_pos = np.array([
            np.random.uniform(*RIGHT_RAND_RANGE["x"]),
            np.random.uniform(*RIGHT_RAND_RANGE["y"]),
            np.random.uniform(*RIGHT_RAND_RANGE["yaw"]),
        ])

        return left_foot_pos, right_foot_pos