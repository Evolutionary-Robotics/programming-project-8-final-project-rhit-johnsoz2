import numpy as np


class Controller:
    def __init__(self, body, ctrnn):
        self.body = body
        self.ctrnn = ctrnn

    def control_step(self):
        # Fetch joint positions
        joint_positions = self.body.get_joint_positions()
        print("Joint Positions:", joint_positions)

        # Apply control signals
        # joint_controls = np.array([100, 100])  # Smaller values for testing
        # print("Joint Controls:", joint_controls)

        self.body.apply_joint_control(joint_controls)


