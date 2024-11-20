import pybullet as p
import pybullet_data
import numpy as np
import time


class Body:
    def __init__(self, urdf):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        self.plane = p.loadURDF("plane.urdf")
        self.body_id = p.loadURDF(urdf, [0, -3, 0.45])  # 0.2 or higher to ensure it's above the plane

        # for joint_index in range(p.getNumJoints(self.body_id)):
        #     joint_info = p.getJointInfo(self.body_id, joint_index)
        #     print(f"Joint {joint_index}: Limits = {joint_info[8]} to {joint_info[9]}")

    def apply_joint_control(self, joint_controls):
        #Apply controls to each joint
        for joint_index in range(len(joint_controls)):
            if joint_index < p.getNumJoints(self.body_id):  # Check if joint_index is valid
                target_position = joint_controls[joint_index]  # No need to scale for testing
                p.setJointMotorControl2(self.body_id, joint_index, p.POSITION_CONTROL, targetPosition=target_position)

    def get_joint_positions(self):
        #current position of each joint.
        joint_positions = [p.getJointState(self.body_id, i)[0] for i in range(p.getNumJoints(self.body_id))]
        return joint_positions

    def step_simulation(self):
        """Step the simulation forward."""
        p.stepSimulation()
        time.sleep(1.0 / 240)  # PyBullet default timestep
