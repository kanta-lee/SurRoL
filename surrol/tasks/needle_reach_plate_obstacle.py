import os
import time
import numpy as np

import pybullet as p
from surrol.tasks.psm_env import PsmEnv
from surrol.utils.pybullet_utils import (
    get_link_pose,
)
from surrol.const import ASSET_DIR_PATH
from typing import Tuple
from scipy.spatial.transform import Rotation


class NeedleReach(PsmEnv):
    """
    Refer to Gym FetchReach
    https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch/reach.py
    """
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.681, 0.745))
    SCALING = 5.

    def _env_setup(self):
        super(NeedleReach, self)._env_setup()
        self.has_object = False

        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               workspace_limits[2][1])
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = True

        # tray pad
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(10, 10, 10))
        self.obj_ids['fixed'].append(obj_id)  # 1

        # needle
        yaw = (np.random.rand() - 0.5) * np.pi
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf'),
                            (workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,
                             workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1,
                             workspace_limits[2][0] + 0.01),
                            p.getQuaternionFromEuler((0, 0, yaw)),
                            useFixedBase=False,
                            globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], 1

        # ==============================================================================
        #                               CYLINDER
        # ==============================================================================
        
        cyl_radius = 0.1
        cyl_length = 0.04
        cyl_pos = (
            workspace_limits[0].mean() - 0.07,
            workspace_limits[1].mean() + 0.02,
            workspace_limits[2][0] + 0.15
        )

        self.cylinder_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,  # no collision shape
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_CYLINDER, 
                radius=cyl_radius, 
                length=cyl_length, 
                rgbaColor=[0, 1, 0, 0.3]
            ),
            basePosition=cyl_pos,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        # Optional hard-disable collisions just in case:
        p.setCollisionFilterGroupMask(self.cylinder_id, -1, 0, 0)
        
        self.obj_ids['obstacle'].append(self.cylinder_id)  # 0

    def _set_action(self, action: np.ndarray):
        action[3] = 0  # no yaw change
        super(NeedleReach, self)._set_action(action)

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        pos, orn = get_link_pose(self.obj_id, self.obj_link1)
        goal = np.array([pos[0], pos[1], pos[2] + 0.005 * self.SCALING])

        # Put the obstacle
        # p.resetBasePositionAndOrientation(
            # self.obj_ids['obstacle'][0], np.array([goal[0], goal[1], goal[2]+0.1]), (0., 0., 0., 1.))
        return goal.copy()

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        delta_pos = (obs['desired_goal'] - obs['achieved_goal']) / 0.01
        if np.linalg.norm(delta_pos) < 1.5:
            delta_pos.fill(0)
        if np.abs(delta_pos).max() > 1:
            delta_pos /= np.abs(delta_pos).max()
        delta_pos *= 0.3

        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0., 0.])
        return action
    
    def get_cylinder_prop(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Retrieves the properties of the cylinder obstacle.
        
        Returns:
            A tuple containing the cylinder's center, axis, length, and radius.
            - cyl_center (np.ndarray): The center coordinates of the cylinder.
            - cyl_axis (np.ndarray): The orientation vector of the cylinder's axis.
            - cyl_length (float): The length of the cylinder.
            - cyl_radius (float): The radius of the cylinder.
        """
        cyl_center, cyl_orn = p.getBasePositionAndOrientation(self.cylinder_id)
        cyl_dimensions = p.getVisualShapeData(self.cylinder_id)[0][3]
        cyl_length, cyl_radius = cyl_dimensions[0], cyl_dimensions[1]
        rotation_matrix = Rotation.from_quat(np.array(cyl_orn)).as_matrix()
        cyl_axis = (rotation_matrix @ np.array([0, 0, 1]).reshape([3, 1])).reshape(-1)
        assert abs(np.linalg.norm(cyl_axis) - 1.0) < 1e-6, "Cylinder axis is not a unit vector."
        return np.array(cyl_center), cyl_axis, cyl_length, cyl_radius
    
    def check_collision(self):
        """
        Check if the robot position is inside the cylinder.
        
        Returns:
            bool: True if robot is inside the cylinder, False otherwise
        """
        cyl_center, cyl_axis, cyl_length, cyl_radius = self.get_cylinder_prop()
        psm_pos = self._get_robot_state(0)[0:3]
        
        # Vector from cylinder center to robot position
        vec_to_psm = psm_pos - cyl_center
        
        # Project the vector onto the cylinder axis to find the distance along the axis
        projection_length = np.dot(vec_to_psm, cyl_axis)
        
        # Check if the robot is within the cylinder's length
        if abs(projection_length) > cyl_length / 2:
            return False

        # Calculate the perpendicular distance from the axis
        projection_vector = projection_length * cyl_axis
        perpendicular_vector = vec_to_psm - projection_vector
        distance_from_axis = np.linalg.norm(perpendicular_vector)
        
        # Check if the robot is within the cylinder's radius
        if distance_from_axis > cyl_radius:
            return False
        
        return True


if __name__ == "__main__":
    env = NeedleReach(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()
    time.sleep(2)
