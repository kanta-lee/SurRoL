import os
import time
import numpy as np

import pybullet as p
from surrol.tasks.psm_env import PsmEnv
from surrol.utils.pybullet_utils import (
    get_link_pose,
)
from surrol.const import ASSET_DIR_PATH
from scipy.spatial.transform import Rotation
from typing import Tuple


class GauzeRetrieveCylinder(PsmEnv):
    """
    Refer to Gym FetchPickAndPlace
    https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch/pick_and_place.py
    """
    POSE_TRAY = ((0.55, 0, 0.6781), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.681, 0.745))
    SCALING = 5.
    REMOTE_CENTER_LINK = 13

    # TODO: grasp is sometimes not stable; check how to fix it

    def _env_setup(self):
        super(GauzeRetrieveCylinder, self)._env_setup()
        self.has_object = True
        self._waypoint_goal = True
        # self._contact_approx = True  # mimic the dVRL setting, prove nothing?

        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False

        # tray pad
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING)
        self.obj_ids['fixed'].append(obj_id)  # 1
        p.changeVisualShape(obj_id, -1, rgbaColor=(225 / 255, 225 / 255, 225 / 255, 1))
        
        # ==============================================================================
        #                                   CYLINDER
        # ==============================================================================

        cyl_radius = 0.1
        cyl_length = 0.15
        cyl_pos = (
            workspace_limits[0].mean(),
            workspace_limits[1].mean(),
            workspace_limits[2][0] + 0.045
        )

        self.cylinder_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_CYLINDER, 
                radius=cyl_radius, 
                length=cyl_length, 
                rgbaColor=[0, 1, 0, 0.3]
            ),
            basePosition=cyl_pos,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        
        self.obj_ids['obstacle'].append(self.cylinder_id)  # 0
        
        # ==============================================================================
        #                                   GAUZE
        # ==============================================================================

        gauze_ranges = np.array((
            (workspace_limits[0].mean() - 0.05, workspace_limits[0].mean() + 0.05),
            (workspace_limits[1][0], workspace_limits[1][0] + 0.08),
            (workspace_limits[2][0] + 0.01, workspace_limits[2][0] + 0.009) # Drawing purpose
        ))
        
        # self.draw_workspace_box(np.array(workspace_limits))
        # self.draw_workspace_box(gauze_ranges, color=[1, 1, 0])
        
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'gauze/gauze.urdf'),
                            (np.random.uniform(gauze_ranges[0][0], gauze_ranges[0][1]),
                             np.random.uniform(gauze_ranges[1][0], gauze_ranges[1][1]),
                             workspace_limits[2][0] + 0.01),
                            (0, 0, 0, 1),
                            useFixedBase=False,
                            globalScaling=self.SCALING)
        
        p.changeVisualShape(obj_id, -1, specularColor=(0, 0, 0))
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], -1

    def _set_action(self, action: np.ndarray):
        action[3] = 0  # no yaw change
        super(GauzeRetrieveCylinder, self)._set_action(action)

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        workspace_limits = self.workspace_limits1
        std = 0.01 * self.SCALING

        goal_ranges = np.array((
            (workspace_limits[0].mean() + 2 * std * -2.576, workspace_limits[0].mean() + 2 * std * 2.576),
            (workspace_limits[1].mean() + std * 2.576, workspace_limits[1][1]),
            (workspace_limits[2][1] - 0.03 * self.SCALING, workspace_limits[2][1] - 0.029 * self.SCALING) # Drawing purpose
        ))

        # self.draw_workspace_box(goal_ranges, color=[1, 0, 0])

        # TODO: Try using uniform?
        goal = np.array([
            np.clip(np.random.normal(goal_ranges[0].mean(), 2 * std), goal_ranges[0][0], goal_ranges[0][1]),
            np.clip(np.random.normal(goal_ranges[1].mean(), 2 * std), goal_ranges[1][0], goal_ranges[1][1]),
            workspace_limits[2][1] - 0.03 * self.SCALING
        ])

        return goal.copy()
    
    def _render_callback(self, mode):
        """ A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        # Doesn't get call if run on local computer
        # psm_pos = self._get_robot_state(0)[0:3]
        # weighted_mp = psm_pos + 0.07 * (self.remote_center - psm_pos)
        
        # p.resetBasePositionAndOrientation(
        #     self.obj_ids['rigid'][1],
        #     weighted_mp,
        #     (0, 0, 0, 1))

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        super()._sample_goal_callback()
        self._waypoints = [None, None, None, None, None]  # five waypoints
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        self._waypoint_z_init = pos_obj[2]

        self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, 0., 0.5])  # approach
        self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, 0., 0.5])  # approach
        self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, 0., -0.5])  # grasp
        self._waypoints[3] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, 0., -0.5])  # grasp
        self._waypoints[4] = np.array([self.goal[0], self.goal[1],
                                       self.goal[2] + 0.0102 * self.SCALING, 0., -0.5])  # lift up

    def _meet_contact_constraint_requirement(self):
        # add a contact constraint to the grasped object to make it stable
        pose = get_link_pose(self.obj_id, self.obj_link1)
        return pose[0][2] > self._waypoint_z_init + 0.0025 * self.SCALING
        # return True  # mimic the dVRL setting

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # four waypoints executed in sequential order
        action = np.zeros(5)
        action[4] = -0.5
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.6
            delta_pos *= scale_factor
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0., waypoint[4]])
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4:
                self._waypoints[i] = None
            break

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
    env = GauzeRetrieveCylinder(render_mode='human')  # create one process and corresponding env
    env.test()
    env.close()
    time.sleep(2)
