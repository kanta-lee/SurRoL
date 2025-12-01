import os
import time
import numpy as np

import pybullet as p
from surrol.tasks.psm_env import PsmEnv
from surrol.utils.pybullet_utils import (
    get_link_pose,
    wrap_angle
)
from surrol.const import ASSET_DIR_PATH
from typing import Tuple


class NeedlePickSphere(PsmEnv):
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.685, 0.745))  # reduce tip pad contact
    SCALING = 5.

    # TODO: grasp is sometimes not stable; check how to fix it
    
    
    # Size of workspace
    # workspace_limits = np.asarray(self.WORKSPACE_LIMITS1) \
    #                     + np.array([0., 0., 0.0102]).reshape((3, 1))  # tip-eef offset with collision margin
    # workspace_limits *= self.SCALING  # use scaling for more stable collistion simulation
    # self.workspace_limits1 = workspace_limits
    #
    # NOTE: workspace_limits = ((2.5, 3), (-0.25, 0.25), (3.476, 3.776))
    
    # NOTE: Weighted midpoint between "psm_tool_yaw_link" (PSM's position in state) 
    #       and "psm_remote_center_link" (which is constant) is used as additional
    #       point to represent the PSM stick (insertion link).
    REMOTE_CENTER_LINK = 13

    def _env_setup(self):
        super(NeedlePickSphere, self)._env_setup()
        # np.random.seed(4)  # for experiment reproduce
        self.has_object = True
        self._waypoint_goal = True

        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False
        # physical interaction
        self._contact_approx = False

        # tray pad
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING)
        self.obj_ids['fixed'].append(obj_id)  # 1
        
        # ==============================================================================
        #                             SPHERE OBSTACLE
        # ==============================================================================
        
        sphere_radius = 0.1
        sphere_pos = [
            workspace_limits[0].mean(), 
            workspace_limits[1].mean(), 
            workspace_limits[2][0] + sphere_radius - 0.03
        ]
        
        self.sphere_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_SPHERE, 
                radius=sphere_radius, 
                rgbaColor=[0, 1, 0, 0.3]
            ),
            basePosition=sphere_pos,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        
        self.obj_ids['obstacle'].append(self.sphere_id)  # 0
        
        # ==============================================================================
        #                                   NEEDLE
        # ==============================================================================
        
        needle_radius = 0.1

        needle_ranges = np.array((
            (workspace_limits[0].mean(), workspace_limits[0].mean() + 0.1),
            (workspace_limits[1][0], workspace_limits[1][0] - 0.001),
            (workspace_limits[2][0] + 0.01, workspace_limits[2][0] + 0.009)
        ))

        pick_up_ranges = np.array((
            (workspace_limits[0].mean() - 0.1, workspace_limits[0].mean() + 0.1),
            (workspace_limits[1][0], workspace_limits[1][0] + 0.1),
            (workspace_limits[2][0] + 0.01, workspace_limits[2][0] + 0.009)
        ))
        
        # self.draw_workspace_box(np.array(workspace_limits), color=[0, 0, 1])
        # self.draw_workspace_box(needle_ranges, color=[1, 1, 0])
        # self.draw_workspace_box(pick_up_ranges, color=[0, 1, 1])
        
        # Random needle position until it falls under valid pick up range.
        for _ in range(1000):
            yaw = (np.random.rand() - 0.5) * np.pi
            needle_pos = (
                np.random.uniform(needle_ranges[0][0], needle_ranges[0][1]),
                workspace_limits[1][0],
                workspace_limits[2][0] + 0.01
            )

            pick_up_pos =(
                needle_pos[0] - needle_radius * np.cos(yaw),
                needle_pos[1] - needle_radius * np.sin(yaw),
                needle_pos[2]
            )

            x, y, _ = pick_up_pos
            x_min, x_max = pick_up_ranges[0][0], pick_up_ranges[0][1]
            y_min, y_max = pick_up_ranges[1][0], pick_up_ranges[1][1]
            
            if (x_min <= x <= x_max and y_min <= y <= y_max):
                break

        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf'),
                            needle_pos,
                            p.getQuaternionFromEuler((0, 0, yaw)),
                            useFixedBase=False,
                            globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], 1

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        workspace_limits = self.workspace_limits1
        std = 0.01 * self.SCALING

        goal_ranges = np.array((
            (workspace_limits[0].mean() + std * -2.576, workspace_limits[0].mean() + std * 2.576),
            (workspace_limits[1].mean() + std * 2.576, workspace_limits[1][1]),
            (workspace_limits[2][1] - 0.04 * self.SCALING, workspace_limits[2][1] - 0.039 * self.SCALING) # Drawing purpose
        ))

        # self.draw_workspace_box(goal_ranges, color=[1, 0, 0])

        goal = np.array([
            np.random.uniform(goal_ranges[0][0], goal_ranges[0][1]),
            np.random.uniform(goal_ranges[1][0], goal_ranges[1][1]),
            workspace_limits[2][1] - 0.04 * self.SCALING
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
        self._waypoints = [None, None, None, None]  # four waypoints
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        self._waypoint_z_init = pos_obj[2]
        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        yaw = orn[2] if abs(wrap_angle(orn[2] - orn_eef[2])) < abs(wrap_angle(orn[2] + np.pi - orn_eef[2])) \
            else wrap_angle(orn[2] + np.pi)  # minimize the delta yaw

        # # for physical deployment only
        # print(" -> Needle pose: {}, {}".format(np.round(pos_obj, 4), np.round(orn_obj, 4)))
        # qs = self.psm1.get_current_joint_position()
        # joint_positions = self.psm1.inverse_kinematics(
        #     (np.array(pos_obj) + np.array([0, 0, (-0.0007 + 0.0102)]) * self.SCALING,
        #      p.getQuaternionFromEuler([-90 / 180 * np.pi, -0 / 180 * np.pi, yaw])),
        #     self.psm1.EEF_LINK_INDEX)
        # self.psm1.reset_joint(joint_positions)
        # print("qs: {}".format(joint_positions))
        # print("Cartesian: {}".format(self.psm1.get_current_position()))
        # self.psm1.reset_joint(qs)

        self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, yaw, 0.5])  # approach
        self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, yaw, 0.5])  # approach
        self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, yaw, -0.5])  # grasp
        self._waypoints[3] = np.array([self.goal[0], self.goal[1],
                                       self.goal[2] + 0.0102 * self.SCALING, yaw, -0.5])  # lift up

    def _meet_contact_constraint_requirement(self):
        # add a contact constraint to the grasped block to make it stable
        if self._contact_approx:
            return True  # mimic the dVRL setting
        else:
            pose = get_link_pose(self.obj_id, self.obj_link1)
            return pose[0][2] > self._waypoint_z_init + 0.005 * self.SCALING

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # psm_pos = self._get_robot_state(0)[0:3]
        # weighted_mp = psm_pos + 0.1 * (self.remote_center - psm_pos)
        
        # p.resetBasePositionAndOrientation(
        #     self.obj_ids['rigid'][1],
        #     weighted_mp,
        #     (0, 0, 0, 1)
        # )
        
        # four waypoints executed in sequential order
        action = np.zeros(5)
        action[4] = -0.5
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
            delta_yaw = (waypoint[3] - obs['observation'][5]).clip(-0.4, 0.4)
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.4
            delta_pos *= scale_factor
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]])
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4 and np.abs(delta_yaw) < 1e-2:
                self._waypoints[i] = None
            break

        return action
    
    def check_collision(self):
        """
        Check if the end-effector is inside the sphere.
        
        Returns:
            bool: True if end-effector is inside the sphere, False otherwise
        """
        psm_pos = self._get_robot_state(0)[0:3]
        center, radius = self.get_sphere_prop()
        b = np.sum((center - psm_pos) ** 2) - radius ** 2
        return b <= 0
    
    def get_sphere_prop(self) -> Tuple[np.ndarray, float]:
        """
        Retrieves the properties of the sphere obstacle.
        
        Returns:
            A tuple containing the sphere's center and radius.
            - center (np.ndarray): The center coordinates of the sphere.
            - radius (float): The radius of the sphere.
        """
        center, _ = p.getBasePositionAndOrientation(self.sphere_id)
        radius = p.getVisualShapeData(self.sphere_id)[0][3][0]
        return np.array(center), radius

    # NOTE: Since we want to take more information about the observation, let's
    #       override the _get_obs method.
    def _get_obs(self) -> dict:
        # PSM position and orientation
        robot_state = self._get_robot_state(idx=0)
    
        # NOTE: Base link for needle is a little weird. 
        #       It is not on the needle itself.
        pos, _ = get_link_pose(self.obj_id, -1)
        object_pos = np.array(pos)

        # Center of the needle (self.obj_link1 = 1)
        pos, orn = get_link_pose(self.obj_id, self.obj_link1)
        waypoint_pos = np.array(pos)
        # rotations
        waypoint_rot = np.array(p.getEulerFromQuaternion(orn))
        # relative position state
        object_rel_pos = object_pos - robot_state[0: 3]

        # ============================= TWO ENDS =============================
        # NOTE: Below are additional information for two ends of the needle.
        #       The two ends are at 90 degrees from the center.
        #       I have modified needle_40mm.urdf to add two additional links.
        left_90_pos, left_90_orn = np.array(get_link_pose(self.obj_id, 6))
        right_90_pos, right_90_orn = np.array(get_link_pose(self.obj_id, 7))
    
        # ============================= END ==================================
        # object/waypoint position
        achieved_goal = object_pos.copy() if not self._waypoint_goal else waypoint_pos.copy()

        observation = np.concatenate([
            robot_state, object_pos.ravel(), object_rel_pos.ravel(),
            waypoint_pos.ravel(), waypoint_rot.ravel(), left_90_pos.ravel(), left_90_orn.ravel(),
            right_90_pos.ravel(), right_90_orn.ravel()
        ])
        
        obs = {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy()
        }
        return obs

if __name__ == "__main__":
    env = NeedlePickSphere(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()
    time.sleep(2)