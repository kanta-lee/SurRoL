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


class NeedlePick(PsmEnv):
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
        super(NeedlePick, self)._env_setup()
        # np.random.seed(4)  # for experiment reproduce
        self.has_object = True
        self._waypoint_goal = True

        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0].mean(),
               workspace_limits[1].mean() + 0.25 * np.cos(np.pi / 3), # 0.25 = radius of hemisphere
               workspace_limits[2][0] - 0.0102 * self.SCALING + 0.25 * np.sin(np.pi / 3) - 0.01)
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
        #                                   HEMISPHERE
        # ==============================================================================
        
        # When SCALING = 5, the radius = 0.05
        scale = 25
        self.sphere_radius = 0.01 * scale
        
        hemisphere_pos = [
            workspace_limits[0].mean(), 
            workspace_limits[1].mean(), 
            workspace_limits[2][0] - 0.0102 * self.SCALING
        ]
        
        # mesh_path = '/Users/kantaphat/Research/DEX/SurRoL/surrol/assets/sphere/half_sphere.urdf'
        hemisphere_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'sphere/half_sphere.urdf'),
                            np.array(hemisphere_pos),
                            p.getQuaternionFromEuler((-np.pi / 2, 0, 0)),
                            globalScaling=scale)
        
        self.obj_ids['obstacle'].append(hemisphere_id)  # 0
        
        # cylinder_visual_id = p.createVisualShape(
        #     p.GEOM_CYLINDER, 
        #     radius=0.005, 
        #     length=self.sphere_radius * 2, 
        #     rgbaColor=[0, 0, 0, 0.3]
        # )

        # cylinder_id = p.createMultiBody(
        #     baseMass=0,
        #     baseVisualShapeIndex=cylinder_visual_id,
        #     basePosition=hemisphere_pos,
        #     baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        # )
        
        # ==============================================================================
        #                                   NEEDLE
        # ==============================================================================
        
        # NOTE: Adjusted workspace_limits = ((2.5, 3), (-0.25, 0.25), (3.476, 3.776))
        # NOTE: Before scale WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.685, 0.745))
        # y - 0.1 <= aabb_min
        # workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,  # TODO: scaling
        needle_radius = 0.1
        yaw = (np.random.rand() - 0.5) * np.pi
        while True:
            needle_x = np.random.uniform(hemisphere_pos[0] - 0.15, hemisphere_pos[0] + 0.15)
            needle_y = np.random.uniform(hemisphere_pos[1] - 0.15, hemisphere_pos[1] + 0.15)
            if (needle_x - hemisphere_pos[0]) ** 2 + (needle_y - hemisphere_pos[1]) ** 2 <= 0.15 ** 2:
                break
            
        # np.random.uniform(hemisphere_pos[0] - self.sphere_radius * np.sqrt(2) + needle_radius, hemisphere_pos[0] + self.sphere_radius * np.sqrt(2) - needle_radius),
        # np.random.uniform(hemisphere_pos[1] - self.sphere_radius * np.sqrt(2), hemisphere_pos[1] + self.sphere_radius * np.sqrt(2) - needle_radius),
            
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf'),
                            (needle_x, needle_y, workspace_limits[2][0] + 0.01),
                            p.getQuaternionFromEuler((0, 0, yaw)),
                            useFixedBase=False,
                            globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], 1
        
        # These variables below are needed for random goal generation.
        self.needle_radius = needle_radius
        self.hemisphere_pos = hemisphere_pos
        # self.sphere_radius = sphere_radius
        self.sphere_center_y = hemisphere_pos[1]
        
        # Create a cyan sphere to visualize point on PSM stick
        psm_stick_visual_shape = p.createVisualShape(
            p.GEOM_SPHERE, 
            radius=0.03, 
            rgbaColor=[0, 1, 1, 1]
        )
        
        self.remote_center = np.array(get_link_pose(self.psm1.body, self.REMOTE_CENTER_LINK)[0])
        psm_pos = self._get_robot_state(0)[0:3]
        weighted_mp = psm_pos + 0.1 * (self.remote_center - psm_pos)
        
        psm_stick_body = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=psm_stick_visual_shape,
            basePosition=weighted_mp
        )
        
        self.obj_ids['rigid'].append(psm_stick_body) # 1

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        workspace_limits = self.workspace_limits1
        
        hemisphere_pos = [
            workspace_limits[0].mean(), 
            workspace_limits[1].mean(), 
            workspace_limits[2][0] - 0.0102 * self.SCALING
        ]
        
        while True:
            x = np.random.uniform(hemisphere_pos[0] - 0.15, hemisphere_pos[0] + 0.15)
            y = np.random.uniform(hemisphere_pos[1], hemisphere_pos[1] + 0.15)
            z = np.random.uniform(hemisphere_pos[2], workspace_limits[2][1] - 0.01)
            if (x - hemisphere_pos[0]) ** 2 + (y - hemisphere_pos[1]) ** 2 + (z - hemisphere_pos[2]) ** 2 <= self.sphere_radius ** 2:
                break
        
        # x = np.random.uniform(self.hemisphere_pos[0] - self.sphere_radius * np.sqrt(2) + self.needle_radius, self.hemisphere_pos[0] + self.sphere_radius * np.sqrt(2) - self.needle_radius)
        # y = np.random.uniform(self.hemisphere_pos[1] - self.sphere_radius * np.sqrt(2), self.hemisphere_pos[1] + self.sphere_radius * np.sqrt(2) - self.needle_radius)
        # print(self.sphere_radius ** 2 - (x - self.hemisphere_pos[0]) ** 2 - (y - self.hemisphere_pos[1]) ** 2)
        # z = np.random.uniform(workspace_limits[2][0] + 0.01, self.hemisphere_pos[2] + np.sqrt(self.sphere_radius ** 2 - (x - self.hemisphere_pos[0]) ** 2 - (y - self.hemisphere_pos[1]) ** 2))
        
        goal = np.array([x, y, z])
        
        return goal.copy()
    
    def _render_callback(self, mode):
        """ A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        # Doesn't get call if run on local computer
        # print("----- Render Callback!")

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
        psm_pos = self._get_robot_state(0)[0:3]
        weighted_mp = psm_pos + 0.1 * (self.remote_center - psm_pos)
        
        p.resetBasePositionAndOrientation(
            self.obj_ids['rigid'][1],
            weighted_mp,
            (0, 0, 0, 1)
        )
        
        
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


if __name__ == "__main__":
    env = NeedlePick(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()
    time.sleep(2)