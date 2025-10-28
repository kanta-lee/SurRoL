import os
import time
import numpy as np

import pybullet as p
from surrol.tasks.psm_env import PsmEnv
from surrol.utils.pybullet_utils import (
    get_link_pose,
)
from surrol.const import ASSET_DIR_PATH


class GauzeRetrieve(PsmEnv):
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
        super(GauzeRetrieve, self)._env_setup()
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
        #                             SPHERE OBSTACLE
        # ==============================================================================
        
        sphere_radius = 0.1
        sphere_pos = [
            workspace_limits[0].mean(), 
            workspace_limits[1].mean(), 
            workspace_limits[2][0] + sphere_radius - 0.03
        ]
        
        
        sphere_visual_id = p.createVisualShape(
            p.GEOM_SPHERE, 
            radius=sphere_radius, 
            rgbaColor=[0, 1, 0, 0.3]
        )

        sphere_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=sphere_visual_id,
            basePosition=sphere_pos,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        
        self.obj_ids['obstacle'].append(sphere_id)  # 0
        
        # ==============================================================================
        #                                   GAUZE
        # ==============================================================================
        
        def get_scaled_obb(obj_path, pybullet_scale):
            # Load vertices from OBJ and apply PyBullet scale
            vertices = []
            with open(obj_path) as f:
                for line in f:
                    if line.startswith("v "):
                        vertex = np.array([float(x) for x in line.split()[1:4]])
                        vertex_scaled = vertex * pybullet_scale  # Apply scale
                        vertices.append(vertex_scaled)
            vertices = np.array(vertices)
            
            # Compute OBB bounds
            return vertices.min(axis=0), vertices.max(axis=0)

        obj_path = os.path.join(ASSET_DIR_PATH, 'gauze/meshes/gauze.obj')
        aabb_min, aabb_max = get_scaled_obb(obj_path, self.SCALING)
        self.dimensions = np.array(aabb_max) - np.array(aabb_min)
        
        offset = 0.05
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'gauze/gauze.urdf'),
                            (np.random.uniform(workspace_limits[0][0] + self.dimensions[0] / 2 , workspace_limits[0][1] - self.dimensions[0] / 2),
                             np.random.uniform(workspace_limits[1][0] + self.dimensions[1] / 2, sphere_pos[1] - sphere_radius - self.dimensions[1] / 2 - offset),
                             workspace_limits[2][0] + 0.01),
                            (0, 0, 0, 1),
                            useFixedBase=False,
                            globalScaling=self.SCALING)
        
        p.changeVisualShape(obj_id, -1, specularColor=(0, 0, 0))
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], -1
        
        self.sphere_radius = sphere_radius
        self.sphere_center_y = sphere_pos[1]
        
        # Create a cyan sphere to visualize point on PSM stick
        psm_stick_visual_shape = p.createVisualShape(
            p.GEOM_SPHERE, 
            radius=0.02, 
            rgbaColor=[0, 1, 1, 1]
        )
        
        self.remote_center = np.array(get_link_pose(self.psm1.body, self.REMOTE_CENTER_LINK)[0])
        psm_pos = self._get_robot_state(0)[0:3]
        weighted_mp = psm_pos + 0.07 * (self.remote_center - psm_pos)
        
        psm_stick_body = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=psm_stick_visual_shape,
            basePosition=weighted_mp
        )
        
        self.obj_ids['rigid'].append(psm_stick_body) # 1

    def _set_action(self, action: np.ndarray):
        action[3] = 0  # no yaw change
        super(GauzeRetrieve, self)._set_action(action)

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        workspace_limits = self.workspace_limits1
        goal = np.array([
            np.random.uniform(workspace_limits[0][0] + self.dimensions[0] / 2 , workspace_limits[0][1] - self.dimensions[0] / 2),
            np.random.uniform(self.sphere_center_y + self.sphere_radius + self.dimensions[1] / 2 , workspace_limits[1][1]),
            np.random.uniform(workspace_limits[2][0] + 0.01, workspace_limits[2][1] - 0.01)
        ])
        return goal.copy()
    
    def _render_callback(self, mode):
        """ A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        # Doesn't get call if run on local computer
        psm_pos = self._get_robot_state(0)[0:3]
        weighted_mp = psm_pos + 0.07 * (self.remote_center - psm_pos)
        
        p.resetBasePositionAndOrientation(
            self.obj_ids['rigid'][1],
            weighted_mp,
            (0, 0, 0, 1))

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


if __name__ == "__main__":
    env = GauzeRetrieve(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()
    time.sleep(2)
