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
        #                                   LIVER
        # ==============================================================================
        
        # Adding liver object
        scale = [0.02, 0.02, 0.02]
        liver_pos = [
            workspace_limits[0].mean(), 
            workspace_limits[1].mean() + 0.05, 
            workspace_limits[2][0] - 0.03
        ]
        rotation = p.getQuaternionFromEuler([0.55 * np.pi, -np.pi / 2, 0])
        liver_file = os.path.join(ASSET_DIR_PATH, 'liver/hepatitis_liver.obj')
        
        # Create a dummy liver object to get the size bouding box
        dummy_liver_collision = p.createCollisionShape(
            p.GEOM_MESH,
            fileName=liver_file,
            meshScale=scale,
        )
        
        dummy_visual_liver_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=liver_file,
            meshScale=scale,
            rgbaColor=[0.8, 0.2, 0.2, 0.8]
        )
        
        dummy_liver_id = p.createMultiBody(
            baseMass = 0,
            baseCollisionShapeIndex=dummy_liver_collision,
            baseVisualShapeIndex=dummy_visual_liver_id,
            basePosition=liver_pos,
            baseOrientation=rotation
        )
        
        def get_scaled_obb(obj_path, pybullet_scale, rotation):
            # 1. Load vertices from OBJ and apply PyBullet scale
            vertices = []
            with open(obj_path) as f:
                for line in f:
                    if line.startswith("v "):
                        vertex = np.array([float(x) for x in line.split()[1:4]])
                        vertex_scaled = vertex * pybullet_scale  # Apply scale
                        vertices.append(vertex_scaled)
            vertices = np.array(vertices)
            
            # 2. Apply rotation (if any)
            if rotation is not None:
                rot_matrix = np.array(p.getMatrixFromQuaternion(rotation)).reshape(3, 3)
                vertices = vertices @ rot_matrix.T  # Rotate vertices
            
            # 3. Compute OBB bounds
            return vertices.min(axis=0), vertices.max(axis=0)

        # Usage:
        aabb_min, aabb_max = get_scaled_obb(liver_file, scale, rotation)
        p.removeBody(dummy_liver_id)
        
        visual_liver_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=liver_file,
            meshScale=scale,
            rgbaColor=[0.8, 0.2, 0.2, 0.8]
        )
        
        liver_id = p.createMultiBody(
            baseMass = 0,
            baseVisualShapeIndex=visual_liver_id,
            basePosition=liver_pos,
            baseOrientation=rotation
        )
        
        # Get the box's center in X-Y plane
        center_x = (aabb_max[0] + aabb_min[0]) / 2
        center_y = (aabb_max[1] + aabb_min[1]) / 2
        center_z = (aabb_max[2] + aabb_min[2]) / 2

        xy_corners = [
            [aabb_min[0], aabb_min[1]],
            [aabb_max[0], aabb_min[1]],
            [aabb_max[0], aabb_max[1]],
            [aabb_min[0], aabb_max[1]]
        ]

        cylinder_radius = max(
            ((x - center_x)**2 + (y - center_y)**2)**0.5
            for x, y in xy_corners
        )

        cylinder_visual_id = p.createVisualShape(
            p.GEOM_CYLINDER, 
            radius=cylinder_radius, 
            length=aabb_max[2] - aabb_min[2], 
            rgbaColor=[0, 1, 0, 0.3]
        )

        self.cylinder_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=cylinder_visual_id,
            basePosition=(
                liver_pos[0] + center_x,
                liver_pos[1] + center_y,
                liver_pos[2] + center_z
            ),
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        
        self.obj_ids['obstacle'].append(self.cylinder_id)  # 0
        
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
        
        gauze_ranges = np.array((
            (workspace_limits[0][0] + self.dimensions[0] / 2, workspace_limits[0][1] - self.dimensions[0] / 2),
            (workspace_limits[1][0], (liver_pos[1] + center_y) - cylinder_radius - self.dimensions[1] / 2),
            (workspace_limits[2][0], workspace_limits[2][1])
        ))
        # self.draw_workspace_box(np.array(workspace_limits))
        # self.draw_workspace_box(gauze_ranges, color=[1, 0, 0])
        
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
        
        self.cylinder_radius = cylinder_radius
        self.cylinder_center_y = liver_pos[1] + center_y
        
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
        super(GauzeRetrieveCylinder, self)._set_action(action)

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        workspace_limits = self.workspace_limits1
        goal = np.array([
            np.random.uniform(workspace_limits[0][0] + self.dimensions[0] / 2 , workspace_limits[0][1] - self.dimensions[0] / 2),
            np.random.uniform(self.cylinder_center_y + self.cylinder_radius + self.dimensions[1] / 2 , workspace_limits[1][1]),
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
        center, orn = p.getBasePositionAndOrientation(self.cylinder_id)
        rotation_matrix = Rotation.from_quat(np.array(orn)).as_matrix()
        axis_vector = np.array([0, 0, 1]).reshape([3, 1])
        axis_vector = (rotation_matrix @ axis_vector).reshape(-1).tolist()
        robot = self._get_robot_state(0)[0:3]
        
        # Actually should divide by norm(axis_vector)
        # Since it is 1, so doesn't matter
        projected_length = np.dot(axis_vector, robot - np.array(center))
        
        # p.getVisualShapeData() return a list of tuples
        # (objectUniqueId, linkIndex, geometryType, dimensions, localFramePosition, localFrameOrientation, rgbaColor)
        dimensions = p.getVisualShapeData(self.cylinder_id)[0][3]
        length, radius = dimensions[0], dimensions[1]
        
        # Check whether it is in the range of cylinder's length
        if projected_length ** 2 <= (length / 2) ** 2:
            # Check whether it is in the range of cylinder's radius
            proj_vec = projected_length * np.array(axis_vector)
            norm_vec = np.array(robot) - (np.array(center) + proj_vec)
            violate_constraint = (np.sum(norm_vec ** 2) - radius ** 2 <= 0)
        else:
            # Check collision condition for the PSM stick
            remote_center = self.remote_center
            weighted_mp = robot + 0.1 * (remote_center - robot)
            projected_length = np.dot(axis_vector, weighted_mp - np.array(center))
            
            # Check whether it is in the range of cylinder's length
            if projected_length ** 2 <= (length / 2 + 0.025) ** 2:
                # Check whether it is in the range of cylinder's radius
                proj_vec = projected_length * np.array(axis_vector)
                norm_vec = weighted_mp - (np.array(center) + proj_vec)
                violate_constraint = (np.sum(norm_vec ** 2) - (radius + 0.025) ** 2 <= 0)
            else:
                violate_constraint = False
                
        return violate_constraint


if __name__ == "__main__":
    env = GauzeRetrieveCylinder(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()
    time.sleep(2)
