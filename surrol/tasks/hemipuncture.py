"""
This module defines the environment for the Hemipuncture task.

It sets up a simulated workspace with two primary obstacles: a large, hollow
hemisphere and a smaller, hollow cylinder. These obstacles form a constrained
path that a surgical robot must navigate to reach a target position.

The environment handles the instantiation and positioning of these obstacles
using the PyBullet physics engine and provides methods for tracking the
robot's state and detecting collisions based on geometric constraints.
"""

import os
import time
from typing import Tuple, Union

import torch
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

from surrol.const import ASSET_DIR_PATH
from surrol.gym.surrol_env import RENDER_HEIGHT, RENDER_WIDTH
from surrol.tasks.psm_env import PsmEnv
from surrol.utils.pybullet_utils import get_link_pose, render_image



class Hemipuncture(PsmEnv):
    """
    Hemipuncture environment class.
    
    This class inherits from PsmEnv to set up a specific task environment
    for a surgical robot to pass through a cylindrical tube and into a
    hemisphere to reach a target.
    """
    
    POSE_TRAY: Tuple[Tuple, Tuple] = ((0.55, 0, 0.6751), (0, 0, 0))
    SCALING: float = 5.0
    
    OUTSIDE: int = 0
    INSIDE_CYLINDER: int = 1
    INSIDE_HEMISPHERE: int = 2
    
    def __init__(self, *args, **kwargs):
        """
        Initializes the Hemipuncture environment.
        """
        super().__init__(*args, **kwargs)
        self.has_object = False
    
    def _env_setup(self):
        """
        Sets up the environment by positioning the robot and obstacles.
        """
        super()._env_setup()
        
        # Place the tray pad
        obj_id = p.loadURDF(
            os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf'),
            np.array(self.POSE_TRAY[0]) * self.SCALING,
            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
            globalScaling=self.SCALING
        )
        p.changeVisualShape(obj_id, -1, specularColor=(10, 10, 10))
        self.obj_ids['fixed'].append(obj_id)  # 1

        # Reset robot position to a starting point
        workspace_limits = self.workspace_limits1
        pos = (
            workspace_limits[0][0],
            workspace_limits[1][1],
            workspace_limits[2][1]
        )
        orn = (0.5, 0.5, -0.5, -0.5)
        
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = True
        
        # Place hemisphere obstacle
        hemisphere_scale = 25.0
        hemisphere_radius = 0.01 * hemisphere_scale
        
        hemisphere_pos = [
            workspace_limits[0].mean(), 
            workspace_limits[1].mean(), 
            workspace_limits[2][0] - 0.016 # 0.0102 * 2.5
        ]
        
        self.hemisphere_id = p.loadURDF(
            os.path.join(ASSET_DIR_PATH, 'sphere/half_sphere.urdf'),
            np.array(hemisphere_pos),
            p.getQuaternionFromEuler((-np.pi / 2, 0, 0)),
            globalScaling=hemisphere_scale
        )
        
        self.obj_ids['obstacle'].append(self.hemisphere_id) # 0
        
        # Place cylinder obstacle
        direction = np.array(pos) - np.array(hemisphere_pos)
        direction = direction / np.linalg.norm(direction)
        cylinder_pos = hemisphere_pos + direction * hemisphere_radius

        default_axis = np.array([0, 0, 1])
        axis_of_rotation = np.cross(default_axis, direction)
        angle_of_rotation = np.arccos(np.dot(default_axis, direction))
        cylinder_orn = p.getQuaternionFromAxisAngle(axis_of_rotation, angle_of_rotation)
        
        cylinder_visual_id = p.createVisualShape(
            p.GEOM_CYLINDER, 
            radius=0.05, 
            length=0.05,
            rgbaColor=[0, 1, 1, 0.5]
        )

        self.cylinder_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=cylinder_visual_id,
            basePosition=cylinder_pos,
            baseOrientation=cylinder_orn
        )
    
        self.obj_ids['obstacle'].append(self.cylinder_id) # 1
        
        psm_visual_shape = p.createVisualShape(
            p.GEOM_SPHERE, 
            radius=0.007, 
            rgbaColor=[0, 0, 0, 1]
        )
        
        self.psm_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=psm_visual_shape,
            basePosition=self._get_robot_state(0)[0:3]
        )
        
        cylinder_center_shape = p.createVisualShape(
            p.GEOM_SPHERE, 
            radius=0.007, 
            rgbaColor=[1.0, 0.41, 0.71, 1]
        )
        
        self.cylinder_center_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=cylinder_center_shape,
            basePosition=cylinder_pos
        )
        
        # TODO: Add other obstacle inside hemisphere.
        
        # TODO: Make sure the stick doesn't collide with cylinder.
        pitch_pos = np.array(get_link_pose(self.psm1.body, 4)[0])
        self.tool_pitch_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_SPHERE, 
                radius=0.007, 
                rgbaColor=[0, 0, 0, 1]
            ),
            basePosition=pitch_pos
        )
        
        remote_center_pos = np.array(get_link_pose(self.psm1.body, 13)[0])
        stick_pos = pitch_pos + 0.1 * (remote_center_pos - pitch_pos)
        
        self.stick_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_SPHERE, 
                radius=0.007, 
                rgbaColor=[0, 0, 0, 1]
            ),
            basePosition=stick_pos
        )
        
        # psm_tool_yaw_link: 5
        # psm_main_insertion_link_3: 15 (Too far)
        # psm_remote_center_link: 13 (RCM)
             
    def _set_action(self, action: np.ndarray):
        action[3] = 0  # no yaw change
        super(Hemipuncture, self)._set_action(action)
        
    def _sample_goal(self) -> np.ndarray:
        """ 
        Samples a new goal and returns it.
        """
        
        # Get the position and radius of the hemisphere obstacle
        pos, _ = get_link_pose(self.obj_ids['obstacle'][0], -1)
        radius = 0.01 * 25.0
        
        while True:
            x = np.random.uniform(pos[0] - radius, pos[0] + radius)
            y = np.random.uniform(pos[1] - radius, pos[1] + radius)
            z = np.random.uniform(pos[2], pos[2] + radius)
            if ((x - pos[0]) ** 2 + (y - pos[1]) ** 2 + (z - pos[2]) ** 2 <= radius ** 2):
                break
        
        goal = np.array([x, y, z])
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
    
    def get_region(self, pos: Union[np.ndarray, torch.Tensor]) -> int:
        """
        Determines the region of the workspace where the given position is located.
        
        Regions:
            0 - Outside both obstacles
            1 - Inside the cylinder only
            2 - Inside the hemisphere
        """
        if isinstance(pos, torch.Tensor):
            pos = pos.squeeze(0).cpu().numpy()
            
        assert pos.shape == (3,), f"Unexpected pos shape: {pos.shape}"
        
        # Get hemisphere properties
        hemi_center, hemi_radius = self.get_hemisphere_prop()

        # Check if 'pos' is inside the hemisphere's spherical volume.
        is_in_hemisphere = np.linalg.norm(pos - hemi_center) <= hemi_radius

        # Get cylinder properties
        cyl_center, cyl_axis, cyl_length, cyl_radius = self.get_cylinder_prop()
        
        pos_to_cyl_center_vec = pos - cyl_center
        projected_dist_along_axis = np.dot(pos_to_cyl_center_vec, cyl_axis)
        is_within_cyl_length = np.abs(projected_dist_along_axis) <= (cyl_length / 2.0)
        
        # Calculate distance from the cylinder's axis
        projection_on_axis = projected_dist_along_axis * cyl_axis
        vec_from_axis_to_pos = pos_to_cyl_center_vec - projection_on_axis
        dist_from_axis = np.linalg.norm(vec_from_axis_to_pos)
        is_within_cyl_radius = dist_from_axis <= cyl_radius
        
        is_in_cylinder = is_within_cyl_length and is_within_cyl_radius
        
        # --- Determine Region ---
        # NOTE: Must first check whether it is in cylinder, otherwise it goes to hemisphere 
        #       directly and cause violation because cylinder is too short.
        #       (2025/10/02) Let me change it back to hemisphere first.
        if is_in_hemisphere:
            return self.INSIDE_HEMISPHERE
        elif is_in_cylinder:
            return self.INSIDE_CYLINDER
        else:
            return self.OUTSIDE
        
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
        rotation_matrix = R.from_quat(np.array(cyl_orn)).as_matrix()
        cyl_axis = (rotation_matrix @ np.array([0, 0, 1]).reshape([3, 1])).reshape(-1)
        assert abs(np.linalg.norm(cyl_axis) - 1.0) < 1e-6, "Cylinder axis is not a unit vector."
        return np.array(cyl_center), cyl_axis, cyl_length, cyl_radius
    
    def get_hemisphere_prop(self) -> Tuple[np.ndarray, float]:
        """
        Retrieves the properties of the hemisphere obstacle.
        
        Returns:
            A tuple containing the hemisphere's center and radius.
            - hemi_center (np.ndarray): The center coordinates of the hemisphere.
            - hemi_radius (float): The radius of the hemisphere.
        """
        hemi_center, _ = p.getBasePositionAndOrientation(self.hemisphere_id)
        hemi_radius = 0.25 # See line 83
        return np.array(hemi_center), hemi_radius

    def check_lateral_collision(self, prev_psm_pos, psm_pos):
        """
        Checks for lateral collisions between the PSM's path and the cylinder obstacle.
        This method determines if the line segment from prev_psm_pos to psm_pos intersects with the cylinder.
        
        Args:
            prev_psm_pos (np.ndarray): The previous position of the PSM.
            psm_pos (np.ndarray): The current position of the PSM.
            
        Returns:
            bool: True if a collision occurred, False otherwise.
        """

        # Convert positions to numpy arrays
        prev_pos = np.array(prev_psm_pos)
        curr_pos = np.array(psm_pos)
        
        # Define ray (PSM path segment)
        ray_origin = prev_pos
        ray_direction = curr_pos - prev_pos
        
        # The following is the math for ray-cylinder intersection
        # based on the geometric properties of the cylinder and the line segment.
        cyl_center, cyl_axis, cyl_length, cyl_radius = self.get_cylinder_prop()
        
        # Let's define some vectors
        oc = ray_origin - np.array(cyl_center)
        
        '''
        P = ray_origin + t * ray_direction

        Solve below quadratic equation in "t"
        || P - C - np.dot(P - C, cyl_axis) * cyl_axis || ** 2 = cyl_radius ** 2
        a t^2 + b t + c = 0
        '''
        
        # Components perpendicular to the cylinder axis
        a = np.sum((ray_direction - np.dot(ray_direction, cyl_axis) * cyl_axis) ** 2)
        b = 2 * np.sum((ray_direction - np.dot(ray_direction, cyl_axis) * cyl_axis) * (oc - np.dot(oc, cyl_axis) * cyl_axis))
        c = np.sum((oc - np.dot(oc, cyl_axis) * cyl_axis) ** 2) - cyl_radius ** 2
        
        # Solve quadratic equation for intersection time 't'
        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            return False # No intersection
        
        # Find the two roots
        t1 = (-b - np.sqrt(discriminant)) / (2 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2 * a)

        # Check if the intersection point is on the line segment
        if (0 <= t1 <= 1) or (0 <= t2 <= 1):
            # Now check if this intersection point is within the cylinder's length
            # This prevents collisions with the caps of the cylinder
            
            # Intersection point on the ray
            intersection_point = ray_origin + t1 * ray_direction
            if not (0 <= t1 <= 1):
                intersection_point = ray_origin + t2 * ray_direction
                
            # Check if the intersection point is within the cylinder's length
            projected_length_at_intersection = np.dot(intersection_point - np.array(cyl_center), cyl_axis)
            
            is_within_cyl_length = np.abs(projected_length_at_intersection) <= (cyl_length / 2.0)
            
            if is_within_cyl_length:
                return True # Collision occurred
        
        return False

    def render_three_views(self, mode='rgb_array'):
        self._render_callback(mode)
        if mode == "human":
            return np.array([])
        
        camera_pos = np.array(self.POSE_TRAY[0]) * self.SCALING
        camera_pos[2] += (0.25 / 2)  # raise camera a bit higher
        
        # front
        _view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_pos,
            distance=0.7,
            yaw=-90,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        front_rgb_array, _ = render_image(RENDER_WIDTH, RENDER_HEIGHT,
                                       _view_matrix, self._proj_matrix)

        # right
        _view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_pos,
            distance=0.7,
            yaw=180,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        right_rgb_array, _ = render_image(RENDER_WIDTH, RENDER_HEIGHT,
                                       _view_matrix, self._proj_matrix)

        # top
        _view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_pos,
            distance=0.7,
            yaw=90,
            pitch=-89,
            roll=0,
            upAxisIndex=2
        )
        top_rgb_array, _ = render_image(RENDER_WIDTH, RENDER_HEIGHT,
                                       _view_matrix, self._proj_matrix)
        if mode == 'rgb_array':
            return [front_rgb_array, right_rgb_array, top_rgb_array]
        else:
            raise ValueError('Masks are not saved')
    
    def _render_callback(self, mode):
        """
        A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        # Doesn't get call if run on local computer
        p.resetBasePositionAndOrientation(
            self.psm_id,
            self._get_robot_state(0)[0:3],
            (0, 0, 0, 1))
        
        pitch_pos = np.array(get_link_pose(self.psm1.body, 4)[0])
        p.resetBasePositionAndOrientation(
            self.tool_pitch_id,
            pitch_pos,
            (0, 0, 0, 1))
        
        remote_center_pos = np.array(get_link_pose(self.psm1.body, 13)[0])
        stick_pos = pitch_pos + 0.1 * (remote_center_pos - pitch_pos)
        p.resetBasePositionAndOrientation(
            self.stick_id,
            stick_pos,
            (0, 0, 0, 1))
        
    def check_collision(
        self, 
        pos: np.ndarray, 
        prev_pos: np.ndarray, 
        curr_region: int, 
        prev_region: int,
        point: str
    ) -> bool:
        # Violation Case 1: PSM enters the hemisphere (region 2) from outside (region 0).
        if prev_region == self.OUTSIDE and curr_region == self.INSIDE_HEMISPHERE:
            print(f"VIOLATION: The {point} entered the hemisphere's solid region.")
            return True

        # Violation Case 2: PSM leaves the hemisphere (region 2) to outside (region 0).
        elif prev_region == self.INSIDE_HEMISPHERE and curr_region == self.OUTSIDE:
            print(f"VIOLATION: {point} left the hemisphere's solid region.")
            return True
            
        # Violation Case 3: PSM enters the cylinder (region 1) but has clipped through the wall.
        # This check is only necessary if the PSM enters the cylinder region.
        elif (curr_region == self.INSIDE_CYLINDER and
            # A transition from region 0 to 1 indicates an entry into the cylinder.
            # We must check if this entry was valid.
            prev_region == self.OUTSIDE and 
            self.check_lateral_collision(prev_pos, pos)):
            print(f"VIOLATION: {point} clipped through the cylinder wall.")
            return True
            # elif prev_region == self.INSIDE_HEMISPHERE:
            #     print(f"WARNING: {point} entered the cylinder from the hemisphere. This should not happen.")

        # Violation Case 4: PSM leaves the cylinder (region 1) but has clipped through the wall.
        elif (prev_region == self.INSIDE_CYLINDER and
            curr_region == self.OUTSIDE and
            self.check_lateral_collision(prev_pos, pos)):
            print(f"VIOLATION: {point} clipped through the cylinder wall on exit.")
            return True
            
        # TODO: Need to add code for newer obstacle later
                
        else:
            return False
        
if __name__ == "__main__":
    env = Hemipuncture(render_mode='human')
    env.test()
    env.close()
    time.sleep(2)