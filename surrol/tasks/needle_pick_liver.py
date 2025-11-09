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
from scipy.spatial.transform import Rotation
from typing import Tuple


class NeedlePickCylinder(PsmEnv):
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
        super(NeedlePickCylinder, self)._env_setup()
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
        #                                   NEEDLE
        # ==============================================================================
        
        # NOTE: Adjusted workspace_limits = ((2.5, 3), (-0.25, 0.25), (3.476, 3.776))
        # NOTE: Before scale WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.685, 0.745))
        # y - 0.1 <= aabb_min
        # workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,  # TODO: scaling
        needle_radius = 0.1
        offset = 0.05
        yaw = 1.5 * np.pi
        
        # needle_ranges = np.array((
        #     (workspace_limits[0][0] + needle_radius, workspace_limits[0][1] - needle_radius), # [2.6, 2.9]
        #     (workspace_limits[1][0], (liver_pos[1] + center_y) - cylinder_radius - offset), # [-0.25, -0.138475805]
        #     (workspace_limits[2][0], workspace_limits[2][1])
        # ))
        
        # (workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,  # TODO: scaling
        #  workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1,
        #  workspace_limits[2][0] + 0.01)

        needle_ranges = np.array((
            (workspace_limits[0].mean() - 0.05, workspace_limits[0].mean() + 0.05), # [2.6, 2.9]
            (workspace_limits[1][0], (liver_pos[1] + center_y) - cylinder_radius - offset), # [-0.25, -0.138475805]
            (workspace_limits[2][0], workspace_limits[2][1])
        ))

        # self.draw_workspace_box(np.array(workspace_limits))
        # self.draw_workspace_box(needle_ranges, color=[1, 0, 0])
        
        while True:
            needle_pos = (
                np.random.uniform(needle_ranges[0][0], needle_ranges[0][1]),
                np.random.uniform(needle_ranges[1][0], needle_ranges[1][1]),
                workspace_limits[2][0] + 0.01
            )
            
            yaw = (np.random.rand() - 0.5) * np.pi
            
            pick_up_pos = (
                needle_pos[0] - needle_radius * np.cos(yaw),
                needle_pos[1] - needle_radius * np.sin(yaw),
                needle_pos[2]
            )
            
            x, y, _ = pick_up_pos
            x_min, x_max = workspace_limits[0][0], workspace_limits[0][1]
            y_min, y_max = workspace_limits[1][0], needle_ranges[1][1]
            
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
        
        # These variables below are needed for random goal generation.
        self.needle_radius = needle_radius
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

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        # goal_ranges = np.array((
        #     (workspace_limits[0][0] + self.needle_radius, workspace_limits[0][1] - self.needle_radius),
        #     (self.cylinder_center_y + self.cylinder_radius + self.needle_radius, workspace_limits[1][1]),
        #     (workspace_limits[2][0] + 0.01, workspace_limits[2][1] - 0.01)
        # ))

        # goal = np.array([
        #     np.random.uniform(goal_ranges[0][0], goal_ranges[0][1]),
        #     np.random.uniform(goal_ranges[1][0], goal_ranges[1][1]),
        #     np.random.uniform(goal_ranges[2][0], goal_ranges[2][1])
        # ])

        workspace_limits = self.workspace_limits1

        original_goal_ranges = np.array((
            (workspace_limits[0].mean() + 0.01 * -2.576 * self.SCALING, workspace_limits[0].mean() + 0.01 * 2.576 * self.SCALING),
            (workspace_limits[1].mean() + 0.01 * -2.576 * self.SCALING, workspace_limits[1].mean() + 0.01 * 2.576 * self.SCALING),
            (workspace_limits[2][1] - 0.04 * self.SCALING, workspace_limits[2][1] - 0.03 * self.SCALING)
        ))

        self.draw_workspace_box(original_goal_ranges, color=[0, 0, 0])

        goal = np.array([workspace_limits[0].mean() + 0.01 * np.random.randn() * self.SCALING,
                         workspace_limits[1].mean() + 0.01 * np.random.randn() * self.SCALING,
                         workspace_limits[2][1] - 0.04 * self.SCALING])
        
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
        # weighted_mp = psm_pos + 0.07 * (self.remote_center - psm_pos)
        
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
    env = NeedlePickCylinder(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()
    time.sleep(2)