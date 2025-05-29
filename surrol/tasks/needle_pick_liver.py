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

    def _env_setup(self):
        super(NeedlePick, self)._env_setup()
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
        # obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf'),
        #                     np.array(self.POSE_TRAY[0]) * self.SCALING,
        #                     p.getQuaternionFromEuler(self.POSE_TRAY[1]),
        #                     globalScaling=self.SCALING)
        # self.obj_ids['fixed'].append(obj_id)  # 1

        # needle
        # yaw = (np.random.rand() - 0.5) * np.pi
        # obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf'),
        #                     (workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,  # TODO: scaling
        #                      workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1,
        #                      workspace_limits[2][0] + 0.01),
        #                     p.getQuaternionFromEuler((0, 0, yaw)),
        #                     useFixedBase=False,
        #                     globalScaling=self.SCALING)
        
        # NOTE: Original
        # yaw = 0.1 * np.pi
        # obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf'),
        #                     (workspace_limits[0].mean() ,  # TODO: scaling
        #                      workspace_limits[1].mean(),
        #                      workspace_limits[2][0] + 0.01),
        #                     p.getQuaternionFromEuler((0, 0, yaw)),
        #                     useFixedBase=False,
        #                     globalScaling=self.SCALING)
        # p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        # self.obj_ids['rigid'].append(obj_id)  # 0
        # self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], 1
        
        # NOTE: Adjusted workspace_limits = ((2.5, 3), (-0.25, 0.25), (3.476, 3.776))
        # NOTE: Before scale WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.685, 0.745))
        yaw = 1.5 * np.pi
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf'),
                            (workspace_limits[0].mean() ,  # TODO: scaling
                             workspace_limits[1][0],
                             workspace_limits[2][0] + 0.01),
                            p.getQuaternionFromEuler((0, 0, yaw)),
                            useFixedBase=False,
                            globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], 1
        
        # Adding liver object
        scale = [0.02, 0.02, 0.02]
        liver_pos = [
            workspace_limits[0].mean(), 
            workspace_limits[1].mean() + 0.05, 
            workspace_limits[2][0] - 0.03
        ] # 3.42
        rotation = p.getQuaternionFromEuler([0.55 * np.pi, -np.pi / 2, 0])
        liver_file = os.path.join(ASSET_DIR_PATH, 'liver/hepatitis_liver.obj')
        
        # Create a dummy liver object to get the size bouding box
        # NOTE: This step relies on the pre-ek
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

        r = max(
            ((x - center_x)**2 + (y - center_y)**2)**0.5
            for x, y in xy_corners
        )

        cylinder_visual_id = p.createVisualShape(
            p.GEOM_CYLINDER, 
            radius=r, 
            length=aabb_max[2] - aabb_min[2], 
            rgbaColor=[0, 1, 0, 0.3]
        )

        cylinder_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=cylinder_visual_id,
            basePosition=(
                liver_pos[0] + center_x,
                liver_pos[1] + center_y,
                liver_pos[2] + center_z
            ),
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        
        # Small sphere for tracking PSM
        # NOTE: For some reason, can't update its position
        # visual_sphere_id = p.createVisualShape(
        #     shapeType=p.GEOM_SPHERE,
        #     radius=0.005,
        #     rgbaColor=[0, 0, 1, 1]
        # )
        
        # self.psm_track_id = p.createMultiBody(
        #     baseMass = 0.0,
        #     baseVisualShapeIndex=visual_sphere_id,
        #     basePosition=self._get_robot_state(0)[0:3],
        #     baseOrientation=(0, 0, 0, 1)
        # )
        
        # self.psm_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'sphere/psm.urdf'),
        #                          globalScaling=self.SCALING)
        
        # self.obj_ids['obstacle'].append(psm_id) # 0
    
    def _render_callback(self, mode):
        """ Display current PSMs' positions.
        """
        # PSMs
        psm1_pos = self._get_robot_state(0)[0:3]

        p.resetBasePositionAndOrientation(
            self.psm_id,
            psm1_pos,
            (0, 0, 0, 1))

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        workspace_limits = self.workspace_limits1
        # goal = np.array([workspace_limits[0].mean() + 0.01 * np.random.randn() * self.SCALING,
        #                  workspace_limits[1].mean() + 0.01 * np.random.randn() * self.SCALING,
        #                  workspace_limits[2][1] - 0.04 * self.SCALING])
        goal = np.array([workspace_limits[0].mean() ,
                         workspace_limits[1][1] ,
                         workspace_limits[2][1] - 0.04 * self.SCALING])
        return goal.copy()

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