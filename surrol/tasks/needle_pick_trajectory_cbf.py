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

OBJ_FILENAME = 'lung/lungs.obj' 

class NeedlePickTrajectoryCBF(PsmEnv):
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05),
                        (0.685, 0.745))  # reduce tip pad contact
    SCALING = 5.
    
    # Trajectory type for obstacle placement: 'line', 'circle', or 'triangle'
    TRAJ_TYPE = 'line'  # Can be overridden via command line or config

    def _env_setup(self):
        super(NeedlePickTrajectoryCBF, self)._env_setup()
        self.has_object = True
        self._waypoint_goal = True

        # robot setup
        workspace_limits = self.workspace_limits1
        #start point
        x_start = workspace_limits[0][0] + 0.2
        y_start = workspace_limits[1][1]
        z_start = (workspace_limits[2][1] + workspace_limits[2][0]) / 2 + 0.05
        pos = (x_start, y_start, z_start)
        # pos = (workspace_limits[0][0] +0.2,
        #        workspace_limits[1][1],
        #        (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics(
            (pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False
        self._contact_approx = False

        # tray pad
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING)
        self.obj_ids['fixed'].append(obj_id)

        # # ==============================================================================
        # #                               SUTURE PAD (OBJ LOADER)
        # # ==============================================================================
        
        obj_path = os.path.join(ASSET_DIR_PATH, OBJ_FILENAME)
        if not os.path.exists(obj_path):
            obj_path = os.path.abspath(OBJ_FILENAME)
            if not os.path.exists(obj_path):
                raise FileNotFoundError(f"❌ 错误: 找不到文件 {OBJ_FILENAME}。\n请确保它在 {ASSET_DIR_PATH} 目录下，或在当前代码运行目录下。")
        
        print(f"[INFO] Loading OBJ from: {obj_path}")

        # 检查 MTL 文件是否存在
        mtl_path = obj_path.replace('.obj', '.mtl')
        if os.path.exists(mtl_path):
            print(f"[INFO] Found corresponding MTL file: {mtl_path}")
        else:
            print(f"[WARNING] MTL file not found at: {mtl_path}")
        # 2. 设置模型参数 (位置 & 缩放)
        raw_scale = 0.00045  # 缩小物体尺寸 
        mesh_scale = [raw_scale * self.SCALING] * 3

        # 使用和旧文件圆柱体相同的位置，但稍微降低高度
        # 旧文件: workspace_limits[2][0] + 0.045
        # 现在降低高度：从 workspace_limits[2][0] 减去一个值
        suture_pad_pos = (
            workspace_limits[0].mean(),
            workspace_limits[1].mean(),
            workspace_limits[2][0] - 0.01  # 减去 0.01 让物体变低（数值越大越低）
        )
        self.cylinder_center = np.array(suture_pad_pos, dtype=float)
        self.cylinder_offset = np.array([0.0, 0.04, -0.05]) # 用于 sample_goal 里的偏移
        self.cyl_radius = 0.05 # 保留用于红点定位

        # # 3. 创建视觉形状 (Visual)
        # visual_shape_id = p.createVisualShape(
        #     shapeType=p.GEOM_MESH,
        #     fileName=obj_path,
        #     meshScale=mesh_scale
        #     # 不设置 rgbaColor，让它使用 MTL 文件中的材质
        #     # 如果 MTL 不生效，可以尝试设置 rgbaColor=[0.8, 0.5, 0.4, 1.0] 等
        # )

        # # 4. 创建碰撞形状 (Collision)
        # collision_shape_id = p.createCollisionShape(
        #     shapeType=p.GEOM_MESH,
        #     fileName=obj_path,
        #     meshScale=mesh_scale,
        #     flags=p.GEOM_FORCE_CONCAVE_TRIMESH
        # )
        # # 让物体正对着观察者竖起来：尝试不同的旋转角度

        # pad_orn = p.getQuaternionFromEuler([0, np.pi / 2, np.pi])  # 先试X轴，如果不对可改为 [0, np.pi/2, 0] 或其他 
        # # 5. 创建多体
        # self.cylinder_id = p.createMultiBody(
        #     baseMass=0, # 0 = 静态物体
        #     # baseCollisionShapeIndex=collision_shape_id,
        #     baseCollisionShapeIndex=-1,
        #     baseVisualShapeIndex=visual_shape_id,
        #     basePosition=np.array(suture_pad_pos) * self.SCALING,
        #     baseOrientation=pad_orn
        # )
        
        # self.obj_ids['obstacle'].append(self.cylinder_id)
        
        # Create obstacle sphere
        self.obstacle_radius = 0.018  # 半径
        self.start_pos = np.array(pos)  # 保存起始位置
        self.obstacle_z_offset = -0.06  # z轴偏移量
        
        # 计算轨迹中点位置，根据轨迹类型
        obstacle_pos = self._calculate_trajectory_midpoint(self.start_pos, self.TRAJ_TYPE)

        self.obstacle_id = p.createMultiBody(
            baseMass=0, # 静态
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_SPHERE, 
                radius=self.obstacle_radius, 
                rgbaColor=[1, 0, 0, 1]  # 红色
            ),
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_SPHERE, 
                radius=self.obstacle_radius
            ),
            basePosition=obstacle_pos,
            baseOrientation=[0, 0, 0, 1]
        )
        
        # 将其加入 obstacle 列表
        self.obj_ids['obstacle'].append(self.obstacle_id)
        # 调试：检查视觉形状信息
        # visual_info = p.getVisualShapeData(self.cylinder_id)
        # print(f"[DEBUG] Visual shapes for suture pad: {len(visual_info)} shapes")
        # for i, shape in enumerate(visual_info):
        #     print(f"[DEBUG] Shape {i}: {shape}")


        # needle 起始位置设置
        yaw = (np.random.rand() - 0.5) * np.pi
        needle_start_pos = (
            workspace_limits[0].mean() + 0.05,  # X坐标（左右）
            workspace_limits[1].mean() +  0.15,  # Y坐标（前后）
            workspace_limits[2][0] + 0.01  # Z坐标（高度）：底部 + 0.01，增大数值让针更高
        )
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf'),
                            needle_start_pos,
                            p.getQuaternionFromEuler((0, 0, yaw)),
                            useFixedBase=False,
                            globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], 1

        # # points
        # sphere_radius = 0.02
        # sphere_pos = [
        #     workspace_limits[0].mean(), 
        #     workspace_limits[1].mean(), 
        #     workspace_limits[2][0] + sphere_radius - 0.03
        # ]
        # self.sphere_id_1 = p.createMultiBody(
        #     baseMass=0,
        #     baseVisualShapeIndex=p.createVisualShape(
        #         p.GEOM_SPHERE, 
        #         radius=sphere_radius, 
        #         rgbaColor=[1, 0, 0, 0.7]
        #     ),
        #     basePosition=sphere_pos,
        #     baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        # )
        # self.obj_ids['fixed'].append(self.sphere_id_1) # 2

        # self.sphere_id_2 = p.createMultiBody(
        #     baseMass=0,
        #     baseVisualShapeIndex=p.createVisualShape(
        #         p.GEOM_SPHERE, 
        #         radius=sphere_radius, 
        #         rgbaColor=[1, 0, 0, 0.7]
        #     ),
        #     basePosition=sphere_pos,
        #     baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        # )
        # self.obj_ids['fixed'].append(self.sphere_id_2) # 3
    
    def _calculate_trajectory_midpoint(self, start_pos: np.ndarray, traj_type: str) -> np.ndarray:
        """
        Calculate the midpoint of the trajectory based on trajectory type.
        The obstacle will be placed at this midpoint with a z offset.
        
        Args:
            start_pos: Starting position of the robot
            traj_type: Type of trajectory ('line', 'circle', 'triangle')
            
        Returns:
            The position for the obstacle (trajectory midpoint with z offset)
        """
        start = np.asarray(start_pos, dtype=np.float32)
        
        if traj_type == 'line':
            # For line trajectory, the goal is in negative y direction
            # Based on CLF._init_trajectory_state with traj_type='line'
            # goal = start + [0, -0.3, 0] (from _init_linear_state_cbf in clf.py)
            goal = start + np.array([0.0, -0.3, 0.0], dtype=np.float32)
            midpoint = (start + goal) / 2
            
        elif traj_type == 'circle':
            # For circle trajectory, midpoint is on the opposite side of the circle
            # Based on CLF._init_trajectory_state with traj_type='circle'
            circle_radius = 0.12
            theta_start = np.pi / 2  # Starting angle
            center_xy = start[:2] - circle_radius * np.array([np.cos(theta_start), np.sin(theta_start)], dtype=np.float32)
            
            # Midpoint is at theta_start - π (opposite side of circle)
            theta_mid = theta_start - np.pi
            midpoint = np.array([
                center_xy[0] + circle_radius * np.cos(theta_mid),
                center_xy[1] + circle_radius * np.sin(theta_mid),
                start[2]
            ], dtype=np.float32)
            
        elif traj_type == 'triangle':
            # For triangle trajectory, midpoint is around vertex1 or middle of edge
            # Based on CLF._init_trajectory_state with traj_type='triangle'
            triangle_size = 0.1
            triangle_height = triangle_size * np.sqrt(3) / 2
            vertex1 = start + np.array([triangle_size, 0, 0], dtype=np.float32)
            vertex2 = start + np.array([triangle_size / 2, -triangle_height, 0], dtype=np.float32)
            
            # Midpoint is the center of the second edge (vertex1 -> vertex2)
            midpoint = (vertex1 + vertex2) / 2
            
        else:
            # Default: just offset from start
            midpoint = start + np.array([0.0, -0.15, 0.0], dtype=np.float32)
        
        # Apply z offset
        obstacle_pos = np.array([midpoint[0], midpoint[1], midpoint[2] + self.obstacle_z_offset])
        
        print(f"[INFO] Obstacle placed at trajectory midpoint for traj_type='{traj_type}': {obstacle_pos}")
        return obstacle_pos

    def get_sphere_prop(self) -> Tuple[np.ndarray, float]:
        """
        Retrieves the properties of the sphere obstacle.
        
        Returns:
            A tuple containing the sphere's center and radius.
            - center (np.ndarray): The center coordinates of the sphere.
            - radius (float): The radius of the sphere.
        """
        center, _ = p.getBasePositionAndOrientation(self.obstacle_id)
        # radius = p.getVisualShapeData(self.obstacle_id)[0][3][0] + 0.008 * self.SCALING
        radius = p.getVisualShapeData(self.obstacle_id)[0][3][0] + 0.008 * self.SCALING
        return np.array(center), radius

    def check_collision(self) -> bool:
        """
        检查机器人是否与小球障碍物发生碰撞。
        
        Returns:
            bool: 如果发生碰撞返回 True，否则返回 False
        """
        # 使用 PyBullet 的接触点检测
        contact_points = p.getContactPoints(
            bodyA=self.psm1.body,
            bodyB=self.obstacle_id
        )
        return len(contact_points) > 0

    def _sample_goal(self) -> np.ndarray:
        workspace_limits = self.workspace_limits1
        x_goal = workspace_limits[0][0] + 0.2
        y_goal = workspace_limits[1][1] - 0.35
        z_goal = (workspace_limits[2][1] + workspace_limits[2][0]) / 2 + 0.05
        goal = np.array([x_goal, y_goal, z_goal])
        # goal = np.array([workspace_limits[0].mean() + 0.01 * np.random.randn() * self.SCALING,
        #                  workspace_limits[1].mean() + 0.01 * np.random.randn() * self.SCALING,
        #                  workspace_limits[2][1] - 0.04 * self.SCALING])
        return goal.copy()

    def _sample_goal_callback(self):
        super()._sample_goal_callback()
        # hide goal visualization
        p.changeVisualShape(self.obj_ids['fixed'][0], -1, rgbaColor=[1, 0, 0, 0])

        self._waypoints = [None, None, None, None]
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        self._waypoint_z_init = pos_obj[2]
        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        yaw = orn[2] if abs(wrap_angle(orn[2] - orn_eef[2])) < abs(wrap_angle(orn[2] + np.pi - orn_eef[2])) \
            else wrap_angle(orn[2] + np.pi)
        
        # update suture pad pose based on goal (和旧文件逻辑一致)
        # goal 在 _sample_goal 中已经乘以了 SCALING，所以直接使用
        # cylinder_offset 需要乘以 SCALING 以匹配缩放后的坐标系
        self.cylinder_center = np.array([self.goal[0], self.goal[1], self.goal[2]]) + self.cylinder_offset * self.SCALING
        
        # 保持竖起来的朝向：与创建时保持一致
        p.resetBasePositionAndOrientation(
            self.obj_ids['obstacle'][0],
            self.cylinder_center,
            p.getQuaternionFromEuler([0, np.pi / 2 , np.pi]))  # 与创建时的朝向保持一致
        
        # 球的位置基于轨迹中点
        obstacle_pos = self._calculate_trajectory_midpoint(self.start_pos, self.TRAJ_TYPE)
        p.resetBasePositionAndOrientation(
            self.obstacle_id,
            obstacle_pos,
            [0, 0, 0, 1]
        )
        # # set points (red markers)
        # point1_angle = np.pi/4
        # point2_angle = np.pi/4 * 3
        # point1_pos = np.array([self.cylinder_center[0] + self.cyl_radius * np.cos(point1_angle),
        #             self.cylinder_center[1] + self.cyl_radius * np.sin(point1_angle),
        #             self.cylinder_center[2]])
        # p.resetBasePositionAndOrientation(
        #     self.obj_ids['fixed'][2],
        #     point1_pos,
        #     p.getQuaternionFromEuler([0, 0, 0]))
        # point2_pos = np.array([self.cylinder_center[0] + self.cyl_radius * np.cos(point2_angle),
        #     self.cylinder_center[1] + self.cyl_radius * np.sin(point2_angle),
        #     self.cylinder_center[2]])
        # p.resetBasePositionAndOrientation(
        #     self.obj_ids['fixed'][3],
        #     point2_pos,
        #     p.getQuaternionFromEuler([0, 0, 0]))

        self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, yaw, 0.5])  # approach
        self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, yaw, 0.5])  # approach
        self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, yaw, -0.5])  # grasp
        # 让轨迹穿过 point：使用 cylinder_center[2] 作为 Z 坐标（与 point 高度一致）
        # 可以选择穿过 point1 或 point2，或者使用中心点
        self._waypoints[3] = np.array([self.goal[0], self.goal[1],
                                       self.cylinder_center[2], yaw, -0.5])  # lift up through point

    def _meet_contact_constraint_requirement(self):
        # add a contact constraint to the grasped block to make it stable
        if self._contact_approx:
            return True  # mimic the dVRL setting
        else:
            pose = get_link_pose(self.obj_id, self.obj_link1)
            return pose[0][2] > self._waypoint_z_init + 0.005 * self.SCALING

    def get_oracle_action(self, obs) -> np.ndarray:
        # ... (expert strategy code remains same) ...
        # four waypoints executed in sequential order
        action = np.zeros(5)
        action[4] = -0.5
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            delta_pos = (waypoint[:3] - obs['observation']
                         [:3]) / 0.01 / self.SCALING
            delta_yaw = (waypoint[3] - obs['observation'][5]).clip(-0.4, 0.4)
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.4
            delta_pos *= scale_factor
            action = np.array([delta_pos[0], delta_pos[1],
                              delta_pos[2], delta_yaw, waypoint[4]])
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4 and np.abs(delta_yaw) < 1e-2:
                self._waypoints[i] = None
            break

        return action


if __name__ == "__main__":
    # 记得实例化正确的类名
    env = NeedlePickTrajectoryCBF(render_mode='human')

    env.test()
    env.close()
    time.sleep(2)