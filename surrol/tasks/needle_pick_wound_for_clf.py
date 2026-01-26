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

# 假设你的 GLB 文件放在 surrol/assets/ 目录下，或者你当前脚本同级目录
# 建议将 GLB 文件名填在这里
# MODEL_FILENAME = 'suture_pad.glb' 
OBJ_FILENAME = 'cut.obj' 

class NeedlePickWoundCLF(PsmEnv):
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05),
                        (0.685, 0.745))  # reduce tip pad contact
    SCALING = 5.

    def _env_setup(self):
        super(NeedlePickWoundCLF, self)._env_setup()
        self.has_object = True
        self._waypoint_goal = True

        # robot setup
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
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

        # ==============================================================================
        #                               SUTURE PAD (OBJ LOADER)
        # ==============================================================================
        
        # 1. 确定文件路径
        # 优先在 surrol/assets/ 下找，找不到就在当前脚本目录下找
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
        # 这里的 0.008 是一个经验值，用于将模型调整到合适大小 (约 10cm 宽)
        # 如果你的模型加载出来太大，减小这个数；太小，增大这个数。
        raw_scale = 0.015  # 从 0.008 增大到 0.012，物体变大 50% 
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
        self.cylinder_offset = np.array([0.024, -0.027, -0.001]) # 用于 sample_goal 里的偏移
        self.cyl_radius = 0.05 # 保留用于红点定位

        # 3. 创建视觉形状 (Visual)
        # 注意：PyBullet 会自动加载同名的 .mtl 文件来获取材质信息
        # 如果材质没有正确显示，可能是因为：
        # 1. MTL 文件路径问题
        # 2. PyBullet 版本不支持某些材质属性
        # 3. 材质中使用了 PyBullet 不支持的纹理
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=obj_path,
            meshScale=mesh_scale
            # 不设置 rgbaColor，让它使用 MTL 文件中的材质
            # 如果 MTL 不生效，可以尝试设置 rgbaColor=[0.8, 0.5, 0.4, 1.0] 等
        )

        # 4. 创建碰撞形状 (Collision)
        # 关键标志: p.GEOM_FORCE_CONCAVE_TRIMESH
        # 这让物理引擎知道这是一个凹陷的网格（有伤口洞），而不是一个凸包（像肥皂泡一样包裹住伤口）
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=obj_path,
            meshScale=mesh_scale,
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH
        )
        # 让物体正对着观察者竖起来：尝试不同的旋转角度
        # [np.pi/2, 0, 0] - 绕X轴旋转90度（侧面竖立）
        # [0, np.pi/2, 0] - 绕Y轴旋转90度（在XZ平面竖立）
        # [np.pi/2, 0, np.pi/2] - 组合旋转
        pad_orn = p.getQuaternionFromEuler([np.pi / 2, 0, np.pi])  # 先试X轴，如果不对可改为 [0, np.pi/2, 0] 或其他 
        # 5. 创建多体
        self.cylinder_id = p.createMultiBody(
            baseMass=0, # 0 = 静态物体
            # baseCollisionShapeIndex=collision_shape_id,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=np.array(suture_pad_pos) * self.SCALING,
            baseOrientation=pad_orn
        )
        
        self.obj_ids['obstacle'].append(self.cylinder_id)

        # 调试：检查视觉形状信息
        visual_info = p.getVisualShapeData(self.cylinder_id)
        print(f"[DEBUG] Visual shapes for suture pad: {len(visual_info)} shapes")
        for i, shape in enumerate(visual_info):
            print(f"[DEBUG] Shape {i}: {shape}")

        # ==============================================================================

        # needle 起始位置设置
        yaw = (np.random.rand() - 0.5) * np.pi
        # 位置参数说明：
        # X坐标：workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1  # 中心 ± 0.05，可修改 0.1 来改变随机范围
        # Y坐标：workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1  # 中心 ± 0.05
        # Z坐标：workspace_limits[2][0] + 0.01  # 工作空间底部 + 0.01，修改 0.01 来改变高度
        # needle_start_pos = (
        #     workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,  # X坐标（左右）
        #     workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1,  # Y坐标（前后）
        #     workspace_limits[2][0] + 0.01  # Z坐标（高度）：底部 + 0.01，增大数值让针更高
        # )
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

        # points
        sphere_radius = 0.02
        sphere_pos = [
            workspace_limits[0].mean(), 
            workspace_limits[1].mean(), 
            workspace_limits[2][0] + sphere_radius - 0.03
        ]
        self.sphere_id_1 = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_SPHERE, 
                radius=sphere_radius, 
                rgbaColor=[1, 0, 0, 0.7]
            ),
            basePosition=sphere_pos,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        self.obj_ids['fixed'].append(self.sphere_id_1) # 2

        self.sphere_id_2 = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_SPHERE, 
                radius=sphere_radius, 
                rgbaColor=[1, 0, 0, 0.7]
            ),
            basePosition=sphere_pos,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        self.obj_ids['fixed'].append(self.sphere_id_2) # 3

    def _sample_goal(self) -> np.ndarray:
        workspace_limits = self.workspace_limits1
        goal = np.array([workspace_limits[0].mean() + 0.01 * np.random.randn() * self.SCALING,
                         workspace_limits[1].mean() + 0.01 * np.random.randn() * self.SCALING,
                         workspace_limits[2][1] - 0.04 * self.SCALING])
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
            p.getQuaternionFromEuler([np.pi / 2, 0, np.pi]))  # 与创建时的朝向保持一致
        
        # set points (red markers)
        point1_angle = np.pi/4
        point2_angle = np.pi/4 * 3
        point1_pos = np.array([self.cylinder_center[0] + self.cyl_radius * np.cos(point1_angle),
                    self.cylinder_center[1] + self.cyl_radius * np.sin(point1_angle),
                    self.cylinder_center[2]])
        p.resetBasePositionAndOrientation(
            self.obj_ids['fixed'][2],
            point1_pos,
            p.getQuaternionFromEuler([0, 0, 0]))
        point2_pos = np.array([self.cylinder_center[0] + self.cyl_radius * np.cos(point2_angle),
            self.cylinder_center[1] + self.cyl_radius * np.sin(point2_angle),
            self.cylinder_center[2]])
        p.resetBasePositionAndOrientation(
            self.obj_ids['fixed'][3],
            point2_pos,
            p.getQuaternionFromEuler([0, 0, 0]))

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
    env = NeedlePickWoundCLF(render_mode='human')

    env.test()
    env.close()
    time.sleep(2)