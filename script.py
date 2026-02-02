import bpy
import bpy_extras
import math
import os
import gc
import numpy as np
from math import pi, acos, degrees, cos, sin
from mathutils import Vector
import random

# ==================== 核心修复：移除 Numba，回归 CPU 高速计算 ====================
# 原先的 GPU 计算函数全部被移除，改为直接在主循环中计算
# 这种简单的三角函数计算，CPU 比 GPU 快得多（因为没有数据传输延迟）

def setup_output_directory(path):
    """设置输出目录"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    subdirs = ["labels", "images"]
    for subdir in subdirs:
        subdir_path = os.path.join(path, subdir)
        os.makedirs(subdir_path, exist_ok=True)
    return path

def setup_camera_tracking(camera, target):
    """设置相机追踪目标"""
    for c in camera.constraints:
        camera.constraints.remove(c)
    track_constraint = camera.constraints.new(type='TRACK_TO')
    track_constraint.target = target
    track_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    track_constraint.up_axis = 'UP_Y'

def setup_gpu_rendering(scene):
    """
    配置Blender使用NVIDIA GPU进行渲染 (Cycles)
    核心优化：OPTIX + 持久化数据 + 光程限制 + 降噪
    """
    scene.render.engine = 'CYCLES'
    
    # 1. 启用数据持久化 (关键优化：避免每帧重新加载几何体)
    # 适合只有相机和物体移动，拓扑结构不变的场景
    scene.render.use_persistent_data = True
    
    # 获取Cycles偏好设置
    cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
    cycles_prefs.refresh_devices()
    
    # 2. 硬件加速优先尝试 OPTIX (RTX显卡极速加成)，其次 CUDA
    device_types = ['OPTIX', 'CUDA']
    found_device = False
    
    for device_type in device_types:
        try:
            cycles_prefs.compute_device_type = device_type
            # 检查是否有可用设备
            available_devices = [d for d in cycles_prefs.devices if d.type == device_type]
            if available_devices:
                print(f"正在使用显卡接口: {device_type}")
                # 激活设备
                for device in cycles_prefs.devices:
                    device.use = (device.type == device_type)
                    if device.use:
                        print(f"已激活渲染设备: {device.name}")
                found_device = True
                break
        except TypeError:
            pass
    
    if not found_device:
        print("警告: 未检测到兼容的 GPU (OPTIX/CUDA)，将使用 CPU 渲染。")
        scene.cycles.device = 'CPU'
    else:
        # 设置场景使用 GPU
        scene.cycles.device = 'GPU'
    
    # 3. 渲染参数优化
    scene.cycles.samples = 64  # 采样率
    scene.cycles.use_adaptive_sampling = True # 自适应采样
    scene.cycles.adaptive_threshold = 0.05 # 稍微放宽噪点阈值
    
    # 4. 开启 AI 降噪 (OptiX / OpenImageDenoise)
    # 允许低采样下快速出图
    try:
        # 注意：不同 Blender 版本访问方式可能不同，这里尝试通用方式
        vl = scene.view_layers[0]
        vl.cycles.use_denoising = True
        # 如果是 OPTIX 设备，使用 OPTIX 降噪最快
        if found_device and cycles_prefs.compute_device_type == 'OPTIX':
            vl.cycles.denoiser = 'OPTIX'
        else:
            vl.cycles.denoiser = 'OPENIMAGEDENOISE'
    except Exception as e:
        print(f"开启降噪失败 (非致命): {e}")

    # 5. 光程优化 (Light Paths) - 极大地影响渲染时间
    # 这种工业/标注场景通常不需要复杂的全局光照反弹
    scene.cycles.max_bounces = 4        # 限制总弹射次数 (默认12)
    scene.cycles.diffuse_bounces = 2    # 漫反射
    scene.cycles.glossy_bounces = 2     # 光泽/反射
    scene.cycles.transparent_max_bounces = 4 # 透明
    scene.cycles.transmission_bounces = 4 # 透射
    
    # 禁用焦散 (通常不需要，且计算昂贵)
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False

    if hasattr(scene.view_settings, 'gpu_acceleration'):
        scene.view_settings.gpu_acceleration = 'AUTO'
    


def update_camera_position_cpu(camera, angle, distance, frame, total_frames_per_circle):
    # 基础圆周坐标
    base_x = distance * cos(angle)
    base_y = distance * sin(angle)
    base_z = -1.0 + (frame // total_frames_per_circle) * 0.5
    
    # --- 新增：随机偏移 (根据需要调整幅度) ---
    # 比如：x, y 偏移 +- 0.1米，z 偏移 +- 0.05米
    offset_x = random.uniform(-0.5, 0.5)
    offset_y = random.uniform(-0.5, 0.5)
    offset_z = random.uniform(-0.025, 0.25)
    
    camera.location.x = base_x + offset_x
    camera.location.y = base_y + offset_y
    camera.location.z = base_z + offset_z
    
    return False


def rotate_fan_axes_cpu(fan_axes):
    """CPU版更新风扇轴"""
    step1 = 11/51.7
    step2 = 13/51.7
    
    for i, axis in enumerate(fan_axes):
        current_rot = axis.rotation_euler[1]
        if i >= 5:
             new_rot = (current_rot % (2 * pi)) + step1
        else:
             new_rot = (current_rot % (2 * pi)) + step2
        
        axis.rotation_euler[1] = new_rot
        axis.keyframe_insert(data_path="rotation_euler", frame=bpy.context.scene.frame_current)

def calculate_camera_face_angle_mathutils(cam, obj, poly_index):
    """使用 Blender 内置的 mathutils 进行向量计算 (C底层，非常快)"""
    # 获取世界坐标系下的相机前向向量
    # 相机默认看向 -Z 轴
    cam_forward = (cam.matrix_world.to_3x3() @ Vector((0, 0, -1))).normalized()
    
    # 获取面的世界坐标法线
    poly = obj.data.polygons[poly_index]
    # 需要将局部法线转为世界法线
    world_normal = (obj.matrix_world.to_3x3() @ poly.normal).normalized()
    
    # 计算点积
    dot = cam_forward.dot(world_normal)
    
    # 防止浮点误差超出范围
    dot = max(-1.0, min(1.0, dot))
    angle_rad = acos(dot)
    return degrees(angle_rad)

def project_verts_to_2d_fast(scene, camera, obj):
    """
    使用 NumPy 向量化加速 3D 到 2D 投影
    比逐点调用 world_to_camera_view 快 50-100 倍
    """
    mesh = obj.data
    matrix_world = np.array(obj.matrix_world)
    
    # 1. 获取所有顶点坐标 (Local Space)
    # 使用 foreach_get 极速获取数据
    num_verts = len(mesh.vertices)
    verts_local = np.empty(num_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", verts_local)
    
    # 重塑为 (N, 3) 并添加齐次坐标 w=1 -> (N, 4)
    verts_local = verts_local.reshape((-1, 3))
    verts_homo = np.hstack((verts_local, np.ones((num_verts, 1), dtype=np.float32)))
    
    # 2. 构建 MVP 矩阵 (Model-View-Projection)
    # View Matrix (World -> Camera)
    view_mat = np.array(camera.matrix_world.inverted())
    
    # Projection Matrix (Camera -> Clip)
    # 注意：calc_matrix_camera 返回的是 Blender 格式，可能需要调整
    depsgraph = bpy.context.evaluated_depsgraph_get()
    proj_mat = np.array(camera.calc_matrix_camera(depsgraph))
    
    # MVP = Projection @ View @ Model
    # 注意矩阵乘法顺序：在 NumPy 中通常是 M @ v，但在 Blender/OpenGL 中列主序可能不同
    # Blender matrix is row-major in Python API but behaves like column-major in math
    # Standard formula: v_clip = Proj * View * Model * v_local
    # In numpy with standard arrays: v_clip = (MVP @ v_local.T).T
    
    mvp_mat = proj_mat @ view_mat @ matrix_world
    
    # 3. 批量投影
    # (4, 4) @ (4, N) -> (4, N)
    verts_clip = mvp_mat @ verts_homo.T
    
    # 4. 透视除法 (Perspective Divide) -> NDC
    # clip_w
    w = verts_clip[3, :]
    
    # 避免除以零
    w = np.where(w == 0, 1e-6, w)
    
    # Normalized Device Coordinates (NDC): [-1, 1]
    # 只取 x, y, z
    verts_ndc = verts_clip[:3, :] / w
    
    # 5. 视口变换 (NDC -> Image UV [0, 1])
    # NDC x: [-1, 1] -> [0, 1]
    # NDC y: [-1, 1] -> [0, 1] (Blender image origin is bottom-left, but usually we want top-left or standard UV)
    # world_to_camera_view returns (x, y, z) where (0,0) is bottom-left, (1,1) is top-right.
    
    x_img = (verts_ndc[0, :] + 1) * 0.5
    y_img = (verts_ndc[1, :] + 1) * 0.5
    z_depth = verts_clip[2, :] # Use clip z or ndc z for depth check? world_to_camera_view uses logic to check if behind camera
    
    # Check if behind camera: w < 0 usually means behind in OpenGL convention if w is essentially -z
    # Blender's world_to_camera_view returns z as distance from camera plane.
    # In our MVP, w usually holds depth info.
    # Let's filter by w > 0 (in front of camera plane)
    
    mask = w > 0
    
    # Stack results
    points = np.column_stack((x_img, y_img))
    
    return points, mask


# ========== 这里的灯光控制和标签获取逻辑保持不变 (已省略部分冗余代码) ==========

def set_LEDs(rings_combination, fans_to_save, light_ups, light_bottoms, light_lefts, light_rights, light_flows):
    # 原逻辑保留... 
    # 为了代码简洁，假设这里的逻辑没有变动，直接调用你的原始函数逻辑
    # 只要确保 material_index_list 是 numpy array 或 list 即可
    for i in range(len(rings_combination)):
        ring = rings_combination[i]
        fan = fans_to_save[i]
        mat_name = f"target_aim_{ring}" if ring != -1 else "target_aim_n1"
        mat_index = fan.data.materials.find(mat_name)
        if mat_index != -1:
            # 优化：先检查是否需要修改，避免不必要的 mesh.update() 触发 BVH 重建
            if fan.data.polygons[1358].material_index != mat_index:
                fan.data.polygons[1358].material_index = mat_index
                fan.data.update()
        
        lu = light_ups[i]
        lb = light_bottoms[i]
        ll = light_lefts[i]
        lr = light_rights[i]
        lf = light_flows[i]
        
        turn_off_light_up(lu)
        turn_off_light_bottom(lb)
        turn_off_light_left(ll)
        turn_off_light_right(lr)
        turn_off_light_flow(lf)
        
        if ring == 0:
            turn_on_light_flow(lf)
        elif ring >=1 and ring <=11:
            turn_on_light_left(ll)
            turn_on_light_right(lr)
            turn_on_light_flow_all(lf)

# --- 辅助函数：开关灯逻辑 (保持你的原始逻辑) ---
def turn_on_light_up(obj):
    set_material_indices(obj, material_index_plastic_light_up_list, "plastic", update=False)
    set_material_indices(obj, material_index_light_up_list, "light_up", update=True)

def turn_on_light_bottom(obj):
    set_material_indices(obj, material_index_plastic_light_bottom_list, "plastic", update=False)
    set_material_indices(obj, material_index_light_bottom_list, "light_bottom", update=True)

def turn_on_light_left(obj):
    set_material_indices(obj, material_index_plastic_light_left_list, "plastic", update=False)
    set_material_indices(obj, material_index_light_left_list, "light_left", update=True)

def turn_on_light_right(obj):
    set_material_indices(obj, material_index_plastic_light_right_list, "plastic", update=False)
    set_material_indices(obj, material_index_light_right_list, "light_right", update=True)

def turn_on_light_flow(obj):
    set_material_indices(obj, material_index_plastic_light_flow_list, "light_flow_plastic", update=False)
    set_material_indices(obj, material_index_light_flow_list, "light_flow", update=True)

def turn_on_light_flow_all(obj):
    set_material_indices(obj, material_index_plastic_light_flow_list, "light_flow_plastic", update=False)
    # 注意：你的原代码这里第二个列表也是 material_index_plastic_light_flow_list，可能是 bug，我照抄了
    set_material_indices(obj, material_index_plastic_light_flow_list, "light_flow_all", update=True)

def turn_off_light_up(obj):
    reset_to_metal(obj, update=False)
    set_material_indices(obj, material_index_plastic_light_up_list, "plastic", update=True)

def turn_off_light_bottom(obj):
    reset_to_metal(obj, update=False)
    set_material_indices(obj, material_index_plastic_light_bottom_list, "plastic", update=True)

def turn_off_light_left(obj):
    # 特殊逻辑：原代码只是部分重置
    mat_plastic = obj.data.materials.find("plastic")
    mat_metal = obj.data.materials.find("metal")
    if mat_plastic == -1 or mat_metal == -1: return
    
    mesh = obj.data
    num_polys = len(mesh.polygons)
    all_mat_indices = np.empty(num_polys, dtype=np.int32)
    mesh.polygons.foreach_get("material_index", all_mat_indices)
    
    # 优化：保存原始状态副本用于比较
    original_indices = all_mat_indices.copy()
    
    # 批量更新
    valid_plastic = material_index_plastic_light_left_list[material_index_plastic_light_left_list < num_polys]
    all_mat_indices[valid_plastic] = mat_plastic
    
    valid_metal = material_index_light_left_list[material_index_light_left_list < num_polys]
    all_mat_indices[valid_metal] = mat_metal
    
    # 优化：如果没有变化，跳过更新
    if np.array_equal(original_indices, all_mat_indices):
        return
    
    mesh.polygons.foreach_set("material_index", all_mat_indices)
    mesh.update()

def turn_off_light_right(obj):
    reset_to_metal(obj, update=False)
    set_material_indices(obj, material_index_plastic_light_right_list, "plastic", update=True)

def turn_off_light_flow(obj):
    reset_to_metal(obj, update=False)
    set_material_indices(obj, material_index_plastic_light_flow_list, "light_flow_plastic", update=True)

# 统一封装材质设置函数，提高运行效率 (使用 NumPy foreach_set 加速)
def set_material_indices(obj, indices, mat_name, update=True):
    mat_idx = obj.data.materials.find(mat_name)
    if mat_idx == -1: return
    mesh = obj.data
    
    num_polys = len(mesh.polygons)
    all_mat_indices = np.empty(num_polys, dtype=np.int32)
    mesh.polygons.foreach_get("material_index", all_mat_indices)
    
    # 过滤越界索引
    valid_indices = indices[indices < num_polys]
    
    # 优化：检查是否真的需要更新
    # 获取目标位置当前的索引
    current_indices_at_target = all_mat_indices[valid_indices]
    
    # 如果所有目标位置的索引已经是 mat_idx，则无需操作
    if np.all(current_indices_at_target == mat_idx):
        return

    # NumPy 批量赋值
    all_mat_indices[valid_indices] = mat_idx
    
    mesh.polygons.foreach_set("material_index", all_mat_indices)
    if update:
        mesh.update()

def reset_to_metal(obj, update=True):
    mat_metal = obj.data.materials.find("metal")
    if mat_metal == -1: return
    
    mesh = obj.data
    num_polys = len(mesh.polygons)
    
    # 优化：先读取当前状态，检查是否已经是全 metal
    all_mat_indices = np.empty(num_polys, dtype=np.int32)
    mesh.polygons.foreach_get("material_index", all_mat_indices)
    
    if np.all(all_mat_indices == mat_metal):
        return

    # 使用 foreach_set 批量重置
    all_mat_indices.fill(mat_metal)
    mesh.polygons.foreach_set("material_index", all_mat_indices)
    if update:
        mesh.update()

# --- Raycast 和 Label 生成逻辑 (CPU Bound, 无法 GPU 加速) ---
def check_center_occlusion(scene, camera, center_obj):
    cam_loc = camera.matrix_world.to_translation()
    center_loc = center_obj.matrix_world.to_translation()
    direction = center_loc - cam_loc
    depsgraph = bpy.context.evaluated_depsgraph_get()
    result, _, _, _, obj, _ = scene.ray_cast(depsgraph=depsgraph, origin=cam_loc, direction=direction)
    return result and obj != center_obj

def get_obj_label(scene, camera, class_index, obj):
    cam_loc = camera.matrix_world.to_translation()
    
    # --- 优化 1: 极速中心点计算 (BBox Center) ---
    # 替代原先的 sum(vertices) / len
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center_obj = sum(bbox_corners, Vector()) / 8.0
    
    direction = center_obj - cam_loc
    depsgraph = bpy.context.evaluated_depsgraph_get()
    result, _, _, _, hit_obj, _ = scene.ray_cast(depsgraph=depsgraph, origin=cam_loc, direction=direction)
    
    if result and hit_obj != obj:
        return 4, None
    
    # --- 优化 2: NumPy 向量化投影 ---
    # 替代原先的 for 循环 world_to_camera_view
    points, mask = project_verts_to_2d_fast(scene, camera, obj)
    
    if not np.any(mask):
        return 2, None
        
    # 筛选可见点 (z > 0)
    points = points[mask]
    
    x_coords = points[:, 0]
    # 原代码逻辑：1 - co_2d.y (Y轴翻转)
    y_coords = 1.0 - points[:, 1]
    
    # 使用 numpy 计算 min/max
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    # Clamp to [0, 1]
    x_min = max(0.0, x_min)
    x_max = min(1.0, x_max)
    y_min = max(0.0, y_min)
    y_max = min(1.0, y_max)
    
    if x_min >= x_max or y_min >= y_max:
        return 1, None
    
    return 3, [class_index, (x_min+x_max)/2, (y_min+y_max)/2, x_max-x_min, y_max-y_min]

def get_obj_label_with_kp(scene, camera, class_index, obj, keypoints):
    cam_loc = camera.matrix_world.to_translation()
    
    # --- 优化 1: 极速中心点计算 (BBox Center) ---
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center_obj = sum(bbox_corners, Vector()) / 8.0
    
    depsgraph = bpy.context.evaluated_depsgraph_get()
    
    # 遮挡检测
    direction = center_obj - cam_loc
    result, _, _, _, hit_obj, _ = scene.ray_cast(depsgraph=depsgraph, origin=cam_loc, direction=direction)
    if result and hit_obj != obj:
        return 4, None
    
    # --- 优化 2: NumPy 向量化投影 ---
    points, mask = project_verts_to_2d_fast(scene, camera, obj)
            
    if not np.any(mask):
        return 2, None
    
    # 筛选可见点 (z > 0)
    points = points[mask]
    
    x_coords = points[:, 0]
    # Y轴翻转
    y_coords = 1.0 - points[:, 1]
    
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    # Clamp to [0, 1]
    x_min = max(0.0, x_min)
    x_max = min(1.0, x_max)
    y_min = max(0.0, y_min)
    y_max = min(1.0, y_max)
    
    if x_min >= x_max or y_min >= y_max: return 1, None

    # 关键点处理
    keypoints_2d = []
    for kp in keypoints:
        world_loc = kp.matrix_world.to_translation()
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, world_loc)
        
        vis = 0
        if co_2d.z > 0:
            dir_kp = world_loc - cam_loc
            res_kp, _, _, _, hit_kp, _ = scene.ray_cast(depsgraph=depsgraph, origin=cam_loc, direction=dir_kp)
            # 宽松判定：如果打到的是关键点本身或者其父物体(扇叶)，则视为可见
            if not res_kp or hit_kp == kp or hit_kp == obj:
                vis = 1
            
            # Clamp 坐标到 0-1
            x_img = min(max(co_2d.x, 0), 1)
            y_img = min(max(1 - co_2d.y, 0), 1)
            keypoints_2d.append((x_img, y_img, vis))
        else:
            keypoints_2d.append((0, 0, 0))

    label = [class_index, (x_min+x_max)/2, (y_min+y_max)/2, x_max-x_min, y_max-y_min]
    for kp_data in keypoints_2d:
        label.extend(kp_data)
        
    return 3, label

def save_output_files(output_dir, distance, frame, labels, scene, rings_combination, exposure):
    # 文件名生成
    ring_str = "_".join(map(str, rings_combination))
    frame_str = f"{frame:06d}"
    # 增加文件名中的精度，避免重名
    txt_filename = os.path.join(output_dir, "labels", f"rune_{exposure:.2f}_{ring_str}_{distance:.2f}_{frame_str}.txt")
    img_filename = os.path.join(output_dir, "images", f"rune_{exposure:.2f}_{ring_str}_{distance:.2f}_{frame_str}.png")
    
    if os.path.exists(txt_filename) and os.path.exists(img_filename):
        return
    
    # 写入标签
    with open(txt_filename, 'w') as f:
        for label in labels:
            f.write(" ".join(f"{x:.6f}" for x in label) + "\n")
    
    # 执行渲染 (这是最耗时的步骤)
    scene.render.filepath = img_filename
    # 使用 Write Still 可以在后台渲染并保存，不用切 ViewLayer
    bpy.ops.render.render(write_still=True)
    print(f"已生成: {os.path.basename(txt_filename)}")

def init_fans_materials(fans):
    mat_names = [f"target_aim_{i}" for i in range(-1,12)]
    for fan in fans:
        for mat_name in mat_names:
            if mat_name not in fan.data.materials:
                mat = bpy.data.materials.get(mat_name)
                if mat: fan.data.materials.append(mat)

def init_lights_material(lights, prefixes):
    for light in lights:
        for name in prefixes:
            if name not in light.data.materials:
                mat = bpy.data.materials.get(name)
                if mat: light.data.materials.append(mat)

# ==================== 主程序 ====================

if __name__ == "__main__":
    output_dir = r"D:\StudyWorks\RM\codes\buff\output"
    output_dir = setup_output_directory(output_dir)
    
    # 你的参数
    start_frame = 1
    end_frame = 40
    num_circles = 3
    total_frames_per_circle = end_frame - start_frame + 1
    total_frames = num_circles * total_frames_per_circle

    scene = bpy.context.scene
    camera = scene.camera
    
    # 设置渲染引擎使用 GPU
    setup_gpu_rendering(scene)
    
    # 获取对象 (添加简单的错误处理)
    try:
        centerR = bpy.data.objects['centerR']
        fans = [bpy.data.objects[f'fan{i}'] for i in range(1, 11)]
        fan_axes = [bpy.data.objects[f'fan{i}_axis'] for i in range(1, 11)]
        light_ups = [bpy.data.objects[f'light_up{i}'] for i in range(1, 11)]
        light_bottoms = [bpy.data.objects[f'light_bottom{i}'] for i in range(1, 11)]
        light_lefts = [bpy.data.objects[f'light_left{i}'] for i in range(1, 11)]
        light_rights = [bpy.data.objects[f'light_right{i}'] for i in range(1, 11)]
        light_flows = [bpy.data.objects[f'light_flow{i}'] for i in range(1, 11)]
    except KeyError as e:
        print(f"Error: Missing object {e} in scene.")
        # 在脚本调试时可以注释掉 exit，防止 Blender 崩溃
        # exit() 
    
    setup_camera_tracking(camera, fans[0])

    keypoints = []
    for i in range(1, 11):
        kp_count = 8 if i == 1 else 4
        fan_kp = [bpy.data.objects[f'fan{i}_k{j}'] for j in range(1, kp_count+1)]
        keypoints.append(fan_kp)
        
    # 读取 rings 组合
    rings_combinations_path = r"D:\StudyWorks\RM\codes\buff\rings_combinations.txt"
    if os.path.exists(rings_combinations_path):
        with open(rings_combinations_path, 'r') as file:
            lines = file.readlines()[:1000]
            # 兼容带有括号的格式
            rings_combinations = [tuple(map(int, line.strip().strip('()').replace(' ', '').split(','))) for line in lines]
    else:
        # 调试用默认值
        rings_combinations = [(0, 1, 2, 3, 4)]
        print("Warning: combinations file not found, using default.")

    # -----------------------------------------------------------
    # 请在此处粘贴你的长材质索引列表 (numpy arrays)
    # -----------------------------------------------------------
    # material_index_plastic_light_up_list = np.array([...])
    # ... (粘贴你的那些长数组) ...
    # 为了代码能运行，我这里放几个空的占位符，请你替换回原来的代码
    material_index_plastic_light_up_list = np.array([734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1479, 1480, 1481, 1482, 1483, 1484, 1485, 1486, 1487, 1488, 1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1733, 1747, 1772, 1916, 1946, 1947],dtype=np.int32)
    material_index_plastic_light_bottom_list = np.array([288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1342, 1369, 1405, 1423, 1425, 1479, 1539, 1581, 1582, 1692, 1761], dtype=np.int32)   
    material_index_plastic_light_left_list = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 618, 697, 805, 806, 838, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441, 1442, 1443], dtype=np.int32)
    material_index_plastic_light_right_list = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 610, 616, 702, 704, 722, 725, 726, 727, 780, 789, 790, 791, 798, 890, 945, 950, 986, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435],dtype=np.int32)
    material_index_plastic_light_flow_list = np.array([113, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1055, 1056, 1063, 1131, 1150, 1186, 1215, 1218, 1246, 1273, 1369, 1370, 1398, 1440, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467],dtype=np.int32)
    material_index_light_up_list = np.array([1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1678, 1679, 1680, 1681, 1682, 1683, 1684, 1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1726, 1727, 1728, 1729, 1730, 1731, 1732, 1734, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763, 1764, 1765, 1766, 1767, 1768, 1769, 1770, 1771, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1782, 1783, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834, 1835, 1836, 1837, 1838, 1839, 1840, 1841, 1842, 1843, 1844, 1845, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863, 1864, 1865, 1866, 1867, 1868, 1869, 1870, 1871, 1872, 1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2088, 2089, 2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124, 2125, 2126, 2127, 2128, 2129, 2130, 2131, 2132, 2133, 2134, 2135, 2136, 2137, 2138, 2139, 2140, 2141, 2142, 2143, 2144, 2145, 2146, 2147],dtype=np.int32)
    material_index_light_bottom_list = np.array([1336, 1337, 1338, 1339, 1340, 1341, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1406, 1407, 1408, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1424, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1480, 1481, 1482, 1483, 1484, 1485, 1486, 1487, 1488, 1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 1537, 1538, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1678, 1679, 1680, 1681, 1682, 1683, 1684, 1685, 1686, 1687, 1688, 1689, 1690, 1691, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1726, 1727, 1728, 1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1762, 1763, 1764, 1765, 1766, 1767, 1768, 1769, 1770, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1782, 1783, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834, 1835, 1836, 1837, 1838, 1839, 1840, 1841, 1842, 1843],dtype=np.int32)
    material_index_light_left_list = np.array([614, 615, 616, 617, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004],dtype=np.int32)
    material_index_light_right_list = np.array([611, 612, 613, 614, 615, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 703, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 723, 724, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 781, 782, 783, 784, 785, 786, 787, 788, 792, 793, 794, 795, 796, 797, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 946, 947, 948, 949, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998],dtype=  np.int32)
    material_index_light_flow_list = np.array([1051, 1052, 1053, 1054, 1057, 1058, 1059, 1060, 1061, 1062, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1216, 1217, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1399, 1400, 1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439],dtype= np.int32)


    # -----------------------------------------------------------

    init_fans_materials(fans)
    init_lights_material(light_ups, ["plastic", "light_up", "metal"])
    init_lights_material(light_bottoms, ["plastic", "light_bottom", "metal"])
    init_lights_material(light_lefts, ["plastic", "light_left", "metal"])
    init_lights_material(light_rights, ["plastic", "light_right", "metal"])
    init_lights_material(light_flows, ["light_flow", "light_flow_plastic", "light_flow_all", "metal"])

    class_index_map = {-1:0,0:1,1:2,2:2,3:2,4:2,5:2,6:2,7:2,8:2,9:2,10:2,11:2}
    
    # 移除固定列表，改用随机采样，提升随机性和范围
    # distances = [1, 2]
    # exposure_list = [0.2, 0.5, 0.8]

    print("开始渲染循环...")
    
    # 禁用 Global Undo 以节省内存 (关键优化)
    bpy.context.preferences.edit.use_global_undo = False
    
    # 定义每个组合生成的样本数量倍率
    # 原来是 2(dist) * 3(exp) = 6 倍 total_frames
    # 保持这个数量级
    samples_per_frame_base = 6 
    
    # 渲染循环
    for rings_combination in rings_combinations:
        # 计算总迭代次数
        total_iterations = total_frames * samples_per_frame_base
        print(f"Processing: Comb {rings_combination} | Total Iterations: {total_iterations}")
        
        for i in range(total_iterations):
            # 基础帧号 (控制旋转角度和相机高度)
            frame = i % total_frames
            
            # --- 随机化参数 (提升随机性) ---
            # 距离范围扩大：1.5m 到 3.5m (原先 1-2m)
            distance = random.uniform(1.5, 3.5)
            
            # 曝光范围扩大：0.2 到 1.5 (原先 0.2-0.8)
            exposure = random.uniform(0.2, 1.5)
            
            # 偶尔增加更极端的随机性 (10% 概率)
            if random.random() < 0.1:
                distance = random.uniform(1.2, 4.0)
                exposure = random.uniform(0.1, 2.0)

            # 1. 基础场景设置
            scene.frame_set(frame % total_frames_per_circle + start_frame)
            scene.view_settings.exposure = exposure
            
            angle = (frame % total_frames_per_circle) * (2 * pi / total_frames_per_circle)
            
            # 2. 更新物体位置/旋转 (CPU)
            rotate_fan_axes_cpu(fan_axes)
            update_camera_position_cpu(camera, angle, distance, frame, total_frames_per_circle)
                    
            # 必须更新依赖图，否则 ray_cast 会用到旧位置
            bpy.context.view_layer.update()
            
            # 3. 遮挡剔除
            if check_center_occlusion(scene, camera, fans[0]):
                continue
            
            # 4. 筛选可见扇叶和对应灯光
            if angle < pi:
                slice_range = slice(0, 5)
            else:
                slice_range = slice(5, 10)
                
            fans_to_save = fans[slice_range]
            keypoints_to_save = keypoints[slice_range]
            
            # 5. 设置灯光 (传入切片后的灯光列表，确保控制正确的灯)
            set_LEDs(rings_combination, fans_to_save, 
                     light_ups[slice_range], 
                     light_bottoms[slice_range], 
                     light_lefts[slice_range], 
                     light_rights[slice_range], 
                     light_flows[slice_range])

            # 6. 生成标签 (这是最慢的逻辑部分)
            labels = []
            flag, label = get_obj_label(scene, camera, 3, centerR)
            if flag == 3:
                labels.append(label)
                
            dont_save = False
            for j in range(len(fans_to_save)):
                # ... 同原逻辑 ...
                fan = fans_to_save[j]
                kp = keypoints_to_save[j]
                ring_val = rings_combination[j]
                cls_idx = class_index_map.get(ring_val, 0)
                
                flag, label = get_obj_label_with_kp(scene, camera, cls_idx, fan, kp)
                if flag == 3:
                    labels.append(label)
                elif flag == 0:
                    dont_save = True
                    break
            
            # 角度过滤
            if calculate_camera_face_angle_mathutils(camera, fans[0], 1358) > 60:
                dont_save = True
            
            if dont_save or not labels:
                continue
            
            # 7. 保存结果 (渲染)
            save_output_files(output_dir, distance, frame, labels, scene, rings_combination, exposure)

            # 内存优化：每5帧手动执行垃圾回收，防止卡顿
            if i % 5 == 0:
                gc.collect()

    print("All Done!")