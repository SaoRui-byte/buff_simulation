# buff_simulation
blender 里的模型有一个是调好参数的，外面单独的两个txt文件，是代码跑出来的，模型里的script.py和另一个代码都是修改过的，理论上讲，只要路径改一下就可以运行*

## [Optimized Version] - 2026-02-02

### 🚀 性能优化 (Performance)

#### 1. 渲染引擎优化 (Rendering Pipeline)
- **启用 OptiX 硬件加速**：针对 NVIDIA RTX 显卡开启 OptiX 接口，显著提升光线追踪速度。
- **启用数据持久化 (`use_persistent_data`)**：缓存静态几何体和 BVH 数据，避免每一帧重新加载场景，大幅减少帧与帧之间的空闲时间。
- **光程限制 (Light Paths)**：将最大光线反弹次数 (`max_bounces`) 从默认值降低到 4，漫反射/光泽/透射限制为 2-4 次，移除焦散计算。在保证标注数据视觉质量的前提下，极大缩短单帧渲染时间。
- **AI 降噪**：启用 OptiX/OpenImageDenoise 降噪器，允许在较低采样率（64 samples）下获得干净图像。

#### 2. 几何算法加速 (Geometry & Math)
- **NumPy 向量化投影 (`project_verts_to_2d_fast`)**：
  - 移除了旧版中对每个顶点循环调用 `world_to_camera_view` 的低效逻辑。
  - 实现了基于 MVP (Model-View-Projection) 矩阵的 NumPy 向量化运算，一次性计算数千个顶点的 2D 投影坐标。
  - **性能提升**：3D 转 2D 投影计算速度提升约 50-100 倍。
- **极速包围盒计算**：
  - 使用物体的 8 个包围盒角点计算中心，替代了旧版遍历所有顶点求平均值的做法。

#### 3. 场景更新机制 (Scene Update & BVH)
- **材质更新“脏检查” (Dirty Checks)**：
  - 在 `set_material_indices`, `reset_to_metal`, `turn_off_light_left` 等函数中增加了状态检测。
  - **关键改进**：只有当目标材质索引确实发生变化时，才执行 `foreach_set` 和 `mesh.update()`。
  - **效果**：消除了绝大多数帧中不必要的几何体更新，防止 Cycles 引擎在视角变化时反复重建 BVH，彻底解决了“视角大幅变化时卡顿”的问题。

#### 4. 内存管理 (Memory Management)
- **禁用全局撤销 (`use_global_undo = False`)**：在脚本模式下禁用 Blender 的撤销系统，防止内存随运行时间无限膨胀。
- **主动垃圾回收 (`gc.collect`)**：在主循环中每 5 帧手动触发一次 Python 垃圾回收，保持内存占用平稳。

### 🐛 Bug 修复 (Bug Fixes)
- **循环变量遮蔽修复**：修复了主循环中内部循环变量使用 `i` 覆盖外部循环变量 `i` 的问题（改为 `j`），确保了遍历逻辑的正确性。
- **材质索引数组化**：将硬编码的材质更新循环改为 NumPy `foreach_set` 批量操作，避免了 Python 层面的循环开销。

### ✨ 功能增强 (Features)
- **增强的随机性**：
  - **相机位姿**：增加了相机位置的随机微小偏移 (`offset_x`, `offset_y`, `offset_z`)，模拟更真实的拍摄抖动。
  - **参数范围**：扩大了 `distance` (1.5m - 3.5m) 和 `exposure` (0.2 - 1.5) 的随机采样范围，增加了数据集的多样性。
