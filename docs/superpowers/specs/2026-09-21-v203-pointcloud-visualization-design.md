# V203 点云可视化设计

## 目标

将最新完整 V203 运行产生的 `tensor_map.npz` 转换为可复用的彩色 PLY 点云，并生成可在浏览器中离线打开的交互式 3D HTML 页面。

## 输入

- 文件：`Results/SparseORBLoop_V203_alpha10000_t030/20260919_181925_f362b2/window_orb_loop_sparse_t030/outputs/MACVO-Fast-WindowICP5-ORBLoop-Sparse-t030@V203/09_19_181927/tensor_map.npz`
- 坐标：`points//pos_Tw`，形状为 `(363706, 3)`
- 颜色：`points//color`，形状为 `(363706, 3)`

## 处理流程

1. 读取世界坐标与 RGB 颜色，检查数组长度一致。
2. 删除坐标中含 `NaN` 或无穷值的点。
3. 用到坐标中位数的欧氏距离做稳健分位数过滤，去除最远端 0.5% 的明显离群点；不改变其余点的位置和颜色。
4. 将过滤后的全部点以 ASCII PLY 导出，便于 CloudCompare、MeshLab 和 Open3D 复用。
5. 从过滤后的点云中使用固定随机种子最多抽样 200,000 个点，生成 Plotly 离线 HTML，平衡细节与浏览器流畅度。

## 交互与视觉

- 鼠标拖动旋转，滚轮缩放，右键拖动平移。
- 使用点云的原始 RGB 颜色，深色背景，坐标轴等比例显示。
- 标题显示 V203、可视化点数和全量 PLY 点数。
- HTML 内嵌 Plotly，打开后不依赖互联网。

## 输出

- `Results/V203_pointcloud_latest.ply`：过滤后的全量彩色点云。
- `Results/V203_pointcloud_3d.html`：浏览器交互式可视化。

## 验证

- PLY 头部的顶点数与实际数据行数一致，所有坐标有限，RGB 在 `[0, 255]` 内。
- HTML 文件存在且非空，包含预期的 Plotly 3D 数据和 V203 标题。
- 启动本地 HTTP 服务后，确认页面返回 HTTP 200。

## 错误处理

若 NPZ 缺少坐标或颜色键、形状不匹配、过滤后无点或输出失败，程序应立即停止并给出明确错误，不留下看似有效的不完整结果。
