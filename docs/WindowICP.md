# MAC-VO 五帧滑动窗口 ICP v0.1

## 版本与范围

- 开发分支：experiment/sliding-window-icp-v1。
- 两帧实验基线：5542d91；标签 baseline/fast-icp-euroc-20260912。
- 求解器首次提交：c0f8d55；前端接入：553913d。
- 已验证代码提交：a0f49b4。
- 完成标签：window-icp-v0.1。
- 当前主目录位于开发分支，main未合并；未推送远程。
- 旧论文、译稿、可视化和用户已有报告不纳入本次提交。旧结果不覆盖。

本版本实现位姿级多帧ICP，不是联合优化地图点的BA，不依赖BAE。
网络仍使用原有FlowFormerCov权重、encoder FP16、decoder BF16、12轮解码、最多200个点、CovAwareSelector_NoDepth及MatchCovariance。
依赖环境不变：Python 3.12，torch 2.4.1+cu124，pypose 0.9.5。

## 配置和数据流

新配置：[MACVO_Fast_WindowICP.yaml](../Config/Experiment/MACVO/MACVO_Fast_WindowICP.yaml)。

~~~yaml
Odometry:
  type: WindowMACVO
  name: MACVO-Fast-WindowICP5
  args:
    window_size: 5
    skip_matching: true
    window_iterations: 10
    window_huber_delta: 3.0
  optimizer:
    type: TwoFrame_PGO
    args:
      graph_type: icp
      autodiff: false
      parallel: false
~~~

optimizer字段指定的是同步两帧ICP初值器。WindowMACVO先执行它，再执行自己的窗口联合求解；不是把旧优化器的frames参数扩成5就结束。

完整步骤：

1. 旧前端对 t-1→t 推理，生成当前双目深度和相邻匹配。
2. 旧两帧ICP根据完整三维协方差给当前帧一个位姿初值。
3. 立即写回并清空初值器结果，避免下次回写覆盖窗口结果。
4. 缓存最近5帧图像、深度和局部三维观测。
5. 用FlowFormerCov直接计算 t-2→t，独立选点、采样深度并生成三维协方差。
6. 窗口内建立相邻与间隔一帧的边，最多4+3=7条，每条最多200个观测。
7. 固定窗口最早位姿，在其局部坐标系联合优化其他最多4个位姿。
8. 将优化结果转换回世界坐标，并同时修正这些帧所拥有的世界点及其协方差。
9. 移除窗口外缓存和约束；窗口外轨迹保留，但不再参与本轮求解。

窗口启动阶段依次使用2、3、4、5帧；稳定阶段始终5帧。
间隔一帧匹配为网络直接推理，不是简单把两段光流相加。
跨帧匹配沿用既有batch=2 CUDA Graph，因此会同时重复计算一次当前立体匹配，只消费时间匹配半部分，并使用已有深度缓存。这是第一版为保持单个CUDA Graph内存池作出的性能取舍。

## 优化目标与求解

令 T_a、T_b 为两个相机到当前窗口锚定坐标系的位姿，p_a、p_b 为各自相机坐标中的固定三维观测：

$$
r_i=T_a p_{i,a}-T_b p_{i,b},
$$

$$
S_i=R_a\Sigma_{i,a}R_a^T+R_b\Sigma_{i,b}R_b^T.
$$

S保留全部非对角项。每个残差同时连接两个位姿，因此形成真实的多位姿耦合。用Cholesky分解进行白化：

$$
S_i=L_iL_i^T,\quad \bar r_i=L_i^{-1}r_i,\quad
\|\bar r_i\|^2=r_i^TS_i^{-1}r_i.
$$

目标为向量Huber损失之和，阈值为白化范数3：

$$
E=\sum_i \rho(\|\bar r_i\|),\qquad
\rho(n)=
\begin{cases}n^2&n\le3\\6n-9&n>3.\end{cases}
$$

左乘SE(3)扰动下，点位置 q_a=T_a p_a、q_b=T_b p_b 的Jacobian为：

$$
J_a=[I,-[q_a]_\times],\qquad
J_b=[-I,[q_b]_\times].
$$

锚定帧不产生变量列。将J用相同L白化，按Huber权重构造正规方程：

$$
(H+\lambda\operatorname{diag}H)\delta=-g.
$$

CPU float64直接求解最多24×24系统，不建立整个观测集合的巨大块对角权重矩阵。
每个外层步骤固定当前白化权重构造Jacobian，不对旋转相关协方差求导；试探位姿重新计算协方差和鲁棒目标，仅接受目标下降的步长。
因此这是显式记录目标的迭代重加权、阻尼ICP，并非包含协方差导数和logdet项的完整最大似然求解。

窗口输入、局部位姿复合和输出均进行四元数正规化，避免FP32历史回写误差被反复坐标变换放大。最早锚定帧的原始位姿行保持不变。

## 地图一致性和退化处理

历史位姿改变时，该帧作为源观测建立的世界点按

$$
\Delta T=T_{\rm new}T_{\rm old}^{-1},\qquad
P_{\rm new}=\Delta T P_{\rm old},\qquad
\Sigma_{\rm new}=R_{\Delta T}\Sigma_{\rm old}R_{\Delta T}^T
$$

更新。构造修正前先正规化旋转。其他帧拥有的点不修改。

- 无效深度、非有限观测和非正定协方差不能进入窗口约束。
- 每条边至少10个有效观测。
- 窗口不连通时保留旧两帧初值，记作not_refined，并记录原因。
- 如果跨帧约束成功恢复原本弱相邻匹配的非锚定帧，清除其插值标志，防止结束时覆盖恢复结果。
- 结束时的轨迹插值若改变位姿，也同步修正对应地图点。
- 当前仅支持AllKeyframe、mapping=false、CovarianceSanityFilter和同步ICP初值器。

## 命令

在当前开发分支运行完整MH01，结束后自动评估：

~~~bash
cd /home/zzh/MACVO
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python MACVO.py \
  --odom Config/Experiment/MACVO/MACVO_Fast_WindowICP.yaml \
  --data Config/Sequence/EuRoC_MH01_local.yaml \
  --seed 0 --resultRoot Results --timing
~~~

仍可使用两个旧配置运行两帧DISP/ICP；未指定Odometry.type时走旧MACVO路径。

三组重新运行并比较同一帧段：

~~~bash
.venv/bin/python Scripts/Experiment/CompareWindowICP.py \
  --sequences MH01 --seq_from 1500 --seq_to 1620 --seed 0
~~~

三组分别为 twoframe、window_adjacent、window_skip。
脚本串行运行，使用独立实验目录，不从旧目录选择“最新结果”，检查运行完整性和时间戳一致性，再输出CSV/JSON。

全部EuRoC完整三组比较：

~~~bash
.venv/bin/python Scripts/Experiment/CompareWindowICP.py \
  --sequences MH01 MH02 MH03 MH04 MH05 V101 V102 V103 V201 V202 V203 \
  --seed 0
~~~

仅运行MH02至V203的五帧“相邻＋间隔一帧”版本，并在每条序列结束后自动评估：

~~~bash
./Scripts/run_euroc_window_icp_mh02_v203.sh
~~~

可用环境变量指定随机种子和输出根目录：

~~~bash
SEED=0 RESULT_ROOT=/home/zzh/MACVO/Results/WindowICP_EuRoC \
./Scripts/run_euroc_window_icp_mh02_v203.sh
~~~

脚本默认顺序为MH02、MH03、MH04、MH05、V101、V102、V103、V201、V202、V203。
也可以在中断后显式传入尚未运行的序列，例如：

~~~bash
./Scripts/run_euroc_window_icp_mh02_v203.sh V103 V201 V202 V203
~~~

脚本使用文件锁防止同一输出根目录同时启动两个批次。MACVO的进度条、模块信息、Timer Report和异常会实时显示在终端，同时原样写入逐序列run.log和批次launcher日志；最终输出metrics.csv和metrics.json。设置DRY_RUN=1可只检查并打印命令。

仅跑所需窗口版本并评估（不重新跑基线）：

~~~bash
.venv/bin/python Scripts/Experiment/CompareWindowICP.py \
  --sequences MH01 --modes window_skip --seed 0
~~~

## 已完成验证

11项CPU测试：

~~~bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m unittest \
  Scripts.UnitTest.test_window_icp \
  Scripts.UnitTest.test_window_map \
  Scripts.UnitTest.test_window_provenance -v
~~~

覆盖：五帧位姿恢复、固定锚、双位姿Jacobian有限差分、完整协方差白化、窗口淘汰、断连拒绝、NaN拒绝、40帧FP32连续回写、地图点及协方差重锚定、恢复位姿不被插值覆盖、初始化失败状态记录。
仓库内置TartanAir双目10帧序列也完成实际GPU推理验证。

### MH01真实帧段

范围为序列索引1500至1619，共120帧，seed=0；不是完整MH01基准。
三组代码版本均为a0f49b4，使用同一时间戳序列。

| 模式 | ATE RMSE (m) | RTE RMSE (m/frame) | ROE RMSE (°/frame) | 平均Odom耗时 |
|---|---:|---:|---:|---:|
| 两帧ICP | 0.037271 | 0.001835 | 0.024682 | 595.8 ms |
| 五帧，仅相邻边 | 0.041320 | 0.001921 | 0.026319 | 657.7 ms |
| 五帧，相邻+间隔一帧 | 0.029949 | 0.001560 | 0.021205 | 1240.7 ms |

两种窗口模式均有119次有效优化，not_refined=0。
仅相邻模式最大缓存5帧/4边；跨帧模式最大5帧/7边。
跨帧模式窗口求解平均约63.7 ms，额外匹配平均约561.1 ms，显存日志约2.9GB（PyTorch保留显存口径）。

本段中跨帧版本更准确，但耗时约为两帧版本的2.08倍；仅扩大窗口并未改善精度。不能据此推断全EuRoC一定改善。
此外，两帧初值器和窗口求解器的鲁棒损失实现、阈值及同步方式不同；这个实验不是“仅改一个window_size、其他数值完全相同”的严格消融。

原始结果目录：
Results/WindowICP_comparison/20260912_214644_1452fb。
完整精度与耗时记录已版本化至 [CSV](validation/window_icp_mh01_1500_1620.csv)。
早期20260912_211312_d1c8a3运行用于定位四元数误差，含大量not_refined窗口，不能用作最终实验结论。

### batch=3融合前端验证

实现提交`ffd2fb4`将每个时间步的两次batch=2前端调用合并为一次固定batch=3 CUDA Graph：

1. 当前左图到当前右图，生成当前立体深度与协方差。
2. `t-1`左图到当前左图，生成相邻时间光流与协方差。
3. `t-2`左图到当前左图，生成跨帧时间光流与协方差。

首个时间步还没有`t-2`时，第三槽用相邻时间图像对填充但不消费输出，因此整条序列只捕获一个batch=3 CUDA Graph。`MACVO.run_pair`已拆分为前端推理和使用预计算结果构建因子两个阶段；普通MACVO仍使用原batch=2路径。WindowMACVO的关键点选择、协方差投影、直接`t-2 -> t`约束和窗口求解器均未改变。

在RTX 4060 Laptop GPU上使用MH01索引1500至1619、共120帧、seed=0验证。历史结果为提交`a0f49b4`的window_skip，新结果运行于`7e7f30a`（核心实现截至`fa4924c`）：

| 指标 | 两次batch=2 | 一次batch=3 | 变化 |
|---|---:|---:|---:|
| Frontend调用数 | 237 | 119 | -49.8% |
| 平均Odom耗时 | 1240.7 ms | 966.2 ms | -22.1% |
| 稳态Odom耗时（去掉前两项） | 1221.1 ms | 936.5 ms | -23.3% |
| 稳态Frontend耗时 | 553.9 ms/次 | 833.2 ms/次 | 单次处理三对输入 |
| ATE RMSE | 0.029949 m | 0.030170 m | +0.74% |
| RTE RMSE | 0.001560 | 0.001555 | -0.37% |
| ROE RMSE | 0.021205° | 0.021166° | -0.18% |
| RPE RMSE | 0.001646 | 0.001640 | -0.35% |
| 未优化窗口 | 0 | 0 | 不变 |

两次运行的120个时间戳完全一致。batch维度改变会引入轻微浮点顺序差异，最终四项误差变化均小于0.8%。新版本119个窗口全部refined，最大缓存仍为5帧/7边；CUDA保留显存日志约4.25GB，低于8GB设备容量。另用仓库内置10帧TartanAir双目序列验证得到9次前端调用、9个有效窗口和有限轨迹。

原定850 ms/帧目标没有达到。融合后的batch=3前端稳态仍需约833.2 ms，已经成为绝对瓶颈；若继续优化，应在不改变约束的前提下优先考虑图像特征缓存或更高效的三对联合前端，而不是优化约57 ms的CPU窗口求解。

## 每次运行的追溯信息

- config.yaml：本次Odometry配置。
- run_provenance.json：Git提交、分支、工作区状态、配置SHA256、种子、torch/PyPose版本、预期帧数、完成/失败状态。
- window_diagnostics.json：逐窗口位姿ID、锚、边数、匹配数、求解目标、步长接受次数、耗时及失败原因。
- poses.npy / ref_poses.npy / tensor_map.npz：兼容已有评估与地图导出。
- elapsed_time.json：前端、协方差及里程计计时。

跨帧约束保存在运行期有界侧缓存，最终tensor_map保留原邻接地图结构；该文件不包含全部历史skip原始观测。记录的边数/统计可用于验证执行路径，若需重建历史skip的完整观测，应重新推理。

## 版本操作

~~~bash
git log --oneline baseline/fast-icp-euroc-20260912..HEAD
git diff --stat baseline/fast-icp-euroc-20260912..window-icp-v0.1
~~~

回到两帧算法进行实验不必切换Git分支，使用MACVO_Fast_ICP_local.yaml即可。
若要基于旧代码继续开发，可从baseline标签新建分支；切换前保留自己的未提交修改，不使用强制重置。

## 尚未实现

没有边缘化先验、回环检测、共享地图点BA或BAE后端；没有估计重复图像观测带来的跨因子相关性。
窗口外约束丢弃，窗口外位姿不再更新；最早窗口位姿固定不等于全局漂移已消除。
完整EuRoC评测仍需运行上面的批量命令。
