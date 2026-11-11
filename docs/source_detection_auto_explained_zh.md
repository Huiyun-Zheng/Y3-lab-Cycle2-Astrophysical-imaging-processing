# 自动源检测代码详解

阅读对象：`source_detection_auto.py` 和 `test_source_detection_auto.py`。两个文件现在都在每个实际代码行旁添加了英文注释；原有 docstring、解释性注释和空行保留，多行表达式的结束括号也标注了结束含义。本次只增加注释，不修改运算、参数或控制流程。

下面的行号对应这次注释后的版本。逐行的“这一句做什么”直接看 Python 文件；本文解释“为什么要这样做、前后怎样连接、结果代表什么”。

## 1. 先理解整体方法

这版属于 **基于阈值连通区域与峰值分离的候选源检测**。不是在全图反复寻找最亮像素、用固定半径圆遮掉后再找下一个。它先形成所有源的 footprint，再分离相邻峰，最后按原图峰值亮度排序测量。

```text
读取 FITS 原始图像
    ↓
识别非有限值和边缘恒值填充区
    ↓
从饱和种子生长不规则污染 mask
    ↓
第一遍局部背景/噪声估计
    ↓
暂时屏蔽明显源，第二遍估计背景/噪声
    ↓
背景扣除 → 轻微平滑 → 估计平滑背景噪声
    ↓
2σ 连通 footprint + 5σ 局部峰
    ↓
检查峰与低谷的差异，去掉浅起伏
    ↓
从保留的峰出发，按亮度地形分配 footprint 像素
    ↓
过滤面积不足的 segment
    ↓
按原图峰值从亮到暗测量位置、形状、flux、质量标记
    ↓
保存 catalogue、mask、诊断图和运行摘要
```

“全自动”的意思是：输入图像及已有默认参数后，无需手动圈区域或逐个确认源。它不等于自动证明每个候选是真实星系，也不保证识别所有污染。

## 2. 必须分清的数组和坐标

对本实验图像，原图数组形状是 `(4611, 2570)`：4611 行、2570 列。

- `image[y, x]`：先行后列，取一个像素。
- 图上标记位置写 `(x, y)`：先横轴后纵轴。
- `np.nonzero(mask)`、`np.mgrid` 的二维数组坐标按 `(row, column)`，也就是 `(y, x)`。
- `bounds = (slice(y0, y1), slice(x0, x1))`：只是快速定位局部区域的矩形，不代表该矩形全部属于源。
- `region = labels[bounds] == source_id`：在矩形内再选出真正属于这个源的不规则像素。
- `True` 对于坏像素 mask 表示“不使用”；对于 `region` 或 `aperture` 表示“属于当前选区”。必须结合变量名理解。

| 变量 | 类型/形状 | 含义 |
|---|---|---|
| `image` | 原图大小，float32 | 原始 ADU 数据转浮点，仍包含背景 |
| `missing` | 原图大小，bool | NaN/Inf 或疑似边缘恒值填充 |
| `saturated` | 原图大小，bool | 达到采用的饱和阈值的种子 |
| `artefact` | 原图大小，bool | 与饱和种子相连并扩大的污染区 |
| `boundary` | 原图大小，bool | missing 的安全余量和最外侧边缘 |
| `excluded` | 原图大小，bool | `artefact | boundary` |
| `background`、`noise` | 原图大小，float32 | 原始图的局部背景和背景散布 |
| `residual` | 原图大小，float32 | `safe_image - background` |
| `smoothed` | 原图大小，浮点 | 平滑后的背景扣除信号 |
| `detection_snr` | 原图大小，浮点 | 平滑信号相对于局部平滑背景散布的标准化值 |
| `detection_excluded` | 原图大小，bool | 再扩大后的检测禁用区 |
| `parents` | 原图大小，整数 | 2σ 连通区域编号，0 是区外 |
| `segments` | 原图大小，int32 | 最终分出的候选源编号，0 是未分配 |
| `metadata` | dict | `segment_id → (parent_id, blended)` |
| `processed_mask` | 原图大小，bool | 最后确实记录了测量的 segment 像素 |

**mask、segment 和 aperture 是三件不同的事。** mask 决定哪里不能用；segment 决定哪些像素归属于哪个候选；aperture 是用于求光度的圆形选区。这里的 `processed_mask` 是结果记录，不是驱动检测循环的排除圆。

局部索引例子：如果 `bounds` 从全图 `(y0, x0) = (100, 200)` 开始，`region[3, 5]` 对应全图 `image[103, 205]`。这就是测量函数中 `local_y + bounds[0].start` 和 `local_x + bounds[1].start` 的作用。

## 3. 导入与 Settings（第 17–73 行）

`dataclass` 自动生成 `Settings(...)` 的初始化方法。冒号后面的 `int`、`float`、`float | None` 是类型提示，不会自动检查所有输入类型；`__post_init__` 只执行代码中明确写出的数值检查。

`asdict(settings)` 把参数变成字典，供 JSON 保存。`Path` 处理文件路径；`argparse` 接收终端参数；`heapq` 是 Python 标准库中的优先队列，后面的分水岭不需要安装新包。

| 参数 | 默认值 | 具体作用 |
|---|---:|---|
| `background_box` | 128 | 每块最多 128 × 128 像素估计背景 |
| `saturation_override` | None | 不指定时读取 `SATURATE` header |
| `artefact_sigma` | 3 | 原始亮像素的污染生长门槛 |
| `halo_sigma` | 1.5 | 平滑光晕的生长门槛；这里仍乘原始全局 scatter |
| `artefact_padding` | 4 | 污染区扩大 4 次 |
| `boundary_padding` | 4 | missing 扩大 4 次，排除最外 4 行/列 |
| `smooth_sigma` | 1.0 pixel | 检测图 Gaussian kernel 的标准差 |
| `detection_sigma` | 5 | 候选 seed 的最低平滑 S/N |
| `growth_sigma` | 2 | 连通 footprint 的最低平滑 S/N |
| `min_pixels` | 8 | parent 和分离后的 child 至少包含 8 像素 |
| `peak_separation` | 8 | 实际构造 9 × 9 maximum-filter 窗口 |
| `peak_prominence_sigma` | 3 | 次峰与连接低谷至少相差 3 个平滑 S/N 单位 |
| `peak_prominence_fraction` | 0.05 | 同时至少相差次峰自身高度的 5% |
| `min_aperture_radius` | 5 pixels | 光度孔径最小半径 |
| `max_aperture_radius` | 35 pixels | 光度孔径最大半径 |

`peak_separation = 8` 不是严格的“源中心至少相距 8 pixels”：窗口是以当前像素为中心、横纵各向外 4 pixels 的正方形。

代码中还有几项固定选择：污染追踪的平滑宽度 1.2 pixels；初步源 mask 使用原图局部 3σ 并扩大 3 次；检测 mask 额外扩大 4 次；孔径取 `2.5 × major_sigma`；elongation > 6 触发细长标记。这些也是方法的一部分。

`CONNECTED = np.ones((3, 3), dtype=bool)` 让显式使用它的步骤考虑对角连接。未传 `structure` 的二值 dilation/propagation 使用 SciPy 默认的二维十字邻域，所以“扩大 4 次”不是精确的半径 4 圆。

## 4. clipped_stats（第 76–88 行）

目的：从可能含亮源的样本中得到较稳健的背景值和散布。

1. `values[np.isfinite(values)]` 去掉 NaN 和 Inf。
2. 少于 20 个值就报错；样本太少不继续估计。
3. `sigma_clipped_stats(..., sigma=3, maxiters=8)` 在内部迭代剔除离群值，返回 clipped mean、median、standard deviation。
4. `_` 表示不使用第一个返回值，也就是 mean；实际保留 median 与 scatter。
5. scatter 必须有限且大于零，否则后续阈值和除法没有意义。
6. 返回普通浮点数 `(median, scatter)`。

这是自动版的统计方法，与原版脚本先用 MAD 迭代五次的实现不同。这里的 scatter 是背景像素分布的宽度，不是背景均值的误差，也没有包含每个亮源自身的完整 Poisson 测光误差。

## 5. find_missing_data（第 91–114 行）

这一步只处理坏数值与疑似填充区，不负责识别 bleeding。饱和像素一般仍是有限数，因此不会仅靠 `~np.isfinite(image)` 被识别。

先取四条图像边界的数值，寻找最常见的精确取值。如果这个值不足边界采样的 20%，就不猜测 padding；如果达到门槛，则只从边界上的同值像素出发，沿相同数值的像素扩展。

`binary_propagation(seeds, mask=possible_padding)` 可以理解为：从种子出发，在 `possible_padding=True` 的通道中蔓延，直到没有可扩展像素。这里没有传 `CONNECTED`，所以用默认四邻接。

最终返回 `invalid | padding`。内部孤立的同值像素不会因为数值恰好一样而全部被删除，但如果与边缘填充相连，仍可能被算入填充区。因此这是启发式识别，不是 FITS 官方的数据质量标记。

## 6. background_maps（第 117–170 行）

目标是得到两个与原图一样大的数组 `B(y, x)` 和 `σ_raw(y, x)`，允许背景随位置变化。

**分块统计。** `(height + box_size - 1) // box_size` 是向上取整的整数除法，保证不满 128 pixels 的最后一块也被保留。每块只使用有限且没有被排除的像素。如果 usable 数量小于 `max(40, 0.25 × tile.size)`，整块暂留 NaN。即至少要有 40 个样本，也至少覆盖该块的四分之一。

**补齐无法测量的块。** `distance_transform_edt(..., return_indices=True)` 返回最近有效块的索引，用其统计值补齐缺失块。如果所有块都无效，直接报错。补齐值是从邻近块借来的，不是该处独立测量。

**平滑粗网格。** 对背景与 scatter 的粗网格分别做 3 × 3 median filter，抑制孤立异常块。这个滤波不是对原图做 3 × 3 中值处理。

**插值回原图。** 对每一块计算真实中心位置；最后一块可能较短，不能简单假设它的中心仍在整齐的 128 像素间隔上。`np.interp` 把原图坐标转换成粗网格的分数坐标，`meshgrid(indexing='ij')` 产生行列坐标图，再用 `map_coordinates(order=1)` 线性插值。

输出都是原图大小的 float32 数组。背景随空间平缓变化，但小于网格尺度的复杂变化可能描述不好；宽广天体也可能有一部分被当作背景。

## 7. identify_artefacts（第 173–199 行）

方法是 **从疑似饱和核心沿相连亮结构生长**，而不是预先指定长方形坐标。

首先每隔三个像素抽样一次，估计全局背景 `B_global` 和原始背景散布 `σ_global`。missing 暂时填为 `B_global`，避免 NaN 通过滤波污染邻近结果。

饱和种子为：

```python
saturated = (safe_image >= saturation) & ~missing
```

允许继续生长的通道由三部分组成：

- 原始像素 `I > B_global + 3 × σ_global`；
- 用 sigma=1.2 pixels 平滑后的值 `I_soft > B_global + 1.5 × σ_global`；
- 饱和种子本身。

第二个门槛的 `σ_global` 仍来自原始图，因此这里只是光晕追踪规则，不能把它解释成平滑噪声下正式的 1.5σ 显著性。

`binary_closing` 先膨胀再腐蚀，补小间隙；重新加入 `saturated` 保证种子不消失。`binary_propagation` 用八邻接沿允许通道增长，能追踪弯曲、斜向和横竖拖尾。随后填封闭孔洞，再加少量余量。

返回 `(saturated, artefact, level, scatter)`。注意：普通有限像素即使没有达到饱和，只要沿亮通道连接到种子，也会被标为污染。反过来，没有饱和种子的污染可能漏掉；与污染相连的真实近邻也可能被一起屏蔽。这里不进行恒星/星系分类。

## 8. run_detection 中的背景扣除与检测图（第 519–568 行）

`run_detection` 才是真正串联各函数的主流程；前面的 `def` 在文件加载时只是定义函数。

读取 `hdul[0]` 的 header 和 data，将 data 转为 float32。这允许背景扣除产生负数，避免对 unsigned integer 直接相减的风险。输入只支持二维 primary image。

默认采用 header 中的 `SATURATE`；若没有该项且未提供 override，则报错，不自动猜阈值。实验图目前采用的 36000 是继承的元数据，不是本代码重新标定的饱和点。

建立 `excluded = artefact | boundary` 后，先估计一次局部背景，再将明显高于原图局部 3σ 的区域扩大、暂时排除，估计第二次背景。这是为了减少亮源对背景估计的影响。暂时的 `initial_sources` 仅用于背景估计，**并没有从最终找源范围中永久移除这些源**。

令 `R = safe_image - background`。在临时滤波输入中将 excluded 位置设为零，再做一像素标准差 Gaussian smoothing。原始 `image` 没有被改成零。为减少 mask 边界的滤波影响，检测禁用区再扩大四次。

对平滑图重新测量背景中值 `B_s` 和散布 `σ_s`，构造：

```text
D(y, x) = [GaussianSmooth(R)(y, x) - B_s(y, x)] / σ_s(y, x)
```

代码将它称为 `detection_snr`。它是局部平滑背景单位下的检测统计量，不是源积分 flux 的 S/N。平滑像素间存在相关性，不能把每一个像素简单当成独立的 Gaussian 试验。

## 9. separate_sources（第 274–328 行）

**两级阈值。** `footprint = (D >= 2) & ~excluded` 控制候选范围；`D >= 5` 控制峰是否足够显著。以较低阈值包住源外围、较高阈值确认其核心，避免只保留最亮几个像素。

`ndi.label` 给每个八邻接 footprint 编号，得到 `parents`。两个天体的外围只要在 2σ 以上连起来，就可能属于同一个 parent，因此 parent 不等于单个物理源。

**寻找种子。** 将被排除像素在搜索数组中替换为 `-inf`，避免掩膜中的亮像素压制外面有效的局部峰。maximum filter 找 9 × 9 范围的最大值；只有等于该最大值且至少 5σ 的像素才能成为峰候选。多个相邻等高峰像素会通过一次 label 聚成一个平台，并选一个代表坐标。

每个 parent 先满足至少 8 pixels，再调用 `select_significant_peaks`。只有一个保留峰时，整个 parent 归它；多个峰时，调用 `grow_peak_regions` 分配像素。每个分离后的 child 再检查至少 8 pixels，太小则丢弃，不在这一轮重新分配其像素。

`segments[bounds]` 返回全图标签数组的局部 view；给 `local_labels[child]` 赋值会更新全图 labels，而不是只改一个无关副本。metadata 记录这个 segment 来自哪个 parent，以及该 parent 是否有多个保留种子。

这里的 `min_pixels=8` 是分割区域面积条件，与原版“5 × 5 邻域中至少五个像素超过 threshold”不是同一规则。

## 10. select_significant_peaks（第 202–231 行）

为什么不能直接保留所有局部峰？一个真实源内部也可能有噪声起伏或复杂结构，直接分峰会把一个源拆成很多个。

对当前峰高度 `P`，要求的下降幅度为：

```text
required_drop = max(3, 0.05 × P)
```

只在 `D > P - required_drop` 的高地里，从当前峰开始找相连像素。如果不经过足够低的谷就能走到一个已接受的更强峰，当前峰只是浅起伏，应合并；如果高地与更强峰断开，则保留。

例子：较弱峰为 19，连接低谷为 18，而要求下降至少 3，那么只下降 1 就能连过去，不应拆开。这正是 test 文件中的一个场景。

旧版曾用“至少达到整个 parent 最亮峰的 5%”筛选，会吞掉亮星旁真实的弱伴源。现在 5% 以次峰自身为基准，但这个规则仍属于启发式 deblending，并不是严格的独立源统计检验。

返回的是局部 `(y, x)` 坐标列表；还没有分配全部像素。

## 11. grow_peak_regions（第 234–271 行）

这是分水岭式的区域分配。可以把亮源想成山峰，把 `D` 变为 `-D` 后，峰变成低洼盆地。每个种子有自己的标签，从这些盆地出发逐步“涨水”，相遇处形成分界。

`heapq` 每次弹出最小的元组。队列元素是：

```text
(flood_level, arrival_order, y, x)
```

- `flood_level` 决定谁先处理；初始值是 `-D`，所以较亮种子先处理。
- `arrival_order` 递增，用于优先级相同情况下的确定性排序。
- `y, x` 是要继续扩展的当前像素。

先给所有种子写好自己的标签。每次处理八个邻居，跳过越界、footprint 外和已经分配的像素。对未分配的邻居：

```python
next_level = max(level, -surface[yy, xx])
```

这里保留沿途需要跨越的最高障碍。例如当前路径水位已到 -10，即使下一处地面是 -20，也不能让这条路径的水位重新降回 -20。因此它不是只按最近距离分配，也不是拿一个固定圆遮罩。

邻居继承当前像素的 label，然后进队列。所有生长严格在 parent 的真实 footprint 内完成，外面没有会“抢走”源像素的背景种子。返回 `children`，它只在本 parent 的局部矩形中编号；`separate_sources` 再把它换成全图唯一的 segment ID。

## 12. measure_sources（第 331–432 行）

### 12.1 排序与位置

从每个 segment 的原图像素中取最大值，按它从大到小排序。`id` 是测量记录的亮度排序编号，`segment_id` 是分割图中的编号，两者不必相等。

每个源的原始背景扣除值为 `signal`，位置权重取 `w_i = max(signal_i, 0)`。因此 centroid 为：

```text
x_c = Σ(w_i x_i) / Σw_i
 y_c = Σ(w_i y_i) / Σw_i
```

`x, y` 是加权质心，可能是小数；`peak_x, peak_y` 是原图最亮像素的整数坐标。两者可以不同。图上的检测圆圈画在质心。

### 12.2 形状与孔径

以质心为原点，计算二维二阶矩：

```text
C = [[<dx²>_w, <dx dy>_w],
     [<dx dy>_w, <dy²>_w]]
```

矩阵特征值开平方得到 minor、major：它们是 footprint 内信号的 RMS 尺度，不是拟合得到的 Gaussian sigma，也不是天体的硬边界或半径。

`elongation = major / max(minor, 0.25)`，0.25 是避免分母过小的数值保护。孔径半径采用：

```text
r_requested = max(5, 2.5 × major)
r_used = min(r_requested, 35)
```

`aperture` 是以质心为圆心、像素中心距离不大于 r 的 Boolean 数组，不计算边缘像素的部分面积权重。

### 12.3 两种 flux

`segment_flux_adu = Σ_segment (I_i - B_i)`：只求被分割算法分给该源的像素。阈值以下的外围被漏掉。

`aperture_flux_adu = Σ_circle (I_i - B_i)`：在自适应圆孔径里求和，可包括 footprint 外的像素。仍可能截断外围光，也没有做 aperture correction。

求 flux 使用的是原始未平滑图的背景扣除信号。仅在位置/形状权重中将负值裁为零，flux 求和本身保留负值。这里没有根据 GAIN 推导电子数、源 Poisson 噪声或 magnitude error。

孔径碰到 bad mask、图像外或另一 segment 时，`aperture_flux` 直接写 NaN，而不是把坏像素删掉后对剩下的部分求和并冒充完整孔径通量。

### 12.4 Flags

| Flag | 触发原因 | 应怎样理解 |
|---|---|---|
| `near_artefact` | segment 接触扩大后的污染近邻区 | 附近污染可能影响测量 |
| `near_boundary` | segment 接触扩大后的边缘近邻区 | 边缘附近测量可能不稳 |
| `blended` | parent 保留了多个 seed | 做过分峰，邻源可能影响 flux |
| `aperture_contaminated` | 孔径碰到 bad mask、图外或另一 segment | 孔径 flux 写 NaN |
| `aperture_capped` | 所需半径大于 35 | 使用了上限，可能遗漏外围 |
| `very_elongated` | elongation > 6 | 细长候选，需要注意；不证明它是 bleeding |
| `usable` | 所有上述 flags 都未触发 | 保守的测光质量筛选，不代表已确认是星系 |

`blended` 在 child 面积筛选之前根据 seed 数量记录，因此即使另一个 child 最后被面积条件删掉，也可能保留 blended 标记。blended 本身不强制把 flux 改为 NaN；真正令 flux 为 NaN 的是 aperture contamination 条件。

`processed_mask` 只标出已记录 segment 的像素，不标整个测光孔径，也没有原版的 `mask_radius=12` 排除圆。

## 13. save_diagnostics（第 435–516 行）

这一部分只做可视化，不改变检测结果。

- `Agg` backend 把图存到文件，不等待 GUI 窗口关闭。
- 灰度底图用 `(image - background) / noise`，是原始图标准化后显示，不是平滑检测图。
- 金色叠加污染，蓝色叠加边缘/padding。
- 青色圈是无 flags 的候选，紫色圈是有 flags 的候选。紫色仍在完整 catalogue 中。
- 概览图使用 `Circle` 和 `PatchCollection`，以每条记录的 `aperture_radius_pix` 为半径；**这些圈现在是实际使用的圆形 photometric aperture，不是 segment 边界。** 旧版的固定大小 `scatter` 标记已替换。
- 按污染组面积选最大的六组做 close-up；再自动选最大污染组的下端、最宽处和上端做细节图。
- 图中的矩形视野只是裁切显示窗口，不表示用矩形遮掉了源。

函数参数 `saturated` 目前保留在接口中，但函数体没有使用它；它在其它步骤和 FITS 产品中仍有用途。`separate_sources` 的 `smoothed` 参数目前只提供输出数组形状，真正分峰与生长用 `detection_snr`。这些都按现有代码注明，没有在本次注释任务中重构。

## 14. 输出与命令行入口（第 570–659 行）

负图控制把 `smoothed` 与 `detection_snr` 取负后，用相同参数检测。对称背景噪声可能在两侧产生峰，因此它可帮助发现噪声或背景问题。但正向污染通常不对称，负图数量不是精确 false-positive count，也不能直接给真实图的误检率。

输出包括完整 ECSV/CSV、仅 usable 的 CSV、不规则污染区 audit CSV，以及包含 masks、segments、background、noise 的 FITS。`artefact_regions.csv` 中的 bounds 只是帮助定位组；真正 mask 存在 FITS 数组中。

FITS 中 `BADMASK` 是原始 `excluded`，`DETMASK` 是额外扩大的检测禁用区，两者不同。产品保留原图像素对齐，但没有复制未经核实的天球 WCS。这个生成 FITS 约 213 MB，目前已加入 `.gitignore`，本地运行仍会正常生成它。

`run_summary.json` 记录参数、计数、限制。最近一次保存的优化结果为 7437 个候选、4692 个无 flags 候选，负图控制 201 个；这是已有运行结果，本次逐行注释本身没有重新生成 catalogue。

`if __name__ == '__main__':` 区分直接运行与导入：

```bash
python source_detection_auto.py
python source_detection_auto.py --fits /path/image.fits --output /path/results
```

直接执行时读参数并运行；被测试文件 import 时，只加载定义，不自动处理整张 mosaic。`Path(__file__).resolve().parent` 以脚本所在位置为基准，而不是终端当前文件夹。

## 15. test_source_detection_auto.py 逐段解释

这是**回归测试**：构造已知答案的小场景，确认以前修复的问题没有再次出现。它不替你检测真实图，也不需要人工圈源。

### 第 7–25 行：共同准备

`unittest.TestCase` 提供 assertions。每个 `test_...` 方法开始前都会重新调用 `setUp`，创建默认 Settings、100 × 100 坐标图和全 False 排除 mask。

`self` 是当前测试实例；不同方法通过它使用各自的 settings、xx、yy、excluded。每个测试都重建这些变量，避免上一个测试修改 mask 影响下一个测试。

`gaussian(x, y, height, width)` 生成：

```text
G(x_i, y_i) = height × exp(-[(x_i - x)² + (y_i - y)²] / [2 × width²])
```

height 是峰值而不是总 flux，width 是标准差而不是半径。这些多为直接构造的检测 surface；测试调用把 surface 同时作为 signal 和 S/N 输入，**没有模拟完整背景估计或完整仪器观测**。

### 第 27–33 行：亮源旁 1% 高度的伴源

两个 Gaussian 的高度为 10000 和 100，中心分别为 `(35, 50)` 与 `(61, 50)`。检查两中心都有正标签，而且标签不同。只检测到一个把两者包住的大源不能通过这个测试。

它针对旧版按最强峰 5% 删除弱伴源的问题，不意味着任意分离度和背景下所有 1% 伴源都能被发现。

### 第 35–44 行：footprint 和种子保留

直接给 `grow_peak_regions` 两个测试 seed，检查所有 footprint 像素都被分配、外面没有被分配、每个 seed 保持自己的编号，而且两组面积都达到 minimum。

这里手动指定 seed 坐标是为了隔离测试“区域分配”这个函数；生产检测的 seed 仍由算法自动寻找。

### 第 46–52 行：浅起伏合并

构造 `[10, 20, 18, 19, 10]` 并重复三行。第二个峰 19 相对低谷 18 只高 1，小于所需 3，因此只应保留峰 20。它检查算法不会无条件地把所有小峰当成独立源。

### 第 54–60 行：掩膜亮像素不压制有效源

在 `(50, 50)` 放一个 Gaussian，再在同一 maximum-filter 窗口内放数值 10000 的像素并把它排除。检查真实中心仍被检测，坏像素仍为 label 0。

这验证了先把 excluded 位置设为 `-inf` 再求 maximum filter 的必要性。

### 第 62–66 行：平顶峰只产生一个 seed

在全零数组中设置 10 × 10 的等高平台。平台包含很多相等的局部最大值，但应被合并成一组，最终 `labels.max() == 1`。

这里用 max 计数依赖生产代码“只对保留源连续编号”的约定。

### 第 68–75 行：空图与单像素尖峰

依次测试中央像素为 0 和 100 的数组。空图没有源；单像素虽然超过 5σ，但不足 8 pixels，所以也没有源。同时要求返回 metadata 为空。

这个测试从直接给定 surface 开始，不是对所有形状的 cosmic ray 做真实性能测试。

### 第 77–93 行：多个有限值拖尾

固定随机种子 16，生成均值 100、scatter 2 的背景。画一条弯曲拖尾和另一条竖直拖尾：拖尾值为 180，低于测试饱和阈值 300，但分别放置一个值为 400 的饱和种子。最上三行写成恒值 100，模拟 padding。

检查两条拖尾全部进入 mask，种子没有丢失，padding 被识别，另外选定的干净位置没有被误遮掉。这验证的是**少数已知几何场景**，不是所有 bleeding 形状的精准识别保证。

### 第 96–97 行：运行测试

```bash
python -m unittest -v test_source_detection_auto
```

也可以直接 `python test_source_detection_auto.py`。命名以 `test_` 开始的方法会被执行。

- `assertGreater(a, 0)`：必须有正标签。
- `assertEqual(a, b)`：必须相等。
- `assertNotEqual(a, b)`：必须不同。
- `assertTrue(...)` / `assertFalse(...)`：条件必须为真/假。
- `assert_array_equal(...)`：整个数组逐元素完全相同。

若 assertion 不满足，测试失败并指出位置。出现 `Ran 7 tests ... OK` 表示这七个受控场景通过，不代表全图检测完整性、分类准确率或测光准确度都已验证。

## 16. 建议的阅读顺序

先读 `run_detection` 看数据如何流动，再读 `Settings` 看参数含义；接着依次读 missing → artefact → background → separate_sources → select_significant_peaks → grow_peak_regions → measure_sources。最后看绘图和测试。

最值得先理解的三点是：

1. 高阈值确认峰、低阈值包住外围：5σ seed 与 2σ footprint。
2. 两个峰能否拆开，看连接低谷深度，再由受 mask 限制的生长分配像素。
3. 检测、分割、测光和质量筛选是不同步骤。检测成功不等于 flux 无污染，也不等于候选已被确认是 galaxy。

算法依然可能漏掉低表面亮度源、非常近的双源和落在污染 mask 内的真实邻源，也可能把复杂天体拆成多个峰。这个版本适合解释与实验探索；若要定量报告 completeness 或 purity，还需要匹配真值或进行带噪声的注入回收实验。
