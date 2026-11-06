# Automatic source detection

独立脚本：`../../source_detection_auto.py`。原有 `source detection.py` 不参与运行，也没有被修改。

## 运行

在 `astro-image` 环境中，从实验文件夹执行：

```bash
python source_detection_auto.py
```

脚本默认定位与自己同目录下的 `Astro/Fits_Data/mosaic.fits`，因此不依赖终端的当前工作目录。
也可以指定路径：

```bash
python source_detection_auto.py --fits /absolute/path/image.fits --output /absolute/path/results
```

全程没有 `input()`，绘图保存到文件，不会等待关闭图窗。再次运行会更新该输出目录内的同名结果。
需要 NumPy、SciPy、Astropy、Matplotlib；本次运行使用了现有的 `astro-image` 环境，没有安装新包。

## 代码阅读顺序

1. `Settings`：所有阈值、尺寸及安全余量。
2. `find_missing_data`：排除非有限值，以及与图像边缘连通的恒值填充区。本图的填充值是 3421，但没有硬编码这个数。
3. `identify_artefacts`：从疑似饱和像素出发，沿显著亮于背景的相连区域生长。轻微平滑补充较弱的光晕信号，最后扩大少量像素作为余量。弯曲、横向、竖直、斜向结构均可跟踪，不使用人工指定矩形或恒星坐标。
4. `background_maps`：分块估计背景和噪声，屏蔽初步源区域后重新估计，减少源光污染。
5. `separate_sources`：在轻微平滑后的图上寻找相连的源区域；`select_significant_peaks` 检查峰与连接低谷的高度差，`grow_peak_regions` 从峰出发、仅在源区域内进行优先队列分水岭生长。所有分开的组保留 blended 标记。
6. `measure_sources`：按原始峰值从亮到暗记录位置、范围和信号，不重复记录同一个区域。
7. `run_detection`：串联以上步骤，保存表格、mask、背景图及诊断图。

这里不是原始教学版的固定圆逐像素搜索：源范围先由连通区域决定，最终 catalogue 按峰值亮度排序。
`PROCESSED` 保存已记录的源像素，不会将原始图像改成零。

## 输出

| 文件 | 内容 |
|---|---|
| `sources.ecsv` / `sources.csv` | 全部候选源和质量标记；ECSV 保留类型和说明 |
| `usable_sources.csv` | 没有质量标记的保守子集，仍可能包含普通恒星 |
| `artefact_regions.csv` | 自动屏蔽的污染区域及其像素数；表内矩形范围只是定位信息，实际 mask 沿形状生成 |
| `detection_products.fits` | `BADMASK`、`ARTEFACT`、`SATURATED`、`BOUNDARY`、`DETMASK`、`SOURCES`、`PROCESSED`、`BACKGROUND`、`RMS` |
| `detection_overview.png` | 全图的污染 mask 和检测结果 |
| `artefact_closeups.png` | 按污染面积自动选取的局部图，包含较小的亮星和短 bleeding |
| `bleeding_detail.png` | 主污染区域的两端和最宽部分，便于检查下方弯曲伪影 |
| `run_summary.json` | 参数、计数及限制说明 |
| `detection_comparison.png` | 本次修改前后，用户截图对应区域的检测对比 |
| `refinement_comparison.json` | 修改前后计数和 A/B/C 示例的像素坐标，仅供验证 |
| `before_refinement/` | 修改前的自动脚本、目录和概览图，供追溯 |

所有像素坐标采用 Python 的 0 起始 `(x, y)`。二维数组仍按 `[y, x]` 访问。
输出 FITS 与原图像素对齐，但不复制未经验证的天空坐标 WCS。

## Flux 与质量标记

- `segment_flux_adu`：源分割区域内的背景扣除信号，遗漏阈值以下的外围光。
- `aperture_flux_adu`：根据源尺寸选择圆孔径后的背景扣除信号，也不是经孔径修正的总 flux。
- `aperture_contaminated`：孔径碰到污染、边缘或另一检测源；此时 aperture flux 为 NaN，不会对残缺区域求和当作有效测量。
- `near_artefact` / `near_boundary`：源区域靠近污染或边缘。
- `blended`：一个相连区域分出了多个峰，需要后续关注；不自动删除 catalogue 记录。
- `aperture_capped`：建议半径超过了上限，测量可能截断明显外围光。
- `very_elongated`：极细长的源被保守标记；这不等于已证实为 bleeding，也可能是真实细长星系。
- `usable`：以上质量标记全部为空。它仅是保守筛选，不是星系身份或完整性保证。

## 参数与限制

- 默认使用原 header 的 `SATURATE=36000` 作为疑似饱和起点；这个 mosaic 的实际适用性仍需核实。可在 `Settings.saturation_override` 覆盖。
- `halo_sigma` 降低会扩大光晕排除范围，也可能吞掉更多近邻源；`artefact_padding` 是轮廓之外的安全余量。
- 未与饱和种子相连的异常，或比追踪阈值更暗的外围，可能漏掉。相连的邻近真实源也可能一起被屏蔽。这是实测轮廓的近似，不是精确污染边界。
- `detection_sigma` 针对平滑图的实测背景波动，不是原始图单像素的 5 sigma，也不是源总 flux 的 S/N。
- `growth_sigma` 控制源区域向外扩张到哪里；`peak_separation` 限制很近的双源能否拆开。复杂星系也可能被分成多个峰。
- `peak_separation = 8` 对应当前实现的 9 × 9 局部最大值窗口，并非严格的 8 像素欧氏距离。
- `peak_prominence_sigma = 3` 与 `peak_prominence_fraction = 0.05`：一个次峰与通向更强峰的低谷之间，高度差至少达到 3 倍平滑背景噪声，且达到次峰自身高度的 5%。它无需达到整个相连区域最强峰的 5%。这些是分峰规则，不是额外的正式统计显著性检验。
- 负图控制将背景扣除信号取反再检测，帮助暴露噪声峰及背景问题；它不是校准后的误检率。
- 没有完成恒星/星系分类、注入回收完整性实验、精确 PSF 拟合、总 flux 修正或测光误差标定。

## 验证

已在真实 `mosaic.fits` 上端到端运行，并检查：全部疑似饱和像素进入坏区域；检测区域不进入检测 mask；catalogue ID 唯一且按峰值降序；受污染孔径的 flux 为 NaN。
另用合成图检查曲线拖尾、另一颗带拖尾亮星、亮度差 100 倍的相邻源、分割像素与种子保留、峰顶小起伏合并、掩膜附近的源、相同亮度的平台、无源图及单像素尖峰。合成检查不是对真实图检测完整性的证明。

可在实验文件夹重新运行这 7 个测试：

```bash
python -m unittest -v test_source_detection_auto
```

本次真实图的 13 项检查记录在 `validation.json`。

## 2026-11-05：修复截图区域的漏检

原分割代码在源区域外设置背景种子，某些亮源的大部分像素最终被赋给背景，余下不足 `min_pixels` 而被丢弃。
另外，按整个相连区域最亮峰的 5% 筛选，会合并亮源旁边清晰可见的较暗伴源。
现在生长严格限制在源区域内，种子始终归其自身，分峰改为检查局部低谷深度。被屏蔽的亮像素也不再参与局部最大值比较。
使用 Python 自带 `heapq` 实现优先队列，没有新增第三方包。

| 指标 | 修改前 | 修改后 |
|---|---:|---:|
| 全图候选源 | 7017 | 7437 |
| 无质量标记的候选源 | 4641 | 4692 |
| 截图对应区域的候选源 | 41 | 44 |
| 负图控制候选源 | 170 | 201 |

检测阈值仍为平滑图的 5 sigma；污染 mask 未改变。图中 A 先前被丢弃，B、C 先前被合并；本次已分别恢复。
图中的示例坐标仅用于验证与作图，没有写进检测规则。改变分割也会合并一些原来的弱起伏，因此净增 420 个候选不等于已证实新增 420 个真实星系。
负图控制计数也有所上升；它仅提供诊断，不能直接当作真实图的误检率。复杂重叠、低表面亮度目标和被污染 mask 覆盖的邻源仍可能漏检。

参考：
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_propagation.html
- https://scikit-image.org/docs/0.21.x/api/skimage.segmentation.html#skimage.segmentation.watershed （优先队列分水岭方法说明；本脚本不依赖 scikit-image）
