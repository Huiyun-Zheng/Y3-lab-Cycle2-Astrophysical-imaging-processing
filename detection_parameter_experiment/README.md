# Detection parameter experiment — 2026-11-16

## 结果

复制现有 source_detection_final.py，在副本中只修改 Settings 的默认参数及
命令行输入/输出默认路径。所有检测、分割、测量函数与原文件的 AST 完全相同。
原有 35 个脚本/输出文件（包括此前 conservative_star_analysis 内的文件）
均通过 SHA256 检查，未发生变化。verification.json 记录验证结果。

固定条件：当前 electron/s 星等换算、0.5 mag 累计阈值间隔、相同局部经验 PSF
分类算法及参数、19–24 mag 描述性线性拟合。每个检测目录分别重建 PSF 参考库；
分类规则相同不表示被选中的参考源相同。没有根据目标斜率删除源或重新选择主拟合范围。

| 变体 | 改动（其余采用原始参数） | 保留计数源 | 负图候选 | 19–24 斜率 |
|---|---|---:|---:|---:|
| baseline | 无 | 4773 | 201 | 0.4621 |
| lower_peak | detection_sigma = 4 | 6312 | 405 | 0.4557 |
| lower_growth | growth_sigma = 1.5 | 4015 | 201 | 0.4534 |
| smooth_1p5 | smooth_sigma = 1.5 | 4360 | 180 | 0.4469 |
| smooth_2p0 | smooth_sigma = 2 | 3828 | 155 | 0.4493 |
| smooth_1p5_lower_growth | smooth_sigma = 1.5; growth_sigma = 1.5 | 3555 | 180 | 0.4456 |
| higher_growth_2p5 | growth_sigma = 2.5 | 5360 | 200 | 0.4449 |
| higher_growth_3p0 | growth_sigma = 3 | 5782 | 194 | 0.4396 |
| smooth_0p7 | smooth_sigma = 0.7 | 4775 | 205 | 0.4429 |

最初六组预先写入 experiment_plan.json。观察到降低生长阈值/加强平滑使质量筛选
通过率下降后，第二轮追加后三组；追加原因和所有结果均保留，未只展示接近 0.6 的结果。

这九组没有达到 0.5。原始检测参数在固定主拟合范围内斜率最高，但这本身不证明
它在 completeness 或 purity 上最优。因此没有将某个变体命名为“最佳优化结果”。

## 如何理解

- 降低峰阈值增加候选，也将负图候选从 201 增到 405；负图比值不是已校准的误检率。
- 降低 growth_sigma 将更多外围像素并入区域，增加混叠/污染标记。baseline 有
  2288 个 blended 标记，growth=1.5 为 3020；可用样本反而减少。
- growth=3 将 blended 标记降到 1355、增加可用样本，但斜率降到 0.440。
  较紧区域也可能漏掉源外围光。因此更多“可用”源并不自动表示更好的总通量测量。
- 这些参数还会改变分割形状、质心和自适应孔径，影响测得通量；保持星等换算公式
  不变，并不意味着每个匹配源的星等不变。
- 基线区间敏感性：19–23 为 0.4897；19–23.5 为 0.4776；19–24 为 0.4621。
  这是拟合区间敏感性，不是检测改进，也没有证明 23 等以内完全完备。
- PSF 分类以保守剔除为目标，保留大量不确定源和可能的恒星；不能把“方法更复杂”
  当作“已经验证了更高分类准确率”。累计点误差相关，不提供未经校准的斜率误差。

0.6 是均匀欧几里得计数的参考斜率，不是给定波段、给定星等区间的硬性正确答案。
例如 SDSS 明亮星系计数研究在 r*<16 讨论这一近似，而并非把所有星等范围都强制
拟合到 0.6：https://arxiv.org/abs/astro-ph/0105545

若要据科学质量选择参数，下一步需要注入-回收测量不同亮度/大小源的 completeness，
验证恒星分类，并检查孔径损失和星等零点。这次未声称完成了这些校准。

## 文件位置

- variants/<名称>/source_detection_final.py：各参数组独立的检测副本。
- variants/<名称>/parameter_changes.json：相对于原始默认值的改动。
- variants/<名称>/detections/：重新运行的检测目录、FITS、诊断图。
- variants/<名称>/analysis/psf_run_*/：两张最终图、分类表、运行摘要。
- star_galaxy_conservative.py：复制的局部经验 PSF 分类脚本，没有修改分类参数。
- photometry_analysis.py：配套测光副本，增加输入/输出目录命令行参数。
- comparison.csv / comparison.png：固定主拟合范围的完整比较。
- quality_diagnostics.csv：质量标记、PSF 参考覆盖、负图和位置匹配诊断。
- fit_sensitivity.csv：预先指定的五个拟合区间，逐组展示，没有择优替换主结果。
- original_file_hashes.json / verification.json：原文件保护及算法一致性检查。

## 运行

在 astro-image 环境，从项目根目录：

```sh
python detection_parameter_experiment/run_experiment.py
python detection_parameter_experiment/summarize_experiment.py
```

已完成阶段会复用。不要通过运行某个检测副本的默认命令来覆盖已有实验结果；
若单独重跑检测，请显式传入一个新的 --output 目录。
单独分析已有检测、同时创建新的分析结果目录，可运行：

```sh
python detection_parameter_experiment/photometry_analysis.py \
  --detections detection_parameter_experiment/variants/baseline/detections \
  --output-root detection_parameter_experiment/additional_analysis \
  --fit-range 19 24
```

分析阶段每次创建新的 psf_run_* 目录。原始 FITS 仅读取，不复制约 2 GB 实验结果
到原 output 目录。PSF 噪声的 effective_gain 默认未指定；未知 mosaic 有效 gain
不直接等同于 header 的 detector gain。星等公式沿用现有版本以保持本次参数对比公平。

## 执行记录

第一轮基线持续刷新 OneDrive 日志时出现一次写入超时，检测表和 FITS 已完成。
随后改为每阶段结束后一次保存日志，使用保存的基线产品补齐诊断图，并继续分析。
基线与此前 PSF 版本重现相同的 4807 个测光源、34 个 likely stars、0.46209 斜率。
九个完整 pipeline 均已运行，所有最终图均已保存。
