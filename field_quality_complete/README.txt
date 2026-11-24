完整复现 field_quality_review
============================

入口：run_all.py
使用安装有 numpy、scipy、astropy、matplotlib 的 astro-image 环境。
从项目根目录运行：
  /opt/anaconda3/envs/astro-image/bin/python -B field_quality_complete/run_all.py

每次运行重新从原始 mosaic.fits 开始检测，并覆盖本文件夹中的结果。
不复制旧的最终 catalogue 作为 detection 结果，不修改原有实验目录。
14 个方法文件沿用原版本计算代码和注释；photometry_analysis.py 的默认
输入/输出路径指向本目录最终结果。run_all.py 负责依次执行。

执行顺序
1. run_aperture_review.py
   原始图像 -> 背景/RMS -> 饱和、右侧 strip、尖锐伪影和已确认十字缺陷 mask
   -> 检测/分峰 -> 自适应孔径 -> 邻源模型质量检查 -> 孔径回退
   -> 已记录的局部孔径复核 -> 邻源核心/共享像素检查 -> 局部恢复/排除
   -> PSF 参考星位置匹配 -> circular Moffat 分类 -> 基准 observed counts。
   中间结果：results/aperture_review/

2. run_field_quality_review.py
   使用上一步刚生成的 catalogue 和 FITS，重新做全图内部多峰检查、
   亮源重新拟合和恢复，再运行恒星分类、测光误差和 observed counts。
   最终结果：results/field_quality_review/

3. validate_field_quality_review.py
   检查质量筛选、固定孔径通量、共享像素、样本选择及原文件保护，生成复核图。

4. compare_reference.py
   与原 bright_source_counts_experiment 中保存的结果逐列比较，
   比较重新生成的所有 FITS 图像数组及 counts/斜率，核对原文件哈希。
   检查报告：results/REPRODUCTION_CHECK.txt
   浮点极小舍入差单独列出；任何实质差异会报错，不会用旧结果替换新结果。

常用输出
  results/field_quality_review/detection_overview.png
  results/field_quality_review/cumulative_number_counts.png
  results/field_quality_review/sources.csv
  results/field_quality_review/classified_sources.ecsv
  results/field_quality_review/number_counts.ecsv
  results/field_quality_review/review_queue.ecsv
  results/field_quality_review/FIELD_REVIEW_SUMMARY.txt
  results/aperture_review/detection_products.fits

最终 catalogue 使用本文件夹 results/aperture_review/ 中新生成的 FITS。
全图复核不改变背景、分割和 mask，所以无需再复制一份大 FITS。
星等：MAGZPT - 2.5 log10(flux_ADU * GAIN / EXPTIME)。
拟合区间：19--23 mag；分类到 22 mag；未知源仍作为候选保留。
counts 是观测计数，没有完备性修正；孔径通量不是保证完整的总通量。

保留的历史输入
  ../Astro/Fits_Data/mosaic.fits：原始科学图像，只读。
  inputs/scored_sources.ecsv：旧参考目录副本，仅用于按位置寻找已选 PSF 参考星。
  inputs/original_file_hashes.tsv 和 protected_files.tsv：原文件保护清单。
  原 bright_source_counts_experiment 的结果仅在最后的复现验证时读取。
  主要分析流程不读取旧 results 下的科学数据。

此版本的目标是忠实复现，不是改变筛选方法或提高斜率。
因此保留 reviewed_wings.py / reviewed_blends.py / reviewed_recovery.py 中
已明确记录的按位置复核规则；不能直接声称它已经是全自动注入验证流程。
未来注入工作需单独处理这些人工复核，不能把它们当作普适选择函数。

run_summary.json 沿用原检测器的简短运行记录；不是额外的实验结果分支。
results/mpl_cache 是绘图缓存；没有程序运行时可以删除。
