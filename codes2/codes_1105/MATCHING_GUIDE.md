# 与现有 injection_recovery.py 的接法

`match_recovery.py` 是单独的匹配模块，不读取 FITS、不运行检测，也不改变你的表。
它适用于你现在的结构：**每个 trial 是一张小图，只注入一颗假源。**

先在 `codes_1105` 中运行 `python3 match_recovery.py`，可以看到五个虚构例子。
这不是你真实图像的恢复率。

## 先解决现有的两个前提

- `run_pipeline()` 现在仍返回 `None`。需要先接上实际 pipeline，返回位置数组 `(N, 2)` 和对应 flux 数组 `(N,)`；没有检测时返回两个空数组。
- `from injection import elliptical_gaussian` 会执行 injection.py 顶层的演示代码。把演示部分放到 `if __name__ == '__main__':` 下面，函数定义留在外面。

位置必须都是相同小图坐标；pipeline 内如果再次裁剪，要先把坐标转换回来。
如果 pipeline 返回的是最终筛选前的源，那么这里的 matched 也只表示检测层面的匹配。

## 1. 循环前多准备一个列表

你现在已有 `injected_rows = []` 和 `detected_rows = []`，在它们旁边加：

```python
original_rows = []
```

## 2. 每张 background 注入之前，也跑一次 pipeline

放在 `background = crop_array(...)` 后面、生成假源前面，保持在外层循环里。
每一张小图都要运行这一段，包括检测数量为零的小图。

```python
# Detect sources before injection
original_centers, original_fluxes = run_pipeline(background)

for detection_id, (position, measured_flux) in enumerate(
        zip(original_centers, original_fluxes), start=1):

    original_rows.append((
        injected_id, detection_id,
        position[0], position[1], measured_flux
    ))
```

你已有的随机参数、注入代码和 `detected_rows` 记录逻辑保持原样。
`original_rows` 记录原本存在的源；`detected_rows` 记录注入之后的源。

## 3. 循环结束，把原图检测也整理成表

放在你已有的 `injected = Table(...)` 和 `detected = Table(...)` 后面：

```python
original_detected = Table(
    rows=original_rows,
    names=('trial_id', 'detection_id', 'x_local', 'y_local', 'flux_before'),
    dtype=('i8', 'i8', 'f8', 'f8', 'f8')
)
```

## 4. 最后调用独立模块

```python
from match_recovery import match_recovery

matches = match_recovery(
    injected, detected, original_detected,
    match_radius=3
)

print(matches)

output_dir = Path(__file__).resolve().parent / 'outputs'
output_dir.mkdir(parents=True, exist_ok=True)
matches.write(output_dir / 'recovery_matches.ecsv', overwrite=True)
```

`3` 的单位是 pixel，是初始示例值，不是通用标准。
应在明亮、孤立的注入源上检查位置差，再决定匹配半径。

## 匹配模块内的四步

1. 根据 `trial_id`，只查看对应那张小图的检测。
2. 计算每个检测位置到注入位置的距离，找出匹配半径内的候选。
3. 检查原图附近有没有已知源。
4. 根据候选数量和原图情况记录状态，不强行选最近的一个。

默认“原图附近”的范围是 `match_radius + 3*sigma_major`，它只是保守的初始规则。
也可以明确传入 `isolation_radius=20`，所有源都用 20 pixel。
这个范围要和匹配半径分开：匹配半径管位置偏移，isolation_radius 管已知邻居。

| status | 意义 |
|---|---|
| matched | 半径内只有一个候选，附近没有原图已知源，且 flux 有效 |
| no_match | 半径内没有候选；也可能被合并到远处的大源，不能证明像素里完全没有信号 |
| ambiguous | 半径内有多个候选，暂不决定哪一个对应假源 |
| near_original | 只有一个候选，但附近原本有源，可能是原有源或混叠 |
| invalid_flux | 只有一个孤立候选，但其 flux 为非正值或非有限值 |

`detection_id=-1` 表示没有选定唯一候选，不是原图的背景 label。
`distance` 是小图坐标的位移；`flux_ratio = flux_out / flux_in` 只在 matched 时计算。
我们不要求 flux_ratio 接近 1，否则会把你要研究的测光误差提前删掉。

## 怎样看结果

```python
matched_sources = matches[matches['status'] == 'matched']
uncertain_sources = matches[
    (matches['status'] == 'ambiguous')
    | (matches['status'] == 'near_original')
]
```

如果先估计孤立区域的完整度，分母也只取 `n_original_near == 0` 的注入实验。
不要删除拥挤案例后仍把结果称为全图完整度。歧义案例应另外报告；把它们全当失败只是保守的匹配规则。

这版只有坐标，没有原图源的轮廓。它不能完整识别远处大星系的光翼、亮星晕、未检出的原有源，也不解混叠。
若之后 pipeline 能返回 segmentation map，再加区域重叠检查会更可靠。
