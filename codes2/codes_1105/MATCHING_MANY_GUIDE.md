# 同一张图注入多颗源

这是另一种实验布局；原来的单源版本保留，不要直接把两个版本的表混用。
`match_many_recovery.py` 是独立模块，不读取 FITS、不运行 pipeline，也不修改你的原脚本。

## 先区分三个 ID

- `frame_id`：哪一张注入图。只有一张图时，全是 1。
- `injected_id`：这一张图里的第几颗假源。
- `detection_id`：这一张图里 pipeline 输出的第几行检测。

同一张图里，多个假源共享 frame_id，但 injected_id 不同。
检测结果只记录一次，不能给每颗假源复制一遍检测结果。
检测编号不需要等于 injected_id。

## 生成流程怎么改

关键是把 `sim_image = ...copy()` 放在假源循环外，把 pipeline 调用放到循环之后。
下面假设 background 已经是你要使用的一张图，make_grid 和 elliptical_gaussian 已经定义。
`run_pipeline` 仍需接上你们真正的检测、测光和筛选代码。

```python
frame_id = 1
rng = np.random.default_rng(42)
positions = make_grid(background.shape, spacing=100, margin=30)

sim_image = background.astype(float).copy()
injected_rows = []
detected_rows = []
original_rows = []

# Detect the original image once
original_centers, original_fluxes = run_pipeline(background)
for position in original_centers:
    original_rows.append((frame_id, position[0], position[1]))

# Add several sources to the same image
for injected_id, center in enumerate(positions, start=1):
    flux = rng.uniform(1e3, 2e4)
    sigma_major = rng.uniform(2, 5)
    sigma_minor = sigma_major * rng.uniform(0.5, 1)
    theta = rng.uniform(0, np.pi)

    source = elliptical_gaussian(
        background.shape, center, flux,
        sigma_major, sigma_minor, theta
    )
    sim_image += source

    injected_rows.append((
        frame_id, injected_id, center[0], center[1],
        flux, source.sum(), sigma_major, sigma_minor, theta
    ))

# Detect the combined image once
detected_centers, output_fluxes = run_pipeline(sim_image)
for detection_id, (position, measured_flux) in enumerate(
        zip(detected_centers, output_fluxes), start=1):
    detected_rows.append((
        frame_id, detection_id, position[0], position[1], measured_flux
    ))
```

这些 spacing 和 margin 仅适用于上面示例的小光斑；尺寸变大时要增大间距。
一次不要注入过多假源，否则会人为改变拥挤度、背景估计和 deblending 行为。

## 整理成表，再调用匹配

```python
from astropy.table import Table
from match_many_recovery import match_many_recovery

injected = Table(
    rows=injected_rows,
    names=(
        'frame_id', 'injected_id', 'x_local', 'y_local',
        'flux_in', 'flux_added', 'sigma_major', 'sigma_minor', 'theta'
    ),
    dtype=('i8', 'i8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
)

detected = Table(
    rows=detected_rows,
    names=('frame_id', 'detection_id', 'x_local', 'y_local', 'flux_out'),
    dtype=('i8', 'i8', 'f8', 'f8', 'f8')
)

original_detected = Table(
    rows=original_rows,
    names=('frame_id', 'x_local', 'y_local'),
    dtype=('i8', 'f8', 'f8')
)

matches = match_many_recovery(injected, detected, original_detected)
print(matches)
```

如果以后跑多张注入图，在外面增加 frame 循环；每张图从未注入的 background 副本开始。
三张表都使用相同的 frame_id 和坐标系统。
没有检测时 pipeline 返回空数组；原图每个 frame 也必须跑，即使没有任何检测。

## 多源版本额外处理什么

假源 A 和 B 如果都只找到检测源 9，不能把两颗都记为 matched。
新版本会把它们记为 `shared_detection`，保留不确定性，不强制挑一个赢家。

| status | 含义 |
|---|---|
| matched | 互相唯一的候选，附近没有已知原源、也没有靠近的其他注入源，flux 有效 |
| no_match | 匹配半径内没有候选 |
| ambiguous | 一颗假源附近有多个检测候选 |
| shared_detection | 同一个检测候选落在多颗假源的匹配半径内 |
| near_original | 附近本来就有真实图像的检测源 |
| near_injected | 附近有另一颗假源，可能人为制造了混叠 |
| invalid_flux | 唯一候选的 flux 非正或非有限 |

只有 `status == 'matched'` 的行才是这套保守规则下接受的恢复候选。
两条 shared_detection 行可以显示同一个 detection_id，表示它们在争同一个候选；这不是重复的成功恢复。
程序保证 matched 行中 `(frame_id, detection_id)` 不重复。

默认位置匹配半径是 3 pixel，只是起点，需要用实际位置误差检查。
与原图源的距离检查仍是 `match_radius + 3*sigma_major`。
假源之间暂用 `3*(sigma_major_i + sigma_major_j)` 判断主要光斑可能重叠，是保守近似。

这不是全局最优分配：即使某些歧义可通过更复杂的优化处理，这版也先保留 ambiguous。
它不使用 flux 一致性阈值，以便之后检查测光误差。
原图星晕、大星系光翼以及未被原图 pipeline 检出的源仍不能仅靠中心距离完整识别。

## 可以先看演示

在 codes_1105 中运行：

```bash
python3 match_many_recovery.py
```

这只会显示虚构的三个注入源：第一颗正常匹配，另外两颗争同一个检测源，因此都不算 clean recovery。
