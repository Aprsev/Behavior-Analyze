# DeepLabCut：遮挡场景小鼠关键点方案

这个目录是项目中的第三条独立分析路径。它使用 DeepLabCut 3 的
`SuperAnimal-TopViewMouse` 预训练模型，不依赖本机训练结果，也不会改动
`traditional/` 或 `unet/`。代码在本仓库维护，通过 GitHub 同步到有 NVIDIA
GPU、视频数据和网络连接的运行电脑。

默认组合为：

- `superanimal_topviewmouse`：俯视实验小鼠的 27 关键点预训练数据体系；
- `hrnet_w32`：top-down 姿态网络；
- `fasterrcnn_resnet50_fpn_v2`：先检测小鼠，再在检测框内识别关键点；
- `video_adapt=true`：用当前视频做无标注的自监督适配，以减小跨场景偏差和抖动。

该模型不能“看穿”完全不透明的障碍物。这里的遮挡鲁棒性来自三个层次：检测框保留
动物整体；头部和身体分别使用多个相关关键点进行置信度融合；只对很短的内部缺口
插值。超过 `max_gap_sec` 的完全遮挡保持 `NaN`，不会伪造轨迹。

## 1. 在运行电脑安装

建议新建独立环境，不要把 DLC 安装进当前传统/U-Net 环境。以下以 Conda 和
CUDA 12.4 为例，实际 PyTorch 命令应按运行电脑的显卡驱动从 PyTorch 官网选择。

```powershell
conda create -n behavior-dlc python=3.11 -y
conda activate behavior-dlc
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r dlc/requirements-dlc.txt
```

首次推理会从 DeepLabCut Model Zoo 自动下载预训练权重，因此运行电脑第一次执行时
需要联网。下载完成后权重保存在该环境的 DLC 缓存中，不需要提交到 GitHub。

## 2. 配置一个视频

```powershell
Copy-Item dlc/config.example.json dlc/config.json
```

编辑 `dlc/config.json`。其中所有相对路径均相对于这个 JSON 所在目录，而不是终端的
当前目录。必须设置：

- `video`：原始视频；
- `roi_json`：已有的四角 ROI 文件，点序为左上、右上、右下、左下；
- `output_dir`：本次结果目录；
- `arena_width_cm`、`arena_height_cm`：场地实际尺寸。

先做环境检查：

```powershell
python dlc/run_dlc.py check --config dlc/config.json
```

## 3. 推理和导出

一条命令完成预训练模型推理、遮挡后处理、厘米坐标换算和视频绘制：

```powershell
python dlc/run_dlc.py all --config dlc/config.json
```

为了先判断零样本效果，可暂时把 `video_adapt` 改为 `false`；正式比较时建议开启。
显存不足时依次把 `batch_size` 降为 2 或 1。`detector_batch_size=1` 是稳妥默认值。

如果 DLC 推理已经完成，仅重跑本项目后处理：

```powershell
python dlc/run_dlc.py postprocess --config dlc/config.json
```

也可以明确指定 DLC 生成的 HDF5：

```powershell
python dlc/run_dlc.py postprocess --config dlc/config.json --predictions "D:\path\video_DLC_....h5"
```

## 输出

`output_dir` 中的主要文件：

| 文件 | 内容 |
| --- | --- |
| `trajectory.csv` | 每帧头部、身体的厘米/像素坐标、原始坐标、置信度和插值标记 |
| `annotated_output.mp4` | 绿色身体质心、红色头部，橙色为通过阈值的原始关键点 |
| `quality_report.json` | 有效率、帧率、所用预测文件及实际找到的 bodyparts |
| `dlc_keypoints.h5/.csv` | 27 个关键点的原始 DLC 输出副本，用于审计 |

厘米坐标与项目其他方案一致：原点为 ROI 左上角，x 向右、y 向下。最终坐标经过
三帧中值滤波；原始融合坐标保留在 `*_px_raw` 列。`*_interpolated=true` 表示该帧
来自短缺口插值。

## 遮挡参数与验收

- `pcutoff=0.35`：越高越保守，低置信度关键点更容易变为缺失；先试 0.35，再根据
  `quality_report.json` 和视频在 0.25–0.6 内调整。
- `max_gap_sec=0.2`：仅填补两端都有可靠观测、持续时间不超过此值的缺口。涉及精确
  时间分析时可设为 0，完全禁用插值。
- `median_window=3`：抑制单帧跳动；设为 1 可关闭。

先裁取一段同时包含无遮挡、部分遮挡、完全遮挡、靠墙、转身和静止的视频做验收。
逐段检查 `annotated_output.mp4`，并将随机帧和所有低置信度片段与人工标注比较。
特别注意：遮挡物若让检测器完全看不到小鼠，预训练模型只能报告缺失；若障碍物外观、
相机角度或头戴设备与预训练域差异很大，下一步应在 DLC 项目中人工标注困难帧并用
SuperAnimal 权重迁移微调，而不是无限放宽置信度阈值。

## 开发机测试

不需要安装 DeepLabCut 权重即可测试 HDF 列解析、关键点融合和坐标变换：

```powershell
python -m unittest dlc.test_postprocess -v
```

完整推理必须在目标 GPU 电脑执行。本机没有真实 DLC 输出和运行环境时，无法替代目标
机上的端到端验收。
