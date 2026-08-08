# AI-Biaozhu 新手完整操作与 MaixCAM 部署指南

本文面向第一次接触本软件、YOLO 训练或 MaixCAM 部署的用户，从软件下载开始，依次
介绍项目创建、图片标注、模型训练、AI 草稿复核、Docker 转换、MaixVision 安装和
真机验证。

适用软件版本：**AI-Biaozhu Maintenance 0.2.3**。

软件概览见：[AI-Biaozhu 软件介绍](README_SOFTWARE_CN.md)。

## 开始前先看这四条

1. 本软件当前只做**矩形框目标检测**，不做分割、OBB、关键点或姿态估计。
2. 训练只读取“已加入训练”并且“人工已确认”的图片；AI 草稿绝不会自动成为训练真值。
3. “生成 Maix 部署文件”不等于“已经安装到设备”，后面还必须使用 MaixVision。
4. 图片、训练快照、模型和 Docker 镜像会占用较多磁盘，正式项目建议放在空间充足的盘。

## 目录

- [1. 下载和安装软件](#1-下载和安装软件)
- [2. 第一次启动前的准备](#2-第一次启动前的准备)
- [3. 新建项目和类别](#3-新建项目和类别)
- [4. 导入图片或 VOC 数据集](#4-导入图片或-voc-数据集)
- [5. 人工标注](#5-人工标注)
- [6. 选择训练图片并通过训练前检查](#6-选择训练图片并通过训练前检查)
- [7. 训练第一个模型](#7-训练第一个模型)
- [8. 使用 AI 自动标注并复核](#8-使用-ai-自动标注并复核)
- [9. 准备 Docker 转换环境](#9-准备-docker-转换环境)
- [10. 生成 MaixCAM 部署文件](#10-生成-maixcam-部署文件)
- [11. 使用 MaixVision 安装到设备](#11-使用-maixvision-安装到设备)
- [12. 真机验收](#12-真机验收)
- [13. 常见问题和解决方法](#13-常见问题和解决方法)
- [14. 备份、日志和求助信息](#14-备份日志和求助信息)

## 1. 下载和安装软件

### 1.1 先确认下载渠道

当前 GitHub 仓库主要提供源码、文档和测试。点击 GitHub 的 `Code → Download ZIP`
得到的是**源代码压缩包**，不是双击即可安装的 Windows 软件。

普通用户应优先从项目维护者明确提供的 Release 或其他可信发布页下载：

```text
AI-Biaozhu-Maintenance-Setup-<版本>-x64.exe
```

如果 Releases 页面没有这个文件，说明当前没有公开安装包。不要从陌生网盘、评论区或
第三方网站下载同名 EXE。

0.2.3 安装器目前没有 Authenticode 数字签名，Windows SmartScreen 可能显示未知发布者。
只有同时满足下面两项时才应继续：

1. 下载地址确实由项目维护者公布；
2. 本机计算出的 SHA-256 与发布页或 `SHA256SUMS` 完全一致。

在安装器所在文件夹打开 PowerShell，可以这样计算：

```powershell
Get-FileHash .\AI-Biaozhu-Maintenance-Setup-0.2.3-x64.exe -Algorithm SHA256
```

SHA-256 只说明文件内容与发布者给出的文件一致，不能单独证明发布者身份。

### 1.2 使用安装版

1. 双击可信的安装器；
2. 阅读安装位置和快捷方式选项；
3. 按向导完成安装；
4. 从开始菜单或可选桌面快捷方式启动 `AI Biaozhu Maintenance 0.2`。

安装版包含 `AI-Biaozhu-Worker.exe` 和运行依赖，普通用户不需要另外安装 Conda。

维护版可以与早期原版共存，但不要同时用两个版本编辑同一个项目目录。

### 1.3 没有安装包时从源码运行（进阶）

源码方式需要较大的下载空间和基本的 PowerShell/Conda 使用经验。新手如果不熟悉环境
配置，建议等待可信安装版。

需要先安装：

- [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)；
- [Git for Windows](https://git-scm.com/download/win)；
- 可用网络；
- 如需 GPU，安装与显卡匹配的 NVIDIA 驱动。

然后：

1. 在 GitHub 选择 `Code → Download ZIP`；
2. 解压到短路径，例如 `D:\AI_Biaozhu_Source`；
3. 打开“Anaconda PowerShell Prompt”；
4. 进入源码根目录；
5. 创建锁定的 `yolo` 环境；
6. 激活环境并启动程序。

```powershell
Set-ExecutionPolicy -Scope Process Bypass
cd D:\AI_Biaozhu_Source
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\create_yolo_env.ps1 -UpdateExisting
conda activate yolo
python -m ai_biaozhu
```

如果必须验证 NVIDIA GPU，把环境创建命令末尾加上 `-RequireGpu`。没有可用 GPU 时不要
加这个参数；训练仍可选择 CPU，但速度通常会慢很多。

## 2. 第一次启动前的准备

### 2.1 电脑侧

建议准备：

- 64 位 Windows 10 1809（build 17763）或更高；
- 一个空间充足的项目磁盘；
- NVIDIA GPU（推荐但不是标注必需）；
- 首次准备官方模型权重时可用的网络；
- Maix 转换时使用的 Docker Desktop。

推荐目录示例：

```text
D:\AI_Biaozhu\Projects\       正式项目
D:\AI_Biaozhu\Deployments\   部署产物
D:\AI_BZ_TMP\                 Docker 转换临时目录，短且纯 ASCII
```

不要把正式项目放进系统临时目录，也不要只备份 `annotations.db` 而忽略图片和训练结果。

### 2.2 Maix 设备侧

部署前准备：

- MaixCAM-Pro 或 MaixCAM2，确认具体型号；
- 质量可靠的 Type-C 数据线；
- MaixCAM-Pro 所需的可启动 TF 卡；
- MaixCAM2 根据具体版本使用 eMMC 或已烧录 TF 卡；
- 设备可以进入系统应用界面；
- PC 和设备能够处于同一网络；
- 最新可兼容的系统与 MaixPy 运行库。

设备官方快速开始：

- [MaixCAM / MaixCAM-Pro 快速开始](https://wiki.sipeed.com/maixpy/doc/zh/README_MaixCAM.html)
- [MaixCAM2 快速开始](https://wiki.sipeed.com/maixpy/doc/zh/README_MaixCAM2.html)
- [系统与 MaixPy 升级](https://wiki.sipeed.com/maixpy/doc/zh/basic/upgrade.html)

升级系统可能格式化设备存储。先备份设备中的应用和数据，再严格选择与硬件型号匹配的
系统镜像。

## 3. 新建项目和类别

### 3.1 新建项目

1. 启动 AI-Biaozhu；
2. 点击左侧“新建”，或选择 `文件 → 新建项目…`；
3. 选择一个**空目录**；
4. 项目名称可直接体现在文件夹名中，例如 `D:\AI_Biaozhu\Projects\ball_detector`。

为兼容传统 YOLOv5 和 Docker，建议项目路径尽量使用短英文、数字、下划线，不要使用
过长路径。

项目建立后会逐步出现：

```text
project.json          项目信息
annotations.db        标注数据库，唯一编辑真源
images/               导入后的图片副本
thumbnails/           缩略图
runs/                 训练快照、日志和 checkpoint
exports/              导出产物
deployments/          设备部署记录
backups/              数据库备份
```

### 3.2 创建类别

1. 打开右侧“标注”页；
2. 在“新类别名称”输入真实类别名；
3. 点击“添加”；
4. 重复直到所有目标类别建立完成。

正式类别名会进入训练、导出和部署包。尽量在大量标注前确定名称和顺序。

如果误建了类别：

- 只是显示文字需要变化：使用显示名称修改；
- 正式类别名需要变化：使用完整重命名，程序会先备份数据库；
- 类别在全项目完全没有框：选中后点“删除空类别”；
- 该类别在任意图片中仍有框：程序会拒绝删除，应先逐张改类或删除错误框；
- 项目必须至少保留一个类别。

## 4. 导入图片或 VOC 数据集

### 4.1 导入普通图片

1. 点击左侧“导入图片”，或 `文件 → 导入图片…`；
2. 选择 JPG、JPEG、PNG、BMP 或 WebP；
3. 等待导入报告；
4. 查看成功、重复、损坏和不支持文件数量。

软件会把图片复制到项目 `images/`、统一 EXIF 方向，并按 SHA-256 去重。删除原始来源
文件不会让已导入项目失效，但项目中的图片副本仍要备份。

新导入图片通常已经加入训练候选，但状态仍是“未复核”。只有后面按 `D` 人工确认，
才会真正进入训练。

### 4.2 导入 MaixHub / Pascal VOC 数据

如果已有 XML 框，不要使用普通“导入图片”，否则 XML 标注不会被读取。

1. 点击左侧“导入 MaixHub/VOC 混合数据”；
2. 或选择 `文件 → 导入 MaixHub/VOC 混合数据集…`；
3. 选择包含 `images/`、`annotations/` 和 XML 的数据集根目录；
4. 阅读预检查结果；
5. 选择“新建项目导入”或“合并到当前项目”；
6. 检查每一个类别映射；
7. 点击“开始导入”；
8. 完成后阅读冲突报告。

混合数据的状态规则：

| 输入情况 | 导入后的含义 |
|---|---|
| XML 中存在框 | 人工已确认正样本 |
| XML 存在但没有框 | 人工已确认负样本 |
| 图片没有对应 XML | 未复核图片 |

合并项目时，人工确认或人工修改过的重复图片不会被静默覆盖。

## 5. 人工标注

### 5.1 创建第一个框

1. 在类别列表选择当前目标类别；
2. 按 `W` 进入矩形框工具；
3. 在目标左上角按下鼠标；
4. 拖到目标右下角后松开；
5. 按 `V` 回到选择工具；
6. 拖动框可移动，拖动八个控制点可缩放；
7. 如果类别错误，选中框、选择正确类别，再点“将类别应用到选中框”。

### 5.2 保存与确认的区别

- 普通“保存”只保存当前编辑；
- `D` 或“确认并下一张 D”会保存并把图片改成“人工已确认”；
- 只有人工已确认图片才能进入训练；
- 没有框的图片也可以按 `D`，首次会提示是否确认为负样本。

画完框忘记按 `D` 是新手最常见的训练数量不足原因。

### 5.3 状态判断

- `unreviewed`：未复核；
- `draft`：AI 待复核或尚未确认的修改；
- `verified`：人工已确认。

左侧筛选可以帮助找到所有未确认或 AI 待复核图片。

### 5.4 常用快捷键

| 快捷键 | 功能 |
|---|---|
| `W` | 框选 |
| `V` | 选择 |
| `A` | 上一张 |
| `S` | 删除选中框 |
| `D` | 保存、确认、下一张 |
| `Ctrl+Z` / `Ctrl+Y` | 撤销 / 重做 |
| `Ctrl+Alt+Z` | 全部撤销当前图片本次打开后的修改 |
| `Ctrl+Shift+Delete` | 批量删除标记入口 |
| `F` | 适应窗口 |
| 按住 `Space` | 平移画布 |

输入框、下拉框或弹窗正在输入文字时，字母快捷键不会触发。

## 6. 选择训练图片并通过训练前检查

### 6.1 训练成员

左侧图片列表支持 Ctrl/Shift 多选、范围选择和当前筛选全选。根据需要使用：

- “加入训练”；
- “移出训练”。

实际训练图片必须同时满足：

```text
已加入训练 + 人工已确认 + 数据完整
```

### 6.2 第一次训练的硬门槛

第一次成功训练前必须满足：

- 至少 100 张不同的训练成员已经人工确认；
- 至少有一张带框正样本；
- 每个启用类别至少有一个实例。

已确认空图片可作为负样本计入 100 张。未复核图片和 AI 草稿都不计数。

如果提示某类别没有实例：

1. 先判断它是真实类别还是误建类别；
2. 真实类别应补充标注，并把相应图片加入训练；
3. 误建类别可回到“标注”页选择它，点击“删除空类别”；
4. 如果程序提示全项目仍有该类别的框，先定位并人工改类，不能强行删除。

项目有一次成功训练后，再训练不再强制 100 张，但仍需要至少一张确认训练成员、一张
正样本和每个启用类别的实例。此时如果当前确认训练成员不足 100 张，应点击“重新训练”；
“一键训练”按钮仍按至少 100 张的首次训练入口规则启用。

## 7. 训练第一个模型

### 7.1 检查 ML 环境

安装版通常保持“ML 环境：自动检测”即可。点击它可以查看 Python、Torch、CUDA、GPU、
Ultralytics 和 YOLOv5 后端状态。

训练设备：

- `0`：明确使用第一块 CUDA GPU；没有 CUDA 会阻止启动；
- `auto`：优先 CUDA，没有时回退 CPU；
- `cpu`：明确使用 CPU；
- 其他检测到的 CUDA 设备：用于多显卡电脑。

### 7.2 选择模型

可选：

- YOLOv5n、YOLOv5s；
- YOLOv8n、YOLOv8s；
- YOLO11n、YOLO11s；
- YOLO26n、YOLO26s。

新手建议先用 n 型模型验证完整流程，再根据精度和速度决定是否改用 s 型。默认模型为
YOLO26n。

### 7.3 设置参数

点击“高级参数…”。常用默认值：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `imgsz` | 640 | 160～2048，必须是 32 的倍数 |
| `epochs` | 100 | 1～5000 |
| `patience` | 20 | 取消勾选“启用早停”后写为 0；启用时不能大于 epochs |
| `batch` | auto | 显存不足时可改小 |
| `device` | 0 | 无 CUDA 时改 `auto` 或 `cpu` |
| `workers` | 0 | Windows 兼容问题时保持 0 |
| `seed` | 42 | 用于复现数据划分和训练设置 |

数据划分可以使用 80/20 的训练/验证，或 70/20/10 的训练/验证/测试。总和必须为
100%，训练和验证必须大于 0；启用测试集后测试比例也必须大于 0。

随机旋转、模糊、水平镜像和垂直镜像默认关闭。先验证基础模型，再根据任务特点增加
增强，不要一次打开所有选项。

### 7.4 训练起点

- “官方预训练权重”：新手默认推荐；
- 历史 `best.pt`：从一次历史成功模型的最佳 checkpoint 开始新训练；
- 历史 `last.pt`：从历史最后 checkpoint 开始一次新训练；
- “恢复中断训练”：只恢复原运行的原快照、原参数和原 `last.pt`。

从历史 checkpoint 新训练和恢复中断训练不是同一件事。

### 7.5 开始训练

1. 点击“一键训练”；
2. 阅读训练前检查摘要；
3. 核对项目总数、已加入训练数、实际训练数、跳过原因和类别统计；
4. 确认后创建不可变快照；
5. 等待训练完成；
6. 查看曲线、混淆矩阵、日志和 `best.pt` / `last.pt`。

训练期间继续编辑项目不会改变这次快照。

如果发生 CUDA OOM，软件会自动降低一次 Batch 重试。仍然失败时依次尝试：

1. 改用 n 型模型；
2. 降低 `imgsz`；
3. 手动设置更小 Batch；
4. 关闭其他占用显存的软件。

## 8. 使用 AI 自动标注并复核

1. 打开右侧“AI 标注”页；
2. 在历史成功模型中选择当前项目的某次训练；
3. 选择 `best.pt` 或 `last.pt`；
4. 设置置信度和 IoU；
5. 建议保持“跳过已完成图片，支持中断后继续”；
6. 点击“AI 自动标注”；
7. 等待任务完成或分阶段继续；
8. 左侧筛选“AI 待复核”；
9. 逐张移动、缩放、删除、补框或改类；
10. 每张确认无误后按 `D`。

AI 标注处理的是当前项目中尚未人工确认的图片，不是只处理左侧临时多选。已确认图片
不会被覆盖；没有检测结果的图片也会记录为空建议。

### 关于 YOLO 数据集导出

设备部署不要求先导出 YOLO 数据集，可以直接使用当前项目成功训练的 checkpoint。

0.2.3 已知问题：`文件 → 导出 YOLO 数据集…` 的图形界面目录选择器只能选择已经存在
的目录，而底层导出又要求目标目录尚不存在，因此当前 GUI 入口可能报“导出目标已存在，
不能覆盖”。这不是标注数据库损坏。不要通过删除项目文件绕过，也不要把导出作为 Maix
部署的必要步骤；等待后续版本修复，或由熟悉项目接口的开发者使用经过验证的导出流程。

## 9. 准备 Docker 转换环境

普通标注和 YOLO 训练不依赖 Docker；只有 Maix 模型转换需要。

### 9.1 安装并启动 Docker Desktop

1. 按 [Docker 官方 Windows 安装说明](https://docs.docker.com/desktop/setup/install/windows-install/)
   安装 Docker Desktop；
2. 推荐使用 Windows 上常见的 WSL2 后端；
3. 启动 Docker Desktop；
4. 等到 Docker daemon 就绪。

转换镜像和 Docker 数据会占用较多磁盘。请提前确认 Docker 数据位置和剩余空间。

### 9.2 在软件中检测

1. 打开“Maix 部署”页；
2. 在“检查目标”选择：
   - `MaixCAM-Pro（TPU-MLIR）`，或
   - `MaixCAM2（Pulsar2）`；
3. 点击“检测 Docker / WSL / 镜像”；
4. 检查 Docker CLI、daemon、WSL2、目标镜像和目录挂载结果。

daemon 未启动时，可以点击“启动 Docker Desktop 并重新检测”。

### 9.3 准备 MaixCAM-Pro 镜像

1. 目标选择 MaixCAM-Pro；
2. 点击“拉取官方镜像…”；
3. 确认下载软件当前要求的 `sophgo/tpuc_dev:latest`；
4. 等待完成；
5. 再次点击“检测 Docker / WSL / 镜像”。

也可以从可信 tar 归档使用“导入镜像 tar…”。`latest` 是可变化标签，每次转换都应让
部署报告记录实际镜像 ID/digest；digest 变化后应重新验证。MaixCAM-Pro 转换容器使用
较高权限，不要运行来源不明的镜像。

### 9.4 准备 MaixCAM2 镜像

MaixCAM2 的 Pulsar2 环境不是普通 `docker pull` 流程：

1. 前往 [AXERA-TECH Pulsar2 6.0](https://huggingface.co/AXERA-TECH/Pulsar2/tree/main/6.0)；
2. 下载完整官方 `ax_pulsar2_6.0.tar.gz`；本项目没有验证 lite 版，不要把它当作替代品；
3. 回到“Maix 部署”页；
4. 选择 MaixCAM2；
5. 点击“导入镜像 tar…”；
6. 选择下载的归档并确认；
7. 等待 `docker load` 完成；
8. 再次检测，确认软件识别到 `pulsar2:6.0`。

镜像操作可以显示字节进度，并支持请求取消。

## 10. 生成 MaixCAM 部署文件

### 10.1 部署硬门槛

- 必须已经有当前项目的一次成功训练；
- 只能选择该项目产生的 `best.pt` 或 `last.pt`；
- 不接受任意外部 `.pt`；
- Docker Linux 容器引擎、正确转换镜像和目录挂载必须可用；WSL2 是推荐后端，检测失败会给出警告；
- 校准图片必须来自当前项目的人工确认图片；
- 临时目录必须是短、纯 ASCII 路径。

### 10.2 打开部署向导

1. 在“Maix 部署”页选择成功模型和 checkpoint；
2. 点击“打开部署向导…”；
3. 在“目标设备”中确认硬件型号；
4. 设置参数；
5. 点击“开始生成部署文件”；
6. 在二次确认窗口再次核对目标设备。

选错 MaixCAM-Pro / MaixCAM2 会生成完全不兼容的模型。

### 10.3 MaixCAM-Pro 推荐起点

| 项目 | 默认或要求 |
|---|---|
| 目标 | MaixCAM-Pro（SG2002 / cv181x） |
| 模型静态输入 | 320×224 |
| 相机分辨率 | 640×480 |
| 量化 | INT8 |
| 校准图片 | 20～200 张人工确认图片 |
| 最终模型 | `model.mud` + `model.cvimodel` |

### 10.4 MaixCAM2 推荐起点

| 项目 | 默认或要求 |
|---|---|
| 目标 | MaixCAM2（AX620E） |
| 模型静态输入 | 640×480 |
| 相机分辨率 | 1920×1080 |
| 量化 | INT8 |
| 校准图片 | 20～100 张人工确认图片 |
| 最终模型 | `model.mud` + `.axmodel` |

NPU 模式：

- “同时生成 NPU2 与 VNPU（推荐）”；
- “仅 NPU2（完整 NPU）”；
- “仅 VNPU / NPU1（保留 AI-ISP）”。

单 NPU2 包要求设备关闭 AI-ISP；单 VNPU 包要求设备开启 AI-ISP。生成应用会检查模式，
不匹配时会提示在设备系统设置的 NPU 选项中调整，并在重启后再运行。

### 10.5 通用参数

- 模型输入宽高：32～4096，且必须是 32 的倍数；
- 相机宽高：1～8192；
- 置信度：默认 0.35；
- NMS IoU：默认 0.45；
- 最大检测数：默认 100；
- 双缓冲：默认开启，降低吞吐抖动但增加内存；
- YOLO26 是端到端输出，向导会提示 IoU/NMS 设置不生效。

### 10.6 选择校准图片

向导只列出当前项目的人工确认图片。使用 Ctrl/Shift 调整选择。

建议：

- 覆盖每一个类别；
- 覆盖明暗、角度、距离和背景变化；
- 不要全部来自连续、几乎相同的视频帧；
- 不要用损坏或刚被外部程序改写的文件。

软件会冻结校准集并核对 SHA-256。校准图在复制过程中变化或损坏时，会尝试使用其他
健康候选补足；仍不足则停止，而不是悄悄减少数量。

### 10.7 产物和临时目录

- “产物目录”：例如 `D:\AI_Biaozhu\Deployments`；
- “ASCII 临时目录”：例如 `D:\AI_BZ_TMP`；
- 临时目录不能包含中文；
- 临时目录和产物目录不能互相包含；
- 不要使用磁盘根目录；
- 不要让其他程序占用本次任务目录。

至少选择一种输出：

- “生成可直接安装的 .maixapp 文件”；
- “生成可编辑工程文件夹（含 main.py，可继续加功能）”。

新手建议两项都保留：`.maixapp` 用于安装，可编辑工程用于排错和二次开发。

### 10.8 转换过程和真实输出

软件会执行：

```text
冻结校准图片
 → 导出 batch 1 / opset 17 / 固定 NCHW ONNX
 → ONNX checker 与 shape inference
 → ONNX Runtime 和 PyTorch 数值对比
 → Docker INT8 转换
 → 白名单打包
 → 回读 ZIP、CRC 和 SHA-256 校验
```

典型输出名称：

```text
YOLO26n-maixcam_pro-best.maixapp
YOLO26n-maixcam_pro-best-editable/
YOLO26n-maixcam_pro-best-deployment-report.json
YOLO26n-maixcam_pro-best-SHA256SUMS.txt
```

`.maixapp` 和可编辑工程通常包含：

```text
app.yaml
main.py
config.json
models/model.mud
models/model.cvimodel              MaixCAM-Pro
models/model_npu.axmodel           MaixCAM2 NPU2
models/model_vnpu.axmodel          MaixCAM2 VNPU
```

包中不会放入 `.pt`、ONNX、校准图片、训练数据、日志、Docker 文件或缓存。

ZIP 或解压后白名单文件总量任一大于 30,000,000 字节时，软件会暂停并列出大文件。
这只是警告，不代表文件损坏。确认设备空间足够后可以“仍然生成”。想缩小可尝试 n 型、
降低部署输入尺寸，或在 MaixCAM2 只生成一种 NPU 模式。

## 11. 使用 MaixVision 安装到设备

### 11.1 安装 MaixVision

从 [Sipeed MaixVision 官方主页](https://wiki.sipeed.com/maixvision) 下载并安装。

不要把生成的 `model.cvimodel` 或 `.axmodel` 当作文本文件在编辑器中打开并保存；二进制
模型被改写后可能无法加载。

### 11.2 连接设备

推荐方法：PC 和设备连接同一个 Wi-Fi 或手机热点。

也可以使用 USB 数据线。MaixCAM 的 USB 通常表现为虚拟网卡，而不是普通串口，这是
正常现象。

1. 在设备 `设置 → WiFi` 连接网络；
2. 在 `设置 → 设备信息` 查看 IP；
3. 打开 MaixVision；
4. 点击左下“连接”；
5. 从扫描结果选择设备；
6. 扫描不到时手工输入设备 IP。

连接成功后，设备的 Launcher 画面可能退出或黑屏，以释放相机和显示资源，这是开发
连接阶段的正常现象。

### 11.3 推荐：先运行可编辑工程

1. 在 MaixVision 选择“打开文件夹/项目”；
2. 打开生成的 `*-editable` 文件夹；
3. 确认其中有 `main.py`、`config.json`、`app.yaml` 和 `models/`；
4. 点击“运行项目”；
5. 不要点“运行当前文件”——它可能只发送 `main.py`，遗漏模型和配置；
6. 查看设备屏幕、MaixVision 图像预览和终端日志；
7. 确认检测框和类别正常后停止运行。

### 11.4 安装成设备应用

使用可编辑工程时：

1. 保持设备连接并在 MaixVision 打开整个工程；
2. 点击左下“安装/安装应用”；
3. 核对应用 ID、名称和版本；
4. 点击“打包应用”；
5. 点击“安装应用”；
6. 安装完成后断开 MaixVision；
7. 在设备 Launcher/应用列表找到应用并运行。

使用已经生成的 `.maixapp` 时，不同 MaixVision 版本的入口可能不同：如果当前版本提供
“选择安装包/安装现有应用包”，直接选择该文件；如果没有这个入口，使用上面的可编辑
工程流程，或按 11.5 节使用官方本地安装方法。

AI-Biaozhu 默认生成：

```yaml
id: ai_biaozhu_detector
name: AI Biaozhu Detector
version: 1.0.0
```

同一设备不能并存两个相同应用 ID。连续安装不同项目的默认包通常会覆盖或更新同一个
应用。如果需要多个模型同时出现在设备菜单中，使用可编辑工程修改 `app.yaml`，为每个
应用设置唯一的 `id`，同时修改 `name` 和 `version`，再由 MaixVision 重新打包安装。

### 11.5 `.maixapp` 无法直接交给 MaixVision 时

软件中的“交给 MaixVision / 默认程序打开 .maixapp”只是在电脑侧调用关联程序，不代表
已经安装。

如果文件关联失败：

1. 点击 AI-Biaozhu 的“打开产物目录”；
2. 手动启动 MaixVision；
3. 优先打开对应的可编辑工程并“运行项目”；
4. 再使用 MaixVision 的安装应用流程。

进阶用户还可以把 `.maixapp` 上传到设备，然后在设备终端执行：

```sh
app_store_cli install /root/你的应用.maixapp
```

也可按 [Sipeed 应用安装说明](https://wiki.sipeed.com/maixpy/doc/zh/basic/app.html)
使用 `maixtool deploy --pkg`。新手优先使用 MaixVision 图形界面。

### 11.6 开机启动

应用安装并验证稳定后，可在设备 `设置 → 开机启动` 选择该应用。先保证程序可以正常
退出、重新进入且不会占死相机或显示资源，再开启自启动。

## 12. 真机验收

至少完成以下检查：

- 设备型号与部署目标一致；
- APP 可以启动，没有模型加载异常；
- 屏幕和 MaixVision 预览持续显示实时画面；
- 框位置与物体边界大体一致；
- 类别顺序和显示名称正确；
- 用训练集内和训练集外实物分别测试；
- 在不同距离、角度、背景和照明下测试；
- 检查漏检、误检和置信度；
- 运行一段时间，观察帧率、内存、卡死和温度；
- 退出再进入一次；
- 重启设备后再运行一次；
- MaixCAM2 单模式时，AI-ISP 与 NPU2/VNPU 配置一致。

只有这些检查完成后，部署才可以从 `needs_device_validation` 视为已真机验证。

软件中“真机验证成功后清理旧备份…”不是部署必需步骤。它会在多次确认后永久删除项目
旧备份和恢复回收区。没有独立备份策略时不要点击；它不会提升模型速度或精度。

## 13. 常见问题和解决方法

### 13.1 下载和启动

| 问题 | 处理方法 |
|---|---|
| GitHub ZIP 双击不能安装 | ZIP 是源码，不是安装器；使用可信 Release 安装包，或按源码步骤建立 Conda 环境。 |
| SmartScreen 提示未知发布者 | 当前安装器未签名。先核对来源与 SHA-256；无法确认时不要运行。 |
| 启动后提示缺少 DLL | 不要只复制单个 EXE。重新安装完整安装版或保留完整 Standalone 目录。 |
| 原版和维护版会不会冲突 | 安装身份和设置隔离，但不要同时打开同一个项目。 |

### 13.2 项目和导入

| 问题 | 处理方法 |
|---|---|
| 新建项目失败，目录非空 | 新建一个真正的空文件夹再选择。 |
| 导入后图片数量变少 | 查看导入报告，常见原因是 SHA-256 重复、图片损坏或格式不支持。 |
| VOC 导入后没有框 | 必须使用“导入 MaixHub/VOC 混合数据”，普通“导入图片”不读 XML。 |
| 合并 VOC 时提示冲突 | 人工确认或人工修改的重复图会保留；按冲突报告人工决定，不要强制覆盖。 |
| 项目越来越大 | 图片会复制进项目，每次训练还有快照、日志和 checkpoint。把项目放大容量磁盘并清理明确无用的历史产物。 |

### 13.3 标注和训练

| 问题 | 处理方法 |
|---|---|
| Delete 键删不掉框 | 当前快捷键是 `S`，也可点“删除选中框”。 |
| 图片有框但训练数量仍不足 | 检查是否按了 `D`，并确认图片仍处于“加入训练”。 |
| 训练按钮不可用 | 首次训练检查 100 张、正样本、所有启用类别实例和 ML 环境。 |
| 提示类别没有实例 | 真实类别补框并加入训练；误建且全项目无框的类别用“删除空类别”。 |
| 删除空类别被拒绝 | 该类别可能在未确认或未加入训练的图片中仍有框，或它是最后一个类别。 |
| `device 0` 需要 CUDA | 高级参数改 `auto`/`cpu`，或在 ML 环境检测确认 CUDA 和 GPU。 |
| CUDA OOM | 软件自动降 Batch 重试一次；仍失败则用 n 型、降低 imgsz、减小 Batch。 |
| 权重下载失败 | 检查网络、代理和磁盘。不要用未知 `.pt` 替换；程序会核对 URL、长度和 SHA-256。 |
| AI 自动标注按钮灰 | 当前项目没有成功训练，或没有可用的 `best.pt`/`last.pt`。 |
| AI 提示没有图片可处理 | 图片可能都已经 verified，或本轮任务已经完成。 |
| AI 草稿为什么没进训练 | 必须逐张复核并按 `D`；这是防止错误标签进入训练的设计。 |
| YOLO 导出提示目标已存在 | 0.2.3 GUI 已知目录选择冲突；不要删除项目数据绕过，等待修复或使用经验证的开发者流程。 |

### 13.4 Docker 和模型转换

| 问题 | 处理方法 |
|---|---|
| Docker CLI 未找到 | 安装 Docker Desktop，并重新打开软件或重启电脑。 |
| Docker daemon 不可用 | 启动 Docker Desktop，等待引擎就绪，再点击检测。 |
| WSL2 不可用 | 按 Docker 官方 Windows 指南启用 WSL2 和虚拟化，重启后再试。 |
| Pro 镜像缺失 | 选择 Pro 后“拉取官方镜像…”，确认 `sophgo/tpuc_dev:latest`。 |
| CAM2 点击拉取没有执行 | 这是设计行为；下载官方 Pulsar2 tar，再点“导入镜像 tar…”。 |
| 检测不到 Pulsar2 | 确认导入后镜像名/tag 为 `pulsar2:6.0`。 |
| 提示短 ASCII 路径 | 把临时目录改成 `D:\AI_BZ_TMP`；避免中文、复杂层级和过长路径。 |
| 校准图不足 | 只能选当前项目的 verified 图片；Pro 至少 20 张，CAM2 至少 20 张。 |
| ONNX 或数值门禁失败 | 不要绕过；换同一成功运行的 best/last，检查模型和类别，必要时重新训练。 |
| 输出目录冲突或已有任务 | 换新的产物根目录或在确认不需要后安全处理失败任务残留，不要覆盖旧结果。 |
| 包超过 30 MB | 这是确认警告；检查设备空间，或用 n 型、较小输入、CAM2 单 NPU 模式。 |
| Pro 文档中看到 `model_int8.cvimodel` | 0.2.3 当前实际打包文件名是 `model.cvimodel`，以部署报告和新指南为准。 |

### 13.5 MaixVision 和真机

| 问题 | 处理方法 |
|---|---|
| MaixVision 找不到设备 | 确保同一局域网；检查数据线和 USB 虚拟网卡；从设备信息读取 IP 手工连接。 |
| USB 插入后没有串口 | MaixCAM 的 USB 通常是虚拟网卡，不是 USB 转串口，属于正常现象。 |
| 运行当前文件后提示找不到模型 | 使用“运行项目”，保证 `models/`、`config.json` 和 `main.py` 一起传输。 |
| APP 启动后黑屏 | 先确认硬件目标没选错、系统和运行库兼容、相机未被其他进程占用。 |
| 有画面但没有框 | 降低置信度，确认类别和模型文件正确，并用训练场景中的目标测试。 |
| `.mud` 提示模型不存在 | 检查对应 `model.cvimodel` 或 `.axmodel` 与 `.mud` 位于包内预期路径，且传输未损坏。 |
| cvimodel 无法解析 | 不要用文本编辑器打开并保存二进制模型；重新从原部署包解压并核对 SHA-256。 |
| CAM2 提示 AI-ISP 模式不匹配 | 在系统设置的 NPU 选项开启或关闭 AI-ISP，重启后运行；NPU2 关闭，VNPU 开启。 |
| 安装新模型后旧应用消失 | 默认 app ID 相同；不同应用需要修改 editable 工程的 `app.yaml` 唯一 ID 后重新打包。 |
| 生成成功仍显示待真机验证 | 正常；电脑转换成功不能替代实体设备测试。 |

如果设备应用闪退，可查看：

```text
/maixapp/tmp/last_run.log
```

也可在 MaixVision 中运行官方 `examples/tools/show_last_run_log.py`，或在设备终端查看：

```sh
cat /maixapp/tmp/last_run.log
```

Sipeed 常见问题页面：
[MaixCAM / MaixPy FAQ](https://wiki.sipeed.com/maixpy/doc/zh/faq.html)。

## 14. 备份、日志和求助信息

### 14.1 备份什么

最可靠的方式是关闭软件后备份整个项目目录：

```text
project.json
annotations.db
images/
runs/
exports/
deployments/
backups/
```

只备份数据库会丢图片，只备份图片会丢框、类别、状态、训练成员和历史运行。

同一块物理硬盘上的另一个分区不等于真正备份。重要项目还应复制到外置盘、NAS 或
受控云端。

### 14.2 在哪里找日志

常见位置：

- 项目 `runs/<run-id>/`：训练任务、日志、指标、快照和 checkpoint；
- 项目 `deployments/` 或选择的产物目录：部署结果；
- 部署产物旁：`deployment-report.json` 和 `SHA256SUMS.txt`；
- `%LOCALAPPDATA%\AI-Biaozhu-Maintenance\AI标注-维护版-0.2\Logs`：应用日志；
- Maix 设备 `/maixapp/tmp/last_run.log`：设备端应用日志。

### 14.3 求助时提供什么

请提供：

- AI-Biaozhu 版本；
- Windows 版本；
- 安装版或源码版；
- 正在执行的阶段；
- 完整报错文字和截图；
- 所选 YOLO 模型、imgsz、Batch 和 device；
- GPU 名称及 ML 环境检测结果；
- Docker 检测结果和目标镜像；
- MaixCAM-Pro / MaixCAM2 具体型号；
- MaixPy 版本；
- 对应任务日志或部署报告。

不要公开上传：

- 训练图片和未脱敏数据集；
- 整个 `annotations.db`；
- GitHub 令牌、密码或 Cookie；
- 家庭目录、客户名称和内部网络地址；
- 包含商业模型或敏感类别的部署文件。

## 官方参考资料

- [MaixVision 下载与使用](https://wiki.sipeed.com/maixpy/doc/zh/basic/maixvision.html)
- [MaixVision 官方主页](https://wiki.sipeed.com/maixvision)
- [MaixCAM / MaixCAM-Pro 快速开始](https://wiki.sipeed.com/maixpy/doc/zh/README_MaixCAM.html)
- [MaixCAM2 快速开始](https://wiki.sipeed.com/maixpy/doc/zh/README_MaixCAM2.html)
- [MaixPy 应用开发与本地安装](https://wiki.sipeed.com/maixpy/doc/zh/basic/app.html)
- [MaixCAM 系统与 MaixPy 升级](https://wiki.sipeed.com/maixpy/doc/zh/basic/upgrade.html)
- [MaixCAM-Pro 模型转换](https://wiki.sipeed.com/maixpy/doc/en/ai_model_converter/maixcam.html)
- [MaixCAM2 模型转换](https://wiki.sipeed.com/maixpy/doc/en/ai_model_converter/maixcam2.html)
- [MaixPy / MaixCAM FAQ](https://wiki.sipeed.com/maixpy/doc/zh/faq.html)
- [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)

## 一页式完成检查表

- [ ] 从可信来源获得安装器，或正确建立源码环境；
- [ ] 项目建立在空间充足且已备份的位置；
- [ ] 类别名称和顺序核对正确；
- [ ] 图片或 VOC 数据正确导入；
- [ ] 训练成员已经逐张按 `D` 人工确认；
- [ ] 首次训练满足 100 张、正样本和所有类别实例；
- [ ] 训练成功并检查 `best.pt` / `last.pt`；
- [ ] AI 草稿已经人工复核，而不是直接用于训练；
- [ ] Docker、WSL2 和目标转换镜像检测通过；
- [ ] 选择了正确的 MaixCAM 目标和校准图片；
- [ ] 部署报告与 SHA-256 清单已经保存；
- [ ] MaixVision 使用“运行项目”验证完整工程；
- [ ] APP 已安装到正确设备；
- [ ] 完成真实目标、不同环境、重启和稳定性测试；
- [ ] 未在没有独立备份时执行“清理旧备份”。
