# AI 标注

Windows 本地 YOLO Detection 数据集标注、训练、AI 草稿复核与 MaixCAM
部署工具。SQLite 是标注编辑的唯一真源；训练、推理和模型转换均在独立进程中
运行，不阻塞 PySide6 界面。

完整操作说明见 [用户指南](docs/USER_GUIDE.md)，Maix 转换与最小部署包规则见
[Maix 模型部署](docs/MAIX_DEPLOYMENT.md)，本机已执行与待外部验收项目见
[验证报告](docs/VALIDATION_REPORT.md)。

## 独立维护版身份

此源码树构建的 Windows standalone 与安装器是 **维护版 0.2**，可以与原版并存，
不会覆盖原版安装或复用原版的设置、模型缓存、日志和恢复状态。两版的隔离边界
如下：

- 安装 GUID：`{4C9330ED-77CB-4F81-A467-06B4D6A8FB2B}`（与原版不同）；
- 默认安装目录：`AI-Biaozhu-Maintenance-0.2`；
- 开始菜单/桌面名称：`AI Biaozhu Maintenance 0.2`；
- Qt 设置身份：`AI-Biaozhu-Maintenance / AI标注-维护版-0.2`；
- 用户数据身份：`AI-Biaozhu-Maintenance / AI标注-维护版-0.2`；
- 安装器和审计发布物统一使用 `AI-Biaozhu-Maintenance-...` 文件名。

安装维护版不要求卸载原版。项目目录仍由用户主动选择；若两版打开同一个项目，
它们操作的是同一份项目数据库，因此不要同时编辑同一项目。

源码模式仍沿用 Python 包名 `ai_biaozhu`。如果需要同时保留两个源码版本，请分别
使用虚拟环境；不要把原版和维护版安装进同一个 Python 环境。这个限制不影响上述
standalone 或安装器的并存。

## 0.2.1 热修复

- 修复空项目（图片列表为 `0/0`）导入 MaixHub/VOC 数据时，被“保存当前图片”
  检查静默中止的问题。
- VOC 初始检查、类别映射复检和正式导入期间显示明确阶段文字、等待光标，并禁用
  重复导入入口；完成或失败后恢复按钮状态。
- VOC 导入关键阶段写入应用日志，便于定位预检、正式导入和异常退出。
- 新增空项目导入、保存失败保护及阶段反馈的端到端 UI 回归测试。
- `0.2.1` 沿用维护版 `0.2.0` 的安装标识和运行时数据身份，可直接覆盖升级；原版
  `AI Biaozhu` 仍保持独立，不会被覆盖。

## 0.2.0 维护版更新摘要

- MaixHub/VOC 混合导入同时识别已标注和未标注图片；重复图片只在安全条件下替换
  纯 AI 草稿，人工确认或人工修改内容会保留并列入冲突报告。
- 标注可在“完整显示、仅显示框、全部隐藏”之间切换；修复缩放手柄拖动时误切换
  到重叠框、残影和连线问题，并支持重叠框循环选择。
- AI 自动标注支持同类别重框去重，默认 IoU 为 `0.80`，可在 `0.70～0.95` 调整；
  只处理尚未人工修改的 AI 草稿。历史草稿也可先预览、备份后批量清理。
- `A` 返回上一张，`S` 删除当前框；`D` 确认前会提示高度重叠框和触碰图像边缘的框。
  “全部撤销”和“删除所有标记”保留可恢复的备份机制。
- 图片列表支持 Ctrl/Shift 多选、当前筛选全选、区间表达式和“选择到此处”；所选图片
  可批量删除、加入训练或移出训练，训练时只使用明确纳入且已人工确认的样本。
- 训练前会分别列出未标注与 AI 未确认图片，并生成不可变训练快照、精确样本计数和
  SHA-256 指纹；早停可由用户开启并设置 patience/监控指标，结束原因会明确显示。
- 训练曲线拆分为损失图（box/cls/dfl）和精度图（mAP50/mAP50-95）。
- Docker 镜像导入显示字节进度、耗时与速度，支持取消；重启后会区分 Docker 服务
  不可用、镜像未检查和镜像确实缺失，并可启动 Docker Desktop 后自动重新检测。
- MaixCAM-Pro 与 MaixCAM2 参数互相隔离；部署校准图片会冻结并校验 SHA-256，损坏
  候选可由健康图片替换。部署可分别生成 `.maixapp` 和可编辑工程文件夹。
- “生成部署文件”与“安装到设备”在界面中明确区分；只有用户确认真机部署成功后，
  才允许把旧备份移动到项目内可恢复的 `.trash`，不会直接永久删除。

## 主要功能

- 三栏标注工作台：`V` 选择、`W` 框选、`A` 上一张、`S` 删除框、`D` 保存、
  人工确认并进入下一张；隐藏标注只改变显示，不修改数据库或导出结果。
- 图片复制导入、SHA-256 去重、EXIF 方向统一、损坏文件报告，以及混合 VOC
  已标注正样本、已确认空负样本和无 XML 未确认图片的严格区分。
- 传统 YOLOv5n/s，以及 YOLOv8n/s、YOLO11n/s、YOLO26n/s 共 8 个模型。
- 首次训练至少需要 100 张已选择且人工确认的可训练图片；已有成功模型后允许用至少
  1 张新确认样本迭代。未标注和 AI 未确认草稿始终跳过，不会因用户继续而进入训练。
- 可调训练/验证/测试比例、分辨率、epochs、patience、batch、设备、workers、
  显式早停开关、随机旋转、模糊及水平/垂直镜像。
- 实时训练进度、分离的损失/精度曲线、日志、预览图、结束原因和检查点；AI 标注
  可选 `best.pt` 或 `last.pt`。
- 安装版优先使用内置 Worker；源码版可自动发现 Conda 环境，也可手动选择环境目录
  或 `python.exe`，检测 Python、Torch、CUDA、GPU、Ultralytics 和 YOLOv5 后端。
- 将项目运行产生的 checkpoint 转换为 MaixCAM-Pro 或 MaixCAM2 部署文件；可独立
  选择 `.maixapp`、可编辑 MaixVision 工程或同时生成两者。生成文件不等于已安装
  到设备，仍须在 MaixVision 中完成连接、安装和真机验证。
- 部署目录使用严格白名单；压缩包或解压总量超过 30,000,000 字节时警告，
  经用户确认后仍可生成。

## 开发环境

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\create_yolo_env.ps1 -UpdateExisting -RequireGpu
conda activate yolo
python -m pytest
python -m ai_biaozhu
```

基线是 Python 3.11、PyTorch 2.11.0+cu128、TorchVision 0.26.0+cu128、
PySide6 6.9.2 和 Ultralytics 8.4.82。环境创建脚本会把传统 YOLOv5 v7.0
按 `third_party/yolov5.lock.json` 中的精确 commit 检出到
`third_party/runtime/yolov5/`。校验通过后脚本会移除 `.git`，并写入
`.ai-biaozhu-yolov5-tag`；构建脚本会再次拒绝任何含 `.git` 或缺少版本标记
的运行时副本。工程不会使用已有的可编辑 `yolov8` 环境。

`environment.yml`、`constraints-ml.txt` 和 `pyproject.toml` 固定直接依赖版本，
并以非 editable 方式安装工程。`locks/` 归档本次 Windows 发布环境的 Conda
explicit URL/MD5、所有已安装 Python 分发版本、wheel `RECORD` 内容哈希和
`pip check` 结果。运行以下命令可从当前 `yolo` 环境重新生成：

```powershell
.\scripts\export_yolo_lock.ps1
```

正式发布可再加 `-MaterializeWheelhouse`：脚本会下载全部精确传递依赖、构建应用
wheel，并生成可由 pip `--require-hashes` 校验的
`locks/requirements-win-64.lock` 与 wheel SHA-256 清单。wheelhouse 位于
`build/`，不会进入源码包或安装包。

没有 ML 依赖时仍可运行数据层测试；没有 Docker 时标注和训练功能不受影响，
转换面板会显示缺少的工具。

## 项目目录

```text
project.json
annotations.db
images/
thumbnails/
runs/
exports/
deployments/
```

部署转换的 ONNX、校准数据和编译临时文件只存在于运行工作区，不进入最终包。
`deployment-report.json` 与 SHA-256 清单位于部署包外。

## Maix 目标

- MaixCAM-Pro：静态 ONNX → TPU-MLIR `cv181x` INT8 →
  `model.mud + model_int8.cvimodel`。
- MaixCAM2：静态 ONNX → Pulsar2 `--target_hardware AX620E` INT8 →
  NPU2、VNPU 或双模式 `.axmodel + .mud`。

转换需要用户安装 Docker Desktop，并准备官方 Pulsar2/TPU-MLIR 镜像。
软件不会静默安装 Docker，也不会把转换器镜像打进安装包。

## 构建

构建主机需要 64 位 Windows、Git、可用的 Conda `yolo` 环境、Visual Studio
2022 C++ Build Tools，以及 Inno Setup 6.3 或更高版本（已兼容 6.7.3）。
Nuitka 的下载缓存固定
在工程 `build/windows/nuitka-cache/`，不会依赖不可写的用户缓存目录。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\test_all.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\test_all.ps1 -IncludeGpu
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\export_yolo_lock.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_installer.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\prepare_release.ps1 `
  -CleanOutput -SourceMirror E:\tools\ai_biaozhu
```

`build_windows.ps1` 使用 Nuitka Multidist 一次生成
`AI-Biaozhu.exe` 与 `AI-Biaozhu-Worker.exe`，两者共享同一个 standalone
依赖树；不采用 onefile，也不会把模型权重打进安装包。构建同时收集关键运行时
依赖的许可文件，并把经过清理的 YOLOv5 v7.0 源码运行时放入第三方目录。

产物位置：

```text
build/windows/AI-Biaozhu.dist/       standalone 目录
dist/AI-Biaozhu-Maintenance-Setup-<版本>-x64.exe 独立维护版安装包
```

`prepare_release.ps1` 从白名单重建源码包，拒绝权重、ONNX、缓存、符号链接、
reparse point、路径穿越和大小写冲突；同时复核 standalone、安装器 PE 头、
许可证和 Nuitka 报告。最终默认在同级
`AI-Biaozhu-0.2.0-maintenance-outputs/` 生成带 `AI-Biaozhu-Maintenance-`
前缀的源码 ZIP、standalone ZIP、安装器、逐文件清单、发布报告和校验清单。
首次同步到空的
`E:\tools\ai_biaozhu` 不会删除已有文件；以后只有存在本工具所有权标记且显式加
`-ReplaceSourceMirror` 才允许替换。

standalone 目录的目标是让最终用户无需安装 Conda。发布前仍必须在无 Conda 的
干净 Windows 机器上完成安装、启动、GPU 检测、训练、卸载验收；首次选择模型时
仍需联网下载对应官方权重，之后可使用本地缓存。

## 许可证

项目代码采用 [AGPL-3.0-only](LICENSE)。第三方依赖与外部工具保留各自许可，
详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
