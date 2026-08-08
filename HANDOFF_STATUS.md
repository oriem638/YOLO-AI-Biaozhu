# AI-Biaozhu 源码交接说明

交接日期：2026-07-26

## 本包用途

这是当前开发进度的源码交接包，用于在另一台 Windows 电脑继续测试、修复和构建。
它不是安装包。包内已排除旧安装包、未完成的 Nuitka 中间文件、模型权重、训练结果、
缓存、日志和本机数据集。

## 当前完成情况

- 已实现本地图片项目、SQLite 标注真源、矩形框编辑、人工确认与 AI 草稿复核流程。
- 已提供 8 个模型选项：YOLOv5/8/11/26 的 n、s 型。
- 已实现训练参数、数据划分、增强选项、独立 ML worker、JSONL 实时训练事件和历史模型选择。
- 已实现 MaixCAM-Pro / MaixCAM2 转换配置、ONNX 门禁、部署白名单与 30,000,000 字节检查。
- 已针对本轮界面问题完成：
  - 内置 Noto Sans SC 字体并增加字体回退、字形检测；
  - 修复全局不透明控件背景造成的文字覆盖；
  - 顶部工具栏改为两行，右侧各页改为垂直滚动；
  - 长文件名使用省略标签，不再撑大窗口或改变三栏比例；
  - `W` 固定进入框选，`V` 固定进入选择；
  - `Ctrl+Z` / `Ctrl+Y` 改为窗口级动作，同时保护输入控件自己的编辑快捷键；
  - `Space` 按住临时平移，松开恢复原工具；
  - `D`、列表单击和切图统一到单一导航入口，避免一张图片重复加载；
  - 最后一张按 `D` 只保存确认并提示，不重复加载或改变布局；
  - Windows manifest 增加 Per-Monitor V2 DPI 声明。
- 环境探测脚本已修复 Ultralytics 配置目录无写权限导致的误报。

## 已完成验证

- 全套静态检查和单元测试：`170 passed, 9 skipped`。
- Conda `yolo` 环境探测：
  - Python 3.11.15
  - PyTorch 2.11.0+cu128
  - TorchVision 0.26.0+cu128
  - Ultralytics 8.4.82
  - CUDA 12.8
  - NVIDIA GeForce RTX 5060 Laptop GPU
- CUDA 2048×2048 矩阵运算通过。
- 1280×720 离屏界面验证通过，右侧训练页可滚动，三栏未塌缩。
- 字体文件 SHA-256：
  `763146584cf0710223441356b4395e279021b0806c196614377a7a0174ae074a`

## 尚未完成

- 最新修复版的 Nuitka standalone 和 Inno Setup 安装包尚未成功生成。
  上一次 Nuitka 构建在第 4 次依赖分析期间被外部中断，没有产生可用 EXE。
- 需要在另一台 Windows 电脑重新执行独立目录构建和安装包验证。
- MaixCAM-Pro / MaixCAM2 的 Docker 转换仍需安装对应官方镜像并进行真机验收。
- 没有真机验证的转换结果必须保持“待真机验证”，不能标记为已验证发布。

## 建议接手步骤

1. 解压到短路径，例如 `E:\tools\ai_biaozhu`。
2. 安装 Miniconda/Anaconda，并在 PowerShell 中执行：

   ```powershell
   Set-ExecutionPolicy -Scope Process Bypass
   .\scripts\create_yolo_env.ps1
   ```

3. 探测环境和 GPU：

   ```powershell
   conda run -n yolo python .\scripts\probe_yolo_environment.py
   ```

4. 运行测试：

   ```powershell
   .\scripts\test_all.ps1
   ```

5. 从源码启动：

   ```powershell
   conda run -n yolo python -m ai_biaozhu
   ```

6. 构建 Windows standalone：

   ```powershell
   .\scripts\build_windows.ps1 -CondaExe "C:\完整路径\conda.exe" -NuitkaJobs 8
   ```

7. 安装 Inno Setup 后生成安装包：

   ```powershell
   .\scripts\build_installer.ps1
   ```

详细用法见 `README.md`、`docs\USER_GUIDE.md`、`docs\TESTING.md` 和
`docs\MAIX_DEPLOYMENT.md`。

## 交接包内容

- `src\`：应用源码与内置字体。
- `tests\`：单元、Qt 交互、训练、部署和打包检查测试。
- `scripts\`：环境、探测、测试、Nuitka 与 Inno Setup 脚本。
- `docs\`：架构、使用、测试、Maix 部署和验证说明。
- `locks\`、`environment.yml`、`constraints-ml.txt`：环境锁与依赖配置。
- `packaging\`：Windows manifest 与 Inno Setup 配置。
- `third_party\`：传统 YOLOv5 运行适配所需源码和锁定信息。
- `LICENSE`、`THIRD_PARTY_NOTICES.md`：AGPL-3.0 与第三方许可说明。

## 2026-07-26 接续执行记录（本机）

本机工作目录为 `<source-workspace>\ai_biaozhu`，
交接 ZIP 的 SHA-256 已复核为
`5c3dfb7d911c74d59df5e1a4665cb85d37fbf08fd0f447efded7131602443b02`。

- 已按 `environment.yml` 重建/同步 `C:\path\to\conda\envs\yolo`；Python 3.11.15、
  Torch 2.11.0+cu128、TorchVision 0.26.0+cu128、Ultralytics 8.4.82、
  PySide6 6.9.2，CUDA 可用并识别 RTX 5060 Laptop GPU。
- `scripts\test_all.ps1 -CondaExe C:\path\to\conda\Scripts\conda.exe` 已通过：
  Ruff、打包静态检查和 `172 passed, 9 deselected`。新增的两项测试覆盖无 IPython
  环境中的传统 YOLOv5 启动。
- 修复传统 YOLOv5 v7 的环境可复现性：ML 依赖现在明确包含
  `GitPython==3.1.55`、`seaborn==0.13.2`、`tensorboard==2.21.0`；启动器在
  IPython 缺失时仅提供 YOLOv5 非 notebook 路径所需的最小兼容接口，避免将完整
  IPython 交互依赖打入桌面版。
- 真实 CUDA 冒烟：YOLOv8n/s、YOLO11n/s、YOLO26n/s 以及 YOLO26n 增强用例均已
  完成 1 epoch 训练、checkpoint 和预测。证据见
  `build\test-results\gpu-matrix-20260726-elevated\summary.json`。
- YOLOv5n/s 尚不可标记为通过：修复缺失依赖后，在当前中文工作目录下会被官方
  YOLOv5 v7 的 Windows 非 ASCII 路径限制阻断；改用 `C:\tmp` ASCII 测试目录后
  本轮微型训练在超过 10 分钟且无进度日志时被安全终止。保留日志，后续应在短 ASCII
  源码路径（例如 `E:\tools\ai_biaozhu`）重跑。
- 本机未安装 Visual Studio C++ Build Tools / Windows SDK `mt.exe` / Inno Setup，
  因此尚未开始 standalone 构建或新安装包生成，避免在构建末尾必然失败。Maix Docker
  和真机验证同样仍未执行。

## 2026-07-26 发布构建完成（当前 Windows 主机）

- 已使用 Visual Studio Build Tools、Windows SDK 和 Inno Setup 6.7.0 编译器完成
  Nuitka standalone multidist；报告为 `completion="yes"`、`mode="standalone"`。
- 冻结 GUI/Worker 已生成，独立目录含 10,950 个文件、约 6.5 GB；Worker `--help`
  返回 0。YOLO26n 已在 RTX 5060 Laptop GPU 上完成一 epoch CUDA 训练与预测。
- Inno Setup 安装包已生成并完成安装/运行/卸载烟测；负载树哈希一致，Worker 环境报告
  为 `valid=true`、`gpu_ready=true`、`cuda_available=true`。
- 安装包烟测为 `partial_sandbox_validation`，因为测试模式有意跳过真实卸载注册表和
  快捷方式写入。Docker CLI 缺失，WSL2 查询被系统拒绝；两个历史 MaixCAM 地址均超时。
  这些外部验收不得标为通过。

## Docker 转换环境补齐

- 已安装 Docker Desktop 4.83.0，Docker Engine 29.6.2、WSL2 和本地目录挂载诊断
  均通过。
- 已拉取项目要求的 MaixCAM-Pro 镜像 `sophgo/tpuc_dev:latest`，固定 digest 为
  `sha256:d46a29a349f10c893fddc089b2433da5b8ef2a0bce52b81419064bf3e26a31a0`。
- 完整 `pulsar2:6.0` 未提供：本机仅有 `pulsar2:6.0-lite`，不能改名冒充官方完整
  转换器；项目要求以官方 tar 归档导入。两个历史 MaixCAM 地址均探测超时。
