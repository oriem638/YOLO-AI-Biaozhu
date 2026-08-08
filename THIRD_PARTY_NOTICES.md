# Third-party notices

## Bundled CJK font

The application includes `src/ai_biaozhu/assets/fonts/NotoSansSC[wght].ttf`
and its adjacent `OFL.txt`. Noto Sans SC is licensed under the SIL Open Font
License 1.1. The canonical source and license are
<https://github.com/google/fonts/tree/main/ofl/notosanssc>. The vendored font
is pinned by `assets/fonts/SHA256SUMS`; the runtime rejects a mismatched asset.

AI 标注本身采用 AGPL-3.0-only。第三方组件的版权和许可仍归各自权利人所有。
发布构建会在 `THIRD_PARTY_LICENSES/` 中复制所用 Python wheel 自带的许可或
notice 文件，并生成记录实际分发版本和文件名的 `index.json`。

## 随安装包分发的主要组件

| 组件（发布基线） | 用途 | 许可/项目信息 |
|---|---|---|
| CPython 3.11 | 内嵌 Python 运行时 | Python Software Foundation License。<https://docs.python.org/3/license.html> |
| PySide6 / Qt 6.9.2、Shiboken 6.9.2 | 桌面界面 | PySide6 元数据提供 LGPL-3.0-only、GPL-2.0-only 或 GPL-3.0-only 选择；本 AGPL 发布使用 GPL-3.0-only 兼容路径。<https://doc.qt.io/qtforpython-6/licenses.html> |
| Ultralytics 8.4.82 | YOLOv8、YOLO11、YOLO26 训练、推理和导出 | AGPL-3.0；商业闭源使用需另行取得 Enterprise 许可。<https://www.ultralytics.com/license> |
| ultralytics/yolov5 v7.0 | 传统 anchor-based YOLOv5n/s 训练、推理和导出 | GPL-3.0-only。锁定 commit `915bbf294bb74c859f0b41f1c23bc395014ea679`。<https://github.com/ultralytics/yolov5/tree/v7.0> |
| PyTorch 2.11.0+cu128 / TorchVision 0.26.0+cu128 | 模型训练和推理 | BSD 类许可及各自 NOTICE。<https://pytorch.org/> |
| NVIDIA CUDA/cuDNN 运行库 | PyTorch GPU 运行时 | 随 PyTorch wheel 分发的 NVIDIA 组件仍受 NVIDIA 对应许可条款约束；发布者须保留 wheel 中的 notices 并复核再分发条款。<https://docs.nvidia.com/cuda/eula/index.html> |
| Pillow 11.3.0 | 图片解码和 EXIF 方向统一 | MIT-CMU。<https://python-pillow.org/> |
| NumPy 1.26.4 / OpenCV Python 4.11.0.86 / Albumentations 2.0.8 | 图像与数组处理、数据增强 | 分别遵循其 wheel 中的 BSD/Apache-2.0/MIT 许可和第三方 notices。 |
| ONNX 1.22.0 / ONNX Runtime GPU 1.22.0 / onnxsim 0.4.36 / onnxslim 0.1.82 | 静态模型导出、简化与验证 | Apache-2.0 / MIT / Apache-2.0 / MIT。<https://onnx.ai/> |
| 其他传递依赖 | HTTP、配置、日志、指标和运行时支持 | 实际组件、版本、许可元数据及随 wheel 提供的许可文件见安装目录 `THIRD_PARTY_LICENSES/index.json`。 |

YOLOv5 源码副本由 `third_party/yolov5.lock.json` 固定到上述 tag 和 commit。
构建前会移除其 `.git` 目录，写入 `.ai-biaozhu-yolov5-tag`，并把原始
`LICENSE` 一同分发；安装包不包含 Git 历史。

## 构建期或用户提供的外部工具

| 组件 | 用途 | 项目信息 |
|---|---|---|
| Nuitka 4.1.3 | Windows standalone 构建 | <https://nuitka.net/> |
| Inno Setup 6.3+（本机构建探测为 6.7.3） | Windows 安装包构建 | <https://jrsoftware.org/isinfo.php> |
| Sipeed MaixPy / MaixCDK 文档 | Maix 部署接口 | <https://github.com/sipeed/MaixPy> |
| Sophgo TPU-MLIR | MaixCAM-Pro 转换工具链 | <https://github.com/sophgo/tpu-mlir> |
| Axera Pulsar2 | MaixCAM2 转换工具链 | <https://pulsar2-docs.readthedocs.io/> |

Nuitka、Visual Studio Build Tools、Inno Setup、Git 和转换器 Docker 镜像是
构建期或用户环境工具，不以 Python 包形式复制到 standalone 目录。模型权重
也不预装；应用首次使用时从上游按锁定清单下载并校验，随后可离线使用。转换器
镜像由用户自行安装，应用只记录其镜像 ID 和 digest。

`THIRD_PARTY_LICENSES/` 是从发布环境实际安装的 wheel 自动生成的清单，
不是对所有潜在外部服务或用户自行安装工具的许可替代。发布者仍应审阅
`index.json`，并在依赖更新时重新生成和核对。
