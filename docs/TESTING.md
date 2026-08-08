# 测试分层

## 默认测试

`pytest -m "not gpu and not docker and not device"` 覆盖数据库、项目导入、标注事务、
训练门禁、快照、协议、配置生成、白名单打包、大小边界和 UI 纯逻辑。

## 发布环境锁

```powershell
.\scripts\export_yolo_lock.ps1
```

该命令更新 `locks/` 中的 Conda explicit MD5 锁、pip 全量版本清单、安装内容哈希
和总清单。正式发布若需要离线逐 wheel 复现，再运行：

```powershell
.\scripts\export_yolo_lock.ps1 -MaterializeWheelhouse
```

后者需要联网和较大的 `build/wheelhouse-win-64/` 临时空间，并生成 pip
`--require-hashes` 锁；wheelhouse 本身是构建缓存，不应提交或装入安装包。

## GPU 测试

在 Conda `yolo` 和 RTX 5060 上运行：

```powershell
.\scripts\test_all.ps1 -IncludeGpu
```

GPU 测试逐一覆盖传统 YOLOv5n/s、YOLOv8n/s、YOLO11n/s、YOLO26n/s 的加载、
单轮微型训练、预测、结果解析和 `best/last` 工件。

## Docker 转换测试

准备 Pulsar2 6.0 与 TPU-MLIR 镜像后运行带 `docker` 标记的测试。CI 的无工具环境
只验证命令、配置、MUD 和包结构；转换成功不能替代真机验收。

## 真机测试

`device` 测试不会在普通测试中自动运行。MaixCAM-Pro 验证 8 个模型；
MaixCAM2 对 8 个模型分别验证 NPU2 和 VNPU。每个包需要相机、显示、坐标回映射
和 1000 帧稳定性检查。
