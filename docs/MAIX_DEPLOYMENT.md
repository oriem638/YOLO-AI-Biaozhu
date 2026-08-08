# Maix 模型部署

## 共同前置条件

1. 选择本项目某个成功训练运行的 `best.pt` 或 `last.pt`。
2. 选择人工确认图片作为 INT8 校准集。
3. 导出 batch 1、opset 17、固定 NCHW 的静态 ONNX。
4. 完成 ONNX checker、shape inference、ONNX Runtime 和 PyTorch 对比。
5. Docker 镜像 ID 和 digest 必须写入转换报告。

教程节点名称会随模型和 Ultralytics 版本变化。软件先分析实际 ONNX 图，再检查
候选节点的数量、尺度和通道含义，不把教程字符串当成永久常量。

## MaixCAM-Pro

- 目标处理器：`cv181x`
- 默认尺寸：`320×224`
- 量化：INT8
- 运行文件：`model.mud`、`model_int8.cvimodel`
- 官方文档：<https://wiki.sipeed.com/maixpy/doc/en/ai_model_converter/maixcam.html>

## MaixCAM2

- Pulsar2 参数：`--target_hardware AX620E`
- 默认尺寸：`640×480`
- `NPU2`：完整 NPU，写入 MUD 的 `model_npu`
- `NPU1`/VNPU：保留 AI-ISP 资源，写入 `model_vnpu`
- 官方文档：<https://wiki.sipeed.com/maixpy/doc/en/ai_model_converter/maixcam2.html>

默认同时生成两个模型；也可单独生成 NPU2 或 VNPU。单模式 MUD 和 APP 必须完成
对应固件上的加载测试后才能标记为已验证。

## 最小包

部署输出先在空目录中依据文件角色生成，再打包并重新读取验证。`.pt`、`.onnx`、
校准图片、日志、转换配置、其他设备模型和缓存均不得进入包。

同时计算 ZIP 大小和解压后总大小。任一值大于 `30,000,000` 字节即显示警告，
但在用户明确确认后允许生成。软件不会为了满足大小而删除必需的模型文件。

完整 MaixPy APP 使用 `app.yaml` 的 `files` 白名单，并通过 `maixtool release`
兼容的目录结构生成。APP 规范：
<https://wiki.sipeed.com/maixcdk/doc/convention/app.html>
