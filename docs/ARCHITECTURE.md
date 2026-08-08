# 架构与数据约束

## 进程边界

```text
PySide6 UI
  ├─ ProjectController ─ SQLite / images / snapshots
  └─ QProcess
       ├─ train worker
       ├─ predict worker
       └─ deploy worker ─ Docker CLI
```

UI 进程不导入 Torch。worker 的 stdout 只写 JSONL 协议，普通日志写 stderr
及运行日志文件，避免进度消息被第三方输出破坏。

消息信封：

```json
{
  "protocol_version": 1,
  "job_id": "uuid",
  "seq": 1,
  "type": "progress",
  "payload": {}
}
```

接收端按 `job_id + seq` 去重并拒绝倒序消息。

## 数据真源

SQLite 是标注编辑真源。YOLO 文本文件只存在于不可变训练快照和显式导出中。
所有写入使用事务，图片修订号用于阻止过期 AI 结果覆盖用户修改。

- `unreviewed`：尚未人工确认。
- `draft`：AI 建议或尚未确认的人工编辑。
- `verified`：用户明确按 `D` 确认。

只有 `verified` 图片进入训练快照。AI 空建议也会记录完成，但在人工确认前不进入
训练。确认后的空框图片是负样本。

## 训练运行

每次运行保存参数、数据快照、类别顺序、划分清单、seed、日志、指标和 checkpoint。
新训练不会隐式使用历史 checkpoint。断点恢复只允许同一运行、同一快照和同一配置。

传统 YOLOv5 和现代 Ultralytics 通过后端适配器统一输出指标及工件；二者不可交换
checkpoint。

## 部署运行

转换工作目录和最终部署目录分离。最终目录从白名单重新构建，绝不直接压缩运行
目录。转换中间产物在发布后清理，审计报告和运行日志保存在部署包外。

项目目前只产生部署包，不自动刷写设备固件。物理设备验收未完成时状态保持
`needs_device_validation`。
