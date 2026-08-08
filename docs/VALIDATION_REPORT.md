# AI 数据集标注与训练维护版 0.2.2 验收报告

验收批次：2026-08-06 至 2026-08-07
源码工作区：`<source-workspace>\AI-Biaozhu-0.2.0-maintenance`

## 结论

维护版 0.2.2 已完成源码修复、自动回归、Nuitka standalone 构建、安装包构建、隔离安装烟测、真实 0.2.1 原位覆盖升级、现有 2084 张项目副本迁移检查，以及已安装 Worker 的 1 Epoch GPU 短训练。

本轮四项维护目标均已实现：

1. 模型下载/校验进度与 Epoch 训练进度分离；准备期间不再显示假 Epoch，实际训练总轮数使用用户设置。
2. “选到这里”按真实鼠标时序完成正向、反向、同端点及保留 Ctrl 既有选择的区间多选。
3. 类别重命名分为“仅修改显示名称”和“完整重命名类别”；完整重命名保持类别 ID 和标注关联，并影响之后的新训练、导出和部署。
4. 编号输入支持英文逗号、中文逗号、顿号及混合输入，非法表达式整体拒绝。

## 自动回归

执行：

```powershell
C:\path\to\conda\envs\yolo\python.exe -m pytest -q
```

结果：

- `357 passed`
- `9 skipped`
- 用时 `110.16 s`
- 无失败

9 项跳过测试是需要显式环境变量启用的模型矩阵专项门禁：8 项等待 `AI_BIAOZHU_GPU_SMOKE`，1 项等待 `AI_BIAOZHU_MODERN_AUG_SMOKE`。这些项目没有被错误记录为已通过。

静态检查：

- Ruff：通过。
- `git diff --check`：通过；仅有 Windows 行尾转换提示，无补丁格式错误。

## 专项功能验收

### 训练准备与 Epoch

- 权重准备事件使用 `resolving_pretrained_weight`、`downloading_weight` 等独立阶段。
- 真正进入训练时发送 `stage=training, current=0, total=用户设置轮数`。
- Epoch 完成后才更新已完成轮数；首轮中途失败不再被计为完成。
- 下载器具备长度和 SHA-256 双重检查、断点续传、Range 被忽略时安全重下、最多三次重试、取消恢复及陈旧锁恢复。
- 构建中只携带固定哈希的 `yolo26s.pt` 种子权重。
- 重复启动、取消、准备失败、Worker 异常退出与软件重启后的陈旧任务状态均有自动恢复测试。

内置权重：

- 文件：`model-seed/yolo26s.pt`
- 大小：`20,422,725` 字节
- SHA-256：`646F8BC3FE0A656803D95C294F7852321748CB29D13466A1AF8862E2DB384A1B`

### 已安装版 1 Epoch GPU 短训练

测试输入使用现有 2084 张项目的只读验证副本中已冻结训练快照：462 张训练图、58 张验证图、58 张测试图。训练输出写入源码工作区的测试结果目录，不修改原项目数据库。

结果：

- 已安装 Worker 退出码：`0`
- 准备结束时事件：`current=0, total=1`
- 实际完成事件：`current=1, total=1`
- 结束原因：`max_epochs`
- 已完成 Epoch：`1`
- 生成 `best.pt`、`last.pt`、`results.csv` 和训练图表
- `best.pt` SHA-256：`FD4C1BF0E488FD15CE169C3576BE669619A1B3EEE97F1A1837B33073D1EA7FD3`
- 该 1 Epoch 冒烟仅验证训练链路和状态，不作为模型精度结论。

证据：

- `build/test-results/installed-short-train-0.2.2-run2.stdout.jsonl`
- `build/test-results/installed-short-train-0.2.2-run2.stderr.log`
- `build/test-results/installed-short-train-runtime/training/model/`

### “选到这里”区间多选

Qt 真实鼠标事件测试覆盖：

- 403 → “选到这里” → 407，选择 403～407。
- 407 → “选到这里” → 403，仍选择 403～407。
- 起点和终点相同。
- 保留此前 Ctrl 选择的其他图片。
- 清除选择、取消等待和连续重复操作。
- 搜索/筛选时按项目全局编号选择，隐藏图片在取消筛选后仍显示为已选。
- 区间选择不会改变训练成员、标注、确认状态或数据库内容。

### 两种类别重命名

- “仅修改显示名称”只影响画布、类别列表和框列表，可恢复规范名称。
- “完整重命名类别”在备份后事务化修改规范名称，保持类别 ID、颜色、顺序和所有框关联。
- 支持 `BALL → ball` 的仅大小写变化及中文名称。
- 完整重命名后的新训练快照、YOLO/VOC 导出、部署配置和新 Maix 应用使用新名称。
- 旧名称保留为不可见 VOC 导入兼容别名，防止重复类别。
- 历史模型与已生成部署包不原地篡改；重新生成时使用当前类别顺序和规范名称。

### 编号输入解析

已测试英文逗号 `,`、中文逗号 `，`、顿号 `、`、混合分隔符及两侧空格；继续支持单编号、闭区间、多范围组合、去重和排序。非法范围或字符会整体拒绝，不执行部分选择。

## 真实项目数据安全

真实项目：`<validation-project>`

在覆盖安装前已正常关闭旧维护版并生成备份：

- `<validation-project>\backups\pre-install-0.2.2-20260807-054243`
- 数据库 SHA-256：`1DA42744E2C5BA2093CB0200CE9826485AF455B009C3F5FDF2FC139F78EE44A5`
- `project.json` SHA-256：`3376A3AF012C7BB4E766FA3623FB876F6E907DCDEBF17700AB033BFA69D0909C`

完整验证副本：

- `<validation-root>\<validation-project>-0.2.2-20260807-053844`
- SQLite `quick_check=ok`
- SQLite `integrity_check=ok`
- 数据库结构版本：`5`
- 图片：`2084`
- 标注框：`796`
- 类别：`1`

真实项目由 v4 首次迁移到 v5 前，软件还自动创建了 `before-schema-v4-to-v5` 数据库备份。迁移后再次检查仍为 2084 张图片和 796 个框。

## Standalone 与安装包

Nuitka 报告：

- `completion="yes"`
- `mode="standalone"`
- payload 文件数：`10,959`
- 文件树 SHA-256：`79897A94C4FDB59928A3041933509520B3FFBFF2DC345515F2FBBECABD67E5CB`
- `AI-Biaozhu.exe` SHA-256：`4AF94685D166C28AE64A1EBA7BBE47D4B266465E18378D8D36AD8561302EB1E1`
- `AI-Biaozhu-Worker.exe` SHA-256：`B6A157AE5B076134A9305B2A8BF2FCCAE960A2086DD764126F398D6D0957BB09`

安装包：

- 文件：`dist/AI-Biaozhu-Maintenance-Setup-0.2.2-x64.exe`
- 大小：`2,514,006,163` 字节
- SHA-256：`FF5BF191E9AED743BF7A9C53674ECD31CC13A2E5E881F41011BB621C6058B950`

隔离安装烟测：

- 安装、Worker `--help`、GPU 环境探测、GUI 存活 8 秒、卸载和清理均通过。
- 安装后 payload 文件数和文件树哈希与 standalone 完全一致。
- GPU 环境：RTX 5060 Laptop GPU、CUDA 12.8、PyTorch 2.11.0+cu128、Ultralytics 8.4.82。
- 状态为 `partial_sandbox_validation` 仅因为隔离烟测显式禁止修改真实卸载注册项；`failure=null`、`cleanup_failure=null`。
- 隔离烟测前后真实维护版和原始版快捷方式哈希均未变化。

证据：`build/test-results/final-installer-smoke-summary.json`。

## 覆盖安装验收

- 安装目录：`<install-root>\AI-Biaozhu-Maintenance-0.2`
- Windows 登记名称：`AI Biaozhu Maintenance 0.2`
- Windows 登记版本：`0.2.2`
- 已安装 GUI/Worker 哈希与 standalone 一致。
- 维护版 0.2.1 被原位升级；项目、设置、模型记录和维护版快捷方式保留。
- 早期原始软件目录、程序和快捷方式未被覆盖或卸载。

安装日志：`build/actual-upgrade-0.2.2.log`。

## 尚未声明为已通过的外部环境项目

下列项目不影响本轮四项维护功能的交付，但仍需在对应外部环境单独验收：

1. 无 Conda、无开发依赖的第二台干净 Windows 主机。
2. 8 个模型全矩阵 GPU 专项 smoke 和现代增强专项 smoke。
3. 真实大型 Docker 转换镜像的长时间导入、取消与停滞诊断。
4. 本版本生成的新部署包在 MaixCAM-Pro / MaixCAM2 真机上的最终安装与推理。
5. Authenticode 代码签名。

因此，本报告结论是：**0.2.2 维护功能、当前主机安装包、真实项目兼容和已安装版短训练通过；未执行的外部环境门禁不虚报为通过。**
