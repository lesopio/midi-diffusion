# 修改说明

## 2026-02-04

### 概览
- 项目聚焦为单一的多声部扩散训练主线。
- 训练/数据/扩散/模型模块集中在 `src/midigen`。
- 改进了 token 化、声部分配、扩散损失与辅助损失对齐方式。

### 破坏性变更
- Token 方案改为：`PAD=0`，`REST=1`，`PITCH=2..(PITCH_RANGE+1)`，旧 checkpoint 与缓存不再兼容。
- 训练入口变更为 `scripts/train_polyphonic.py`。

### 算法与训练改进
- 去除分布式训练中的重复数据切分，统一由 `DistributedSampler` 处理。
- 新增“声部连续性”分配策略，减少 voice 在时间轴上的跳变。
- 扩散损失与辅助损失均支持 PAD 忽略，并使用相同的噪声输入。
- 正式启用梯度累积（`grad_accum_steps`）。

### 结构调整
- 合并代码文件：核心逻辑收敛为 `data.py`、`model.py`、`diffusion.py`、`utils.py`。
- 新增 `scripts/train_polyphonic.py` 与 `scripts/generate.py`。
- 旧版与重复脚本移动到 `legacy/` 目录以保留历史参考。
- 预处理与下载脚本移动到 `tools/` 目录。

### 测试
- 新增基础测试：token 化、声部分配与扩散形状检查。

### 备注
- 由于环境策略限制，旧文件未删除而是移动到 `legacy/`。
