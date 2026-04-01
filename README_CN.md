# 项目使用说明（中文）

## 简介
本项目为多声部 MIDI 生成训练管线，采用离散扩散建模，支持多声部 token 化、声部连续性分配与可视化训练面板。

## 目录结构
- `src/midigen/`：核心代码
  - `config.py`：训练配置
  - `data.py`：MIDI 处理、token 化、数据集
  - `model.py`：多声部 Transformer 模型
  - `diffusion.py`：离散扩散过程
  - `utils.py`：分布式/面板工具
  - `train.py`：训练入口（被脚本调用）
  - `sample.py`：采样与导出
- `scripts/`：运行脚本
  - `train_polyphonic.py`：训练
  - `generate.py`：生成
  - `prepare_midi_dataset.py`：数据清洗与打包
- `frontend/`：React 前端（Vite + TS）
- `data/midi/`：MIDI 数据集目录
- `checkpoints/`：训练输出

## 依赖
建议使用 Python 3.10+。

核心依赖：
- `torch`
- `pretty_midi`
- `mido`
- `numpy`
- `tqdm`
- `psutil`（用于清洗 CPU 曲线，可选）

## 数据准备
把 MIDI 文件放到：
```
data/midi/
```
支持子目录递归扫描。

可选：先做清洗并打包（建议推到服务器前使用）
```
python scripts/prepare_midi_dataset.py --input-dir data/midi --output-dir data/cleaned_dataset --zip-path data/cleaned_dataset.zip
```
可选参数：
- `--workers`：清洗并行度（默认 CPU-1，传 1 可回退串行）
输出：
- `data/cleaned_dataset/midi/`：清洗后的 MIDI
- `data/cleaned_dataset/manifest.json`：逐文件处理信息
- `data/cleaned_dataset/stats.json`：统计
- `data/cleaned_dataset/prepare.log`：日志
- `data/cleaned_dataset.zip`：打包产物

## 训练
单卡训练：
```
python scripts/train_polyphonic.py --device 4090 --dashboard
```

多卡训练：
```
torchrun --nproc_per_node=4 scripts/train_polyphonic.py --device 4090 --dashboard
```

从清洗后的 zip 直接训练（自动解包）：
```
python scripts/train_polyphonic.py --data-zip data/cleaned_dataset.zip --extract-dir data/packed_dataset
```

常用参数：
- `--data-dir`：数据路径（默认 `data/midi`）
- `--save-dir`：输出路径（默认 `checkpoints`）
- `--epochs`：训练轮数
- `--batch-size`：覆盖配置
- `--steps-per-beat`：节拍网格密度
- `--max-voices`：最大声部数
- `--aux-weight`：辅助损失权重
- `--dashboard`：启用面板
- `--resume`：从最新 checkpoint 恢复
- `--dynamic-length`：启用“单 MIDI 为单位”的动态长度训练
- `--max-seq-len`：动态长度上限（不填则自动推荐）
- `--bucket-size`：长度分桶粒度（默认 64）
- `--val-split`：验证集比例（默认 0.05）
- `--val-every-epochs`：每隔多少 epoch 跑一次验证（默认 1）
- `--eval-data-dirs`：额外评估数据集（逗号分隔）
- `--eval-max-samples`：每次评估最多跑多少 batch（0 表示全量）

说明：
- 启用 `--dynamic-length` 且未指定 `--max-seq-len` 时，会根据数据长度自动推荐一个更高效的上限
- 长度统计会缓存到 `checkpoints/lengths_cache_*.json` 以加速后续训练

## 数据分布分析（过拟合/偏置排查）
```
python scripts/analyze_dataset.py --data-dir data/midi --steps-per-beat 4 --out dataset_report.json
```
对比两个数据集分布：
```
python scripts/analyze_dataset.py --data-dir data/midi --compare-dir data/other_midi --out compare_report.json
```

## React 前端（工业夜间风格）
依赖：
- Node.js 18+
- Python：fastapi / uvicorn / sse-starlette / python-multipart

首次构建前端：
```
cd frontend
npm install
npm run build
```

启动（推荐）：
```
python scripts/start_app.py
```
选择 `1` 启动 React 前端（默认端口 8000）。

开发模式（前后端分离）：
```
# 终端 1
python scripts/web_api.py

# 终端 2
cd frontend
npm run dev
```

说明：
- 前端包含：数据导入、清洗、训练、分析、生成
- 训练指标曲线与终端日志均内置

## 生成
```
python scripts/generate.py --checkpoint checkpoints/checkpoint_latest.pt --num-samples 2 --seq-len 256
```

常用参数：
- `--steps`：扩散采样步数
- `--max-voices`：声部数
- `--steps-per-beat`：网格密度
- `--temperature`：采样温度
- `--tempo`：输出 MIDI 的速度

生成输出：
- `outputs/sample_*.mid`
- `outputs/sample_*.pt`

提示：`--steps` 不能超过模型训练时的 `diffusion_steps`。

## 训练监控
使用 `--dashboard` 后，浏览器打开：
```
http://localhost:8080
```

## 重要提示
- Token 方案已变更：`PAD=0`，`REST=1`，`PITCH=2..(PITCH_RANGE+1)`。
- 旧 checkpoint 与缓存不兼容，请重新训练。

## 常见问题
- `pytest` 未安装：可先忽略测试，或自行 `pip install pytest` 后运行。
- 采样速度慢：降低 `--steps` 或缩短 `--seq-len`。
