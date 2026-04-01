import json
import re
import shutil
import subprocess
import time
import zipfile
from collections import deque
from pathlib import Path
import sys
from typing import Iterator, Optional

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

THEME = gr.themes.Soft()
CSS = """
:root{
  --bg:#FFFFFF;
  --card:#F5F6F7;
  --ink:#2E3A43;
  --border:#E2E5E8;
}
body, .gradio-container{
  background: var(--bg) !important;
  color: var(--ink);
}
.gradio-container .block, .gradio-container .form, .gradio-container .panel, .gradio-container .gr-box{
  background: var(--card) !important;
  border-color: var(--border) !important;
}
"""

GLOBAL_TERMINAL_LOG = deque(maxlen=5000)


def _push_global_log(tag: str, line: str) -> None:
    if not line:
        return
    ts = time.strftime("%H:%M:%S")
    prefix = f"[{ts}][{tag}] "
    for raw in str(line).splitlines():
        if raw.strip() == "":
            continue
        GLOBAL_TERMINAL_LOG.append(prefix + raw)


def _get_global_terminal() -> str:
    return "\n".join(GLOBAL_TERMINAL_LOG) if GLOBAL_TERMINAL_LOG else "暂无日志"


def _safe_extract(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = out_dir / member.filename
            if not str(member_path.resolve()).startswith(str(out_dir.resolve())):
                continue
            zf.extract(member, out_dir)


def _run_command(cmd: list[str], cwd: Path, tag: Optional[str] = None) -> Iterator[str]:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    logs = deque(maxlen=2000)
    if proc.stdout is None:
        message = "无法捕获输出"
        if tag:
            _push_global_log(tag, message)
        yield message
        return
    for line in proc.stdout:
        line = line.rstrip()
        logs.append(line)
        if tag:
            _push_global_log(tag, line)
        yield "\n".join(logs)
    proc.wait()
    return


def _stream_with_terminal(log_iter: Iterator[str]) -> Iterator[tuple[str, str]]:
    for log in log_iter:
        yield log, _get_global_terminal()


def _sparkline(values: list[float], color: str, width: int = 320, height: int = 120) -> str:
    if not values:
        return '<div style="height:120px;display:flex;align-items:center;justify-content:center;color:#6b7a84;">no data</div>'
    vmin = min(values)
    vmax = max(values)
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5
    span = vmax - vmin if vmax != vmin else 1.0
    pts = []
    for i, v in enumerate(values):
        x = (i / max(1, len(values) - 1)) * (width - 16) + 8
        y = height - ((v - vmin) / span) * (height - 16) - 8
        pts.append(f"{x:.1f},{y:.1f}")
    points = " ".join(pts)
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'style="background:#EAF0F3;border-radius:0;">'
        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"></polyline>'
        '</svg>'
    )


def _render_metrics_html(metrics: dict[str, list[float]]) -> str:
    def latest(key: str) -> str:
        arr = metrics.get(key, [])
        if not arr:
            return "-"
        val = arr[-1]
        if key in ("lr",):
            return f"{val:.2e}"
        if key in ("vram",):
            return f"{val:.2f}"
        if key in ("throughput",):
            return f"{val:.1f}"
        return f"{val:.4f}"

    def card(title: str, key: str, color: str) -> str:
        return (
            '<div style="background:#E6ECEF;border-radius:0;padding:10px 12px;">'
            f'<div style="font-size:12px;color:#4b5a64;margin-bottom:4px;">{title} '
            f'<span style="float:right;font-weight:600;">{latest(key)}</span></div>'
            f'{_sparkline(metrics.get(key, []), color)}'
            '</div>'
        )

    return (
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;">'
        f'{card("损失", "loss", "#7FB39A")}'
        f'{card("扩散损失", "diffusion", "#B55656")}'
        f'{card("学习率", "lr", "#7FB39A")}'
        f'{card("吞吐量", "throughput", "#7FB39A")}'
        f'{card("显存(GB)", "vram", "#B55656")}'
        '</div>'
    )


def import_zip(zip_file: Optional[str], out_dir: str, clear_dir: bool) -> tuple[str, str, str]:
    if not zip_file:
        message = "请先上传 zip 文件。"
        _push_global_log("导入", message)
        return message, out_dir, _get_global_terminal()
    out_path = Path(out_dir)
    if clear_dir and out_path.exists():
        shutil.rmtree(out_path)
    _safe_extract(Path(zip_file), out_path)
    message = f"已解压到: {out_path}"
    _push_global_log("导入", message)
    return message, str(out_path), _get_global_terminal()


def use_existing_dir(data_dir: str) -> tuple[str, str, str]:
    path = Path(data_dir)
    if not path.exists():
        message = "目录不存在，请检查路径。"
        _push_global_log("导入", message)
        return message, data_dir, _get_global_terminal()
    message = f"已设置数据目录: {path}"
    _push_global_log("导入", message)
    return message, str(path), _get_global_terminal()


def run_preprocess(
    data_dir: str,
    output_dir: str,
    zip_path: str,
    min_note_length: float,
    max_size_mb: int,
    keep_structure: bool,
    skip_zip: bool,
) -> Iterator[tuple[str, str]]:
    cmd = [
        PYTHON,
        "scripts/prepare_midi_dataset.py",
        "--input-dir",
        data_dir,
        "--output-dir",
        output_dir,
        "--zip-path",
        zip_path,
        "--min-note-length",
        str(min_note_length),
        "--max-size-mb",
        str(max_size_mb),
    ]
    if keep_structure:
        cmd.append("--keep-structure")
    if skip_zip:
        cmd.append("--skip-zip")
    return _stream_with_terminal(_run_command(cmd, ROOT, tag="清洗"))


def run_train(
    data_dir: str,
    data_zip: str,
    use_zip: bool,
    extract_dir: str,
    save_dir: str,
    device: str,
    epochs: int,
    batch_size: int,
    steps_per_beat: int,
    max_voices: int,
    aux_weight: float,
    dashboard: bool,
    resume: bool,
    save_every_steps: int,
    dynamic_length: bool,
    max_seq_len: int,
    bucket_size: int,
    val_split: float,
    val_every_epochs: int,
    eval_data_dirs: str,
    eval_max_samples: int,
) -> Iterator[tuple[str, str, str, str, str]]:
    cmd = [PYTHON, "scripts/train_polyphonic.py", "--device", device]
    if use_zip and data_zip:
        cmd += ["--data-zip", data_zip, "--extract-dir", extract_dir]
    else:
        cmd += ["--data-dir", data_dir]
    if save_dir:
        cmd += ["--save-dir", save_dir]
    if epochs > 0:
        cmd += ["--epochs", str(epochs)]
    if batch_size > 0:
        cmd += ["--batch-size", str(batch_size)]
    if steps_per_beat > 0:
        cmd += ["--steps-per-beat", str(steps_per_beat)]
    if max_voices > 0:
        cmd += ["--max-voices", str(max_voices)]
    cmd += ["--aux-weight", str(aux_weight)]
    if dashboard:
        cmd.append("--dashboard")
    if resume:
        cmd.append("--resume")
    if save_every_steps > 0:
        cmd += ["--save-every-steps", str(save_every_steps)]
    if dynamic_length:
        cmd.append("--dynamic-length")
        if max_seq_len > 0:
            cmd += ["--max-seq-len", str(max_seq_len)]
        cmd += ["--bucket-size", str(bucket_size)]
    cmd += ["--val-split", str(val_split), "--val-every-epochs", str(val_every_epochs)]
    if eval_data_dirs:
        cmd += ["--eval-data-dirs", eval_data_dirs]
    if eval_max_samples > 0:
        cmd += ["--eval-max-samples", str(eval_max_samples)]

    metrics_dir = Path(save_dir or "checkpoints")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"metrics_live_{int(time.time())}.jsonl"
    cmd += ["--metrics-file", str(metrics_path)]

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    logs = deque(maxlen=3000)
    total_epochs = epochs if epochs > 0 else None
    current_epoch = 0
    progress_html = (
        '<div style="height:10px;background:#D5DEE4;border-radius:0;overflow:hidden;">'
        '<div style="width:0%;height:100%;background:#7FB39A;"></div></div>'
    )
    status = "等待日志输出..."
    metrics = {"loss": [], "diffusion": [], "lr": [], "throughput": [], "vram": []}
    metrics_html = _render_metrics_html(metrics)
    metrics_pos = 0
    epoch_re = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)")
    percent_re = re.compile(r"(\d+)%")

    if proc.stdout is None:
        _push_global_log("训练", "无法捕获日志输出")
        yield "无法捕获日志输出", progress_html, "训练启动失败", metrics_html, _get_global_terminal()
        return

    def _push(key: str, val: float, maxlen: int = 200):
        arr = metrics.get(key, [])
        arr.append(val)
        if len(arr) > maxlen:
            arr.pop(0)
        metrics[key] = arr

    def _read_metrics():
        nonlocal metrics_pos, metrics_html
        if not metrics_path.exists():
            return
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                f.seek(metrics_pos)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    if "loss" in payload:
                        _push("loss", float(payload["loss"]))
                    if "diffusion_loss" in payload:
                        _push("diffusion", float(payload["diffusion_loss"]))
                    if "lr" in payload:
                        _push("lr", float(payload["lr"]))
                    if "throughput" in payload:
                        _push("throughput", float(payload["throughput"]))
                    if "vram_gb" in payload:
                        _push("vram", float(payload["vram_gb"]))
                metrics_pos = f.tell()
            metrics_html = _render_metrics_html(metrics)
        except Exception:
            return

    for raw in proc.stdout:
        line = raw.replace("\r", "").strip()
        if line:
            logs.append(line)
            _push_global_log("训练", line)
        match = epoch_re.search(line)
        percent_match = percent_re.search(line)
        if match:
            try:
                current_epoch = int(match.group(1))
                if total_epochs is None:
                    total_epochs = int(match.group(2))
            except Exception:
                pass
        percent = None
        if percent_match:
            try:
                percent = int(percent_match.group(1))
            except Exception:
                percent = None
        if total_epochs:
            base = max(0, current_epoch - 1)
            frac = (percent or 0) / 100.0
            progress = min(1.0, (base + frac) / float(total_epochs))
            progress_width = f"{progress * 100:.1f}%"
            progress_html = (
                '<div style="height:10px;background:#D5DEE4;border-radius:0;overflow:hidden;">'
                f'<div style="width:{progress_width};height:100%;background:#7FB39A;"></div></div>'
            )
            status = f"轮次 {current_epoch}/{total_epochs} | {progress_width}"
        _read_metrics()
        yield "\n".join(logs), progress_html, status, metrics_html, _get_global_terminal()
    proc.wait()
    if total_epochs:
        progress_html = (
            '<div style="height:10px;background:#D5DEE4;border-radius:0;overflow:hidden;">'
            '<div style="width:100%;height:100%;background:#7FB39A;"></div></div>'
        )
        status = "训练结束"
        _push_global_log("训练", status)
    _read_metrics()
    yield "\n".join(logs), progress_html, status, metrics_html, _get_global_terminal()


def run_analysis(
    data_dir: str,
    compare_dir: str,
    steps_per_beat: int,
    max_seq_len: int,
    sample_limit: int,
    out_path: str,
) -> Iterator[tuple[str, str]]:
    cmd = [
        PYTHON,
        "scripts/analyze_dataset.py",
        "--data-dir",
        data_dir,
        "--steps-per-beat",
        str(steps_per_beat),
        "--out",
        out_path,
    ]
    if compare_dir:
        cmd += ["--compare-dir", compare_dir]
    if max_seq_len > 0:
        cmd += ["--max-seq-len", str(max_seq_len)]
    if sample_limit > 0:
        cmd += ["--sample-limit", str(sample_limit)]
    return _stream_with_terminal(_run_command(cmd, ROOT, tag="分析"))


def run_generate(
    checkpoint: str,
    out_dir: str,
    num_samples: int,
    seq_len: int,
    steps: int,
    max_voices: int,
    steps_per_beat: int,
    tempo: float,
    temperature: float,
) -> Iterator[tuple[str, str]]:
    cmd = [
        PYTHON,
        "scripts/generate.py",
        "--checkpoint",
        checkpoint,
        "--out-dir",
        out_dir,
        "--num-samples",
        str(num_samples),
        "--seq-len",
        str(seq_len),
        "--steps",
        str(steps),
        "--max-voices",
        str(max_voices),
        "--steps-per-beat",
        str(steps_per_beat),
        "--tempo",
        str(tempo),
        "--temperature",
        str(temperature),
    ]
    return _stream_with_terminal(_run_command(cmd, ROOT, tag="生成"))
