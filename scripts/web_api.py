import asyncio
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
import zipfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple
import re
import sys
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("PYTHON", sys.executable)
FRONTEND_DIST = ROOT / "frontend" / "dist"
RES_DIR = ROOT / "res"

app = FastAPI()

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

PREPROCESS_RE = re.compile(
    r"Processing MIDI:\s*(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([0-9:]+)<([0-9:]+),\s*([\d.]+)\s*(?:file|it)s?/s\]",
    re.IGNORECASE,
)


def _parse_time_to_sec(value: str) -> Optional[float]:
    parts = value.strip().split(":")
    try:
        nums = [int(p) for p in parts]
    except Exception:
        return None
    if len(nums) == 2:
        return float(nums[0] * 60 + nums[1])
    if len(nums) == 3:
        return float(nums[0] * 3600 + nums[1] * 60 + nums[2])
    if len(nums) == 4:
        return float(nums[0] * 86400 + nums[1] * 3600 + nums[2] * 60 + nums[3])
    return None


@dataclass
class Job:
    job_id: str
    job_type: str
    cmd: list[str]
    status: str = "running"
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    progress: float = 0.0
    detail: str = ""
    log_seq: int = 0
    logs: Deque[Tuple[int, str]] = field(default_factory=lambda: deque(maxlen=5000))
    metric_seq: int = 0
    metrics: Deque[Tuple[int, dict]] = field(default_factory=lambda: deque(maxlen=2000))
    metrics_path: Optional[Path] = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    process: Optional[subprocess.Popen] = None

    def add_log(self, line: str) -> None:
        with self.lock:
            self.log_seq += 1
            self.logs.append((self.log_seq, line))

    def add_metric(self, payload: dict) -> None:
        with self.lock:
            self.metric_seq += 1
            self.metrics.append((self.metric_seq, payload))


jobs: Dict[str, Job] = {}
jobs_lock = threading.Lock()


def safe_extract(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = out_dir / member.filename
            if not str(member_path.resolve()).startswith(str(out_dir.resolve())):
                continue
            zf.extract(member, out_dir)


def _start_job(job_type: str, cmd: list[str], metrics_path: Optional[Path] = None) -> Job:
    job_id = uuid.uuid4().hex[:12]
    job = Job(job_id=job_id, job_type=job_type, cmd=cmd, metrics_path=metrics_path)
    with jobs_lock:
        jobs[job_id] = job
    thread = threading.Thread(target=_run_job, args=(job,), daemon=True)
    thread.start()
    if metrics_path:
        threading.Thread(target=_tail_metrics, args=(job,), daemon=True).start()
    return job


def _run_job(job: Job) -> None:
    epoch_re = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)")
    percent_re = re.compile(r"(\d+)%")
    total_epochs = None
    current_epoch = 0

    try:
        job.process = subprocess.Popen(
            job.cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        job.status = "failed"
        job.detail = f"启动失败: {exc}"
        job.finished_at = time.time()
        job.add_log(job.detail)
        return

    if job.process.stdout is None:
        job.status = "failed"
        job.detail = "无法捕获输出"
        job.finished_at = time.time()
        job.add_log(job.detail)
        return

    if psutil:
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass
        try:
            proc = psutil.Process(job.process.pid)
            proc.cpu_percent(interval=None)
        except Exception:
            pass

    for raw in job.process.stdout:
        line = raw.strip()
        if line:
            job.add_log(line)
        if job.job_type == "train" and line:
            match = epoch_re.search(line)
            percent_match = percent_re.search(line)
            if match:
                try:
                    current_epoch = int(match.group(1))
                    total_epochs = int(match.group(2))
                except Exception:
                    pass
            if total_epochs:
                percent = 0
                if percent_match:
                    try:
                        percent = int(percent_match.group(1))
                    except Exception:
                        percent = 0
                base = max(0, current_epoch - 1)
                progress = min(1.0, (base + percent / 100.0) / float(total_epochs))
                job.progress = progress
                job.status = f"轮次 {current_epoch}/{total_epochs}"
        if job.job_type == "preprocess" and line:
            pmatch = PREPROCESS_RE.search(line)
            if pmatch:
                percent = int(pmatch.group(1))
                count = int(pmatch.group(2))
                total = int(pmatch.group(3))
                elapsed = _parse_time_to_sec(pmatch.group(4))
                remaining = _parse_time_to_sec(pmatch.group(5))
                speed = float(pmatch.group(6))
                cpu = None
                if psutil and job.process and job.process.pid:
                    try:
                        cpu = psutil.cpu_percent(interval=None)
                    except Exception:
                        cpu = None
                payload = {
                    "preprocess_percent": percent,
                    "preprocess_count": count,
                    "preprocess_total": total,
                    "preprocess_speed": speed,
                    "preprocess_elapsed": elapsed,
                    "preprocess_remaining": remaining,
                    "cpu_percent": cpu,
                }
                job.add_metric(payload)

    job.process.wait()
    job.exit_code = job.process.returncode
    job.finished_at = time.time()
    if job.exit_code == 0:
        job.status = "finished"
    else:
        job.status = "failed"
    job.add_log(f"任务结束，退出码: {job.exit_code}")


def _tail_metrics(job: Job) -> None:
    if not job.metrics_path:
        return
    pos = 0
    while True:
        if job.metrics_path.exists():
            try:
                with open(job.metrics_path, "r", encoding="utf-8") as f:
                    f.seek(pos)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except Exception:
                            continue
                        job.add_metric(payload)
                    pos = f.tell()
            except Exception:
                pass
        if job.finished_at is not None and (job.process is None or job.process.poll() is not None):
            break
        time.sleep(0.5)


@app.post("/api/import")
async def import_zip(
    zip_file: Optional[UploadFile] = File(None),
    out_dir: str = Form("data/midi"),
    clear_dir: str = Form("false"),
):
    out_path = ROOT / out_dir
    clear = str(clear_dir).lower() in {"1", "true", "yes", "on"}
    if zip_file is None:
        if not out_path.exists():
            raise HTTPException(status_code=400, detail="目录不存在，请检查路径。")
        return {"message": f"已设置数据目录: {out_path}", "out_dir": str(out_path)}

    tmp_path = ROOT / "data" / f"_upload_{uuid.uuid4().hex}.zip"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "wb") as f:
        content = await zip_file.read()
        f.write(content)

    if clear and out_path.exists():
        shutil.rmtree(out_path)

    safe_extract(tmp_path, out_path)
    tmp_path.unlink(missing_ok=True)
    return {"message": f"已解压到: {out_path}", "out_dir": str(out_path)}


@app.post("/api/preprocess")
async def preprocess(payload: dict):
    cmd = [
        PYTHON,
        "scripts/prepare_midi_dataset.py",
        "--input-dir",
        payload.get("data_dir", "data/midi"),
        "--output-dir",
        payload.get("output_dir", "data/cleaned_dataset"),
        "--zip-path",
        payload.get("zip_path", "data/cleaned_dataset.zip"),
        "--min-note-length",
        str(payload.get("min_note_length", 0.05)),
        "--max-size-mb",
        str(payload.get("max_size_mb", 10)),
    ]
    workers = payload.get("workers", 0)
    try:
        workers = int(workers)
    except Exception:
        workers = 0
    if workers and workers > 0:
        cmd += ["--workers", str(workers)]
    if payload.get("keep_structure"):
        cmd.append("--keep-structure")
    if payload.get("skip_zip"):
        cmd.append("--skip-zip")
    job = _start_job("preprocess", cmd)
    return {"job_id": job.job_id}


@app.post("/api/train")
async def train(payload: dict):
    cmd = [PYTHON, "scripts/train_polyphonic.py", "--device", str(payload.get("device", "4090"))]
    if payload.get("use_zip") and payload.get("data_zip"):
        cmd += ["--data-zip", str(payload.get("data_zip")), "--extract-dir", str(payload.get("extract_dir", "data/packed_dataset"))]
    else:
        cmd += ["--data-dir", str(payload.get("data_dir", "data/midi"))]
    if payload.get("save_dir"):
        cmd += ["--save-dir", str(payload.get("save_dir"))]
    if payload.get("epochs", 0) > 0:
        cmd += ["--epochs", str(payload.get("epochs"))]
    if payload.get("batch_size", 0) > 0:
        cmd += ["--batch-size", str(payload.get("batch_size"))]
    if payload.get("steps_per_beat", 0) > 0:
        cmd += ["--steps-per-beat", str(payload.get("steps_per_beat"))]
    if payload.get("max_voices", 0) > 0:
        cmd += ["--max-voices", str(payload.get("max_voices"))]
    cmd += ["--aux-weight", str(payload.get("aux_weight", 0.2))]
    if payload.get("dashboard"):
        cmd.append("--dashboard")
    if payload.get("resume"):
        cmd.append("--resume")
    if payload.get("save_every_steps", 0) > 0:
        cmd += ["--save-every-steps", str(payload.get("save_every_steps"))]
    if payload.get("dynamic_length"):
        cmd.append("--dynamic-length")
        if payload.get("max_seq_len", 0) > 0:
            cmd += ["--max-seq-len", str(payload.get("max_seq_len"))]
        cmd += ["--bucket-size", str(payload.get("bucket_size", 64))]
    cmd += ["--val-split", str(payload.get("val_split", 0.05)), "--val-every-epochs", str(payload.get("val_every_epochs", 1))]
    if payload.get("eval_data_dirs"):
        cmd += ["--eval-data-dirs", str(payload.get("eval_data_dirs"))]
    if payload.get("eval_max_samples", 0) > 0:
        cmd += ["--eval-max-samples", str(payload.get("eval_max_samples"))]

    metrics_dir = Path(payload.get("save_dir") or "checkpoints")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"metrics_live_{int(time.time())}.jsonl"
    cmd += ["--metrics-file", str(metrics_path)]

    job = _start_job("train", cmd, metrics_path=metrics_path)
    return {"job_id": job.job_id, "metrics_path": str(metrics_path)}


@app.post("/api/analysis")
async def analysis(payload: dict):
    cmd = [
        PYTHON,
        "scripts/analyze_dataset.py",
        "--data-dir",
        payload.get("data_dir", "data/midi"),
        "--steps-per-beat",
        str(payload.get("steps_per_beat", 4)),
        "--out",
        payload.get("out_path", "dataset_report.json"),
    ]
    if payload.get("compare_dir"):
        cmd += ["--compare-dir", payload.get("compare_dir")]
    if payload.get("max_seq_len", 0) > 0:
        cmd += ["--max-seq-len", str(payload.get("max_seq_len"))]
    if payload.get("sample_limit", 0) > 0:
        cmd += ["--sample-limit", str(payload.get("sample_limit"))]
    job = _start_job("analysis", cmd)
    return {"job_id": job.job_id}


@app.post("/api/generate")
async def generate(payload: dict):
    cmd = [
        PYTHON,
        "scripts/generate.py",
        "--checkpoint",
        payload.get("checkpoint", "checkpoints/checkpoint_latest.pt"),
        "--out-dir",
        payload.get("out_dir", "outputs"),
        "--num-samples",
        str(payload.get("num_samples", 1)),
        "--seq-len",
        str(payload.get("seq_len", 256)),
        "--steps",
        str(payload.get("steps", 50)),
        "--max-voices",
        str(payload.get("max_voices", 4)),
        "--steps-per-beat",
        str(payload.get("steps_per_beat", 4)),
        "--tempo",
        str(payload.get("tempo", 120.0)),
        "--temperature",
        str(payload.get("temperature", 1.0)),
    ]
    job = _start_job("generate", cmd)
    return {"job_id": job.job_id}


@app.post("/api/render_flac")
async def render_flac(payload: dict):
    cmd = [PYTHON, "scripts/render_flac.py"]
    midi_file = str(payload.get("midi_file", "")).strip()
    midi_dir = str(payload.get("midi_dir", "")).strip()
    if not midi_file and not midi_dir:
        raise HTTPException(status_code=400, detail="midi_file 或 midi_dir 不能为空")
    if midi_file:
        cmd += ["--midi-file", midi_file]
    if midi_dir:
        cmd += ["--midi-dir", midi_dir]
    cmd += ["--out-dir", str(payload.get("out_dir", "outputs_flac"))]
    soundfont = str(payload.get("soundfont", "")).strip()
    if soundfont:
        cmd += ["--soundfont", soundfont]
    if payload.get("sample_rate"):
        cmd += ["--sample-rate", str(payload.get("sample_rate"))]
    if payload.get("normalize"):
        cmd.append("--normalize")
    job = _start_job("render", cmd)
    return {"job_id": job.job_id}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    with job.lock:
        tail = [line for _, line in list(job.logs)[-200:]]
        payload = {
            "job_id": job.job_id,
            "type": job.job_type,
            "status": job.status,
            "progress": job.progress,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "exit_code": job.exit_code,
            "logs_tail": tail,
        }
    return JSONResponse(payload)


@app.get("/api/jobs/{job_id}/stream")
async def job_stream(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        last_seq = 0
        last_status = None
        while True:
            with job.lock:
                logs = list(job.logs)
                status_payload = {
                    "status": job.status,
                    "progress": job.progress,
                    "detail": job.detail,
                    "finished": job.finished_at is not None,
                    "exit_code": job.exit_code,
                }
                finished = job.finished_at is not None

            for seq, line in logs:
                if seq > last_seq:
                    last_seq = seq
                    yield {
                        "event": "log",
                        "data": json.dumps({"line": line}),
                    }

            if status_payload != last_status:
                last_status = status_payload
                yield {
                    "event": "status",
                    "data": json.dumps(status_payload),
                }

            if finished and last_seq >= job.log_seq:
                break

            await asyncio.sleep(0.3)

    return EventSourceResponse(event_generator())


@app.get("/api/jobs/{job_id}/metrics")
async def job_metrics(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def metric_generator():
        last_seq = 0
        while True:
            with job.lock:
                metrics = list(job.metrics)
                finished = job.finished_at is not None
            for seq, payload in metrics:
                if seq > last_seq:
                    last_seq = seq
                    yield {"event": "metrics", "data": json.dumps(payload)}
            if finished and last_seq >= job.metric_seq:
                break
            await asyncio.sleep(0.5)

    return EventSourceResponse(metric_generator())


@app.get("/api/system/stream")
async def system_stream():
    async def system_generator():
        while True:
            cpu = None
            if psutil:
                try:
                    cpu = psutil.cpu_percent(interval=None)
                except Exception:
                    cpu = None
            yield {"event": "system", "data": json.dumps({"cpu_percent": cpu})}
            await asyncio.sleep(0.1)

    return EventSourceResponse(system_generator())


@app.post("/api/jobs/{job_id}/stop")
async def stop_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.process and job.process.poll() is None:
        job.process.terminate()
        job.status = "stopped"
        job.finished_at = time.time()
        job.exit_code = -1
    return {"message": "stopped"}


if RES_DIR.exists():
    app.mount("/res", StaticFiles(directory=str(RES_DIR)), name="res")

if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app=app, host="0.0.0.0", port=8000, reload=False)