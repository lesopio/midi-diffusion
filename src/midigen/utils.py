from __future__ import annotations

import http.server
import json
import os
import socketserver
import threading
import time

import torch
import torch.distributed as dist

SSE_CLIENTS = []
SSE_LOCK = threading.Lock()

HTML_DASHBOARD = r'''
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Training Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root{
      --fog-blue:#A9B9C6;
      --mint:#A8D5BA;
      --mint-dark:#7FB39A;
      --cherry:#D96C6C;
      --cherry-dark:#B55656;
      --ink:#2E3A43;
      --card:#E6ECEF;
      --grid:#C6D3DB;
    }
    body{
      font-family: "Segoe UI", Arial, sans-serif;
      margin:0;
      background:var(--fog-blue);
      color:var(--ink);
    }
    .wrap{max-width:1200px;margin:24px auto;padding:0 16px;}
    h2{margin:8px 0 6px 0;font-weight:600;letter-spacing:0.5px;}
    .sub{opacity:0.8;margin-bottom:16px;}
    .cards{
      display:grid;
      grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
      gap:12px;
      margin-bottom:16px;
    }
    .card{
      background:var(--card);
      border-radius:12px;
      padding:12px 14px;
      box-shadow:0 4px 14px rgba(34,45,54,0.08);
    }
    .label{font-size:12px;opacity:0.7;margin-bottom:6px;}
    .value{font-size:18px;font-weight:600;}
    .charts{
      display:grid;
      grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
      gap:16px;
    }
    .chart-card{
      background:var(--card);
      border-radius:14px;
      padding:12px;
      box-shadow:0 4px 14px rgba(34,45,54,0.08);
    }
    canvas{width:100%;height:240px;display:block;}
    .legend{display:flex;gap:12px;align-items:center;margin:6px 0 10px 2px;font-size:12px;}
    .dot{width:10px;height:10px;border-radius:50%;}
    .dot.mint{background:var(--mint-dark);}
    .dot.cherry{background:var(--cherry-dark);}
    .info{
      margin-top:16px;
      background:rgba(255,255,255,0.35);
      border-radius:12px;
      padding:12px 14px;
      font-size:13px;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h2>Training Dashboard</h2>
    <div class="sub">Status + losses (Morandi palette)</div>
    <div class="cards">
      <div class="card"><div class="label">Epoch</div><div class="value" id="epoch">1</div></div>
      <div class="card"><div class="label">Step</div><div class="value" id="step">0</div></div>
      <div class="card"><div class="label">Elapsed</div><div class="value" id="elapsed">0s</div></div>
      <div class="card"><div class="label">Remaining</div><div class="value" id="remaining">-</div></div>
      <div class="card"><div class="label">Loss</div><div class="value" id="lossText">-</div></div>
      <div class="card"><div class="label">Diffusion</div><div class="value" id="diffText">-</div></div>
      <div class="card"><div class="label">LR</div><div class="value" id="lrText">-</div></div>
      <div class="card"><div class="label">Throughput</div><div class="value" id="throughputText">-</div></div>
      <div class="card"><div class="label">VRAM</div><div class="value" id="vramText">-</div></div>
    </div>
    <div class="charts">
      <div class="chart-card">
        <div class="legend"><span class="dot mint"></span> Total Loss</div>
        <canvas id="lossChart"></canvas>
      </div>
      <div class="chart-card">
        <div class="legend"><span class="dot cherry"></span> Diffusion Loss</div>
        <canvas id="diffChart"></canvas>
      </div>
      <div class="chart-card">
        <div class="legend"><span class="dot mint"></span> Learning Rate</div>
        <canvas id="lrChart"></canvas>
      </div>
      <div class="chart-card">
        <div class="legend"><span class="dot mint"></span> Throughput (samples/s)</div>
        <canvas id="throughputChart"></canvas>
      </div>
      <div class="chart-card">
        <div class="legend"><span class="dot cherry"></span> VRAM (GB)</div>
        <canvas id="vramChart"></canvas>
      </div>
    </div>
    <div class="info" id="infoText">Waiting for metrics...</div>
  </div>
  <script>
    const lossCanvas = document.getElementById('lossChart');
    const diffCanvas = document.getElementById('diffChart');
    const lrCanvas = document.getElementById('lrChart');
    const throughputCanvas = document.getElementById('throughputChart');
    const vramCanvas = document.getElementById('vramChart');

    const state = {
      loss: [],
      diff: [],
      lr: [],
      throughput: [],
      vram: [],
      labels: [],
      maxPoints: 200,
    };

    function resizeCanvas(canvas) {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return ctx;
    }

    function drawChart(canvas, data, color) {
      const ctx = resizeCanvas(canvas);
      const w = canvas.width / (window.devicePixelRatio || 1);
      const h = canvas.height / (window.devicePixelRatio || 1);
      ctx.clearRect(0, 0, w, h);

      ctx.fillStyle = '#EAF0F3';
      ctx.fillRect(0, 0, w, h);

      ctx.strokeStyle = '#C6D3DB';
      ctx.lineWidth = 1;
      const gridLines = 4;
      for (let i = 1; i <= gridLines; i++) {
        const y = (h / (gridLines + 1)) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      if (data.length < 2) {
        return;
      }

      let min = Math.min(...data);
      let max = Math.max(...data);
      if (!isFinite(min) || !isFinite(max)) return;
      if (min === max) {
        min -= 0.5;
        max += 0.5;
      }
      const pad = (max - min) * 0.1;
      min -= pad;
      max += pad;

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      data.forEach((v, i) => {
        const x = (i / (data.length - 1)) * (w - 16) + 8;
        const y = h - ((v - min) / (max - min)) * (h - 16) - 8;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    function pushData(arr, v) {
      arr.push(v);
      if (arr.length > state.maxPoints) arr.shift();
    }

    function redraw() {
      drawChart(lossCanvas, state.loss, '#7FB39A');
      drawChart(diffCanvas, state.diff, '#B55656');
      drawChart(lrCanvas, state.lr, '#7FB39A');
      drawChart(throughputCanvas, state.throughput, '#7FB39A');
      drawChart(vramCanvas, state.vram, '#B55656');
    }

    window.addEventListener('resize', () => redraw());

    const es = new EventSource('/events');
    es.onmessage = (e) => {
      const m = JSON.parse(e.data);
      document.getElementById('elapsed').innerText = m.elapsed_str || m.elapsed || '-';
      document.getElementById('remaining').innerText = m.remaining_str || m.remaining || '-';
      document.getElementById('epoch').innerText = m.epoch || 1;
      document.getElementById('step').innerText = m.step || 0;

      if (typeof m.loss === 'number') {
        document.getElementById('lossText').innerText = m.loss.toFixed(4);
        pushData(state.loss, m.loss);
      }
      if (typeof m.diffusion_loss === 'number') {
        document.getElementById('diffText').innerText = m.diffusion_loss.toFixed(4);
        pushData(state.diff, m.diffusion_loss);
      }
      if (typeof m.lr === 'number') {
        document.getElementById('lrText').innerText = m.lr.toExponential(3);
        pushData(state.lr, m.lr);
      }
      if (typeof m.throughput === 'number') {
        document.getElementById('throughputText').innerText = m.throughput.toFixed(1);
        pushData(state.throughput, m.throughput);
      }
      if (typeof m.vram_gb === 'number') {
        document.getElementById('vramText').innerText = m.vram_gb.toFixed(2);
        pushData(state.vram, m.vram_gb);
      }

      document.getElementById('infoText').innerText =
        `epoch=${m.epoch || 1}, step=${m.step || 0}, loss=${(m.loss || 0).toFixed ? m.loss.toFixed(4) : m.loss}, diff=${(m.diffusion_loss || 0).toFixed ? m.diffusion_loss.toFixed(4) : m.diffusion_loss}, lr=${m.lr || '-'}, thr=${m.throughput || '-'}, vram=${m.vram_gb || '-'}`;

      redraw();
    }
  </script>
</body>
</html>
'''


class _SSEHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_DASHBOARD.encode('utf-8'))
            return
        if self.path == '/events':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()
            with SSE_LOCK:
                SSE_CLIENTS.append(self.wfile)
            try:
                while True:
                    time.sleep(1)
            except Exception:
                pass
            finally:
                with SSE_LOCK:
                    if self.wfile in SSE_CLIENTS:
                        SSE_CLIENTS.remove(self.wfile)
            return
        self.send_response(404)
        self.end_headers()


def start_dashboard_server(host: str = '0.0.0.0', port: int = 8080) -> None:
    def _serve():
        class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
        server = ThreadingHTTPServer((host, port), _SSEHandler)
        try:
            server.serve_forever()
        except Exception:
            pass

    t = threading.Thread(target=_serve, daemon=True)
    t.start()


def broadcast_metrics(payload: dict) -> None:
    data = f"data: {json.dumps(payload)}\n\n".encode('utf-8')
    remove = []
    with SSE_LOCK:
        for w in list(SSE_CLIENTS):
            try:
                w.write(data)
                w.flush()
            except Exception:
                remove.append(w)
        for w in remove:
            if w in SSE_CLIENTS:
                SSE_CLIENTS.remove(w)


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return rank, world_size, local_rank, device, True

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = torch.cuda.device_count()
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return local_rank, world_size, local_rank, device, True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, 0, device, False
