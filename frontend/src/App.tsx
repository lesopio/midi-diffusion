

import { useEffect, useMemo, useRef, useState } from "react";
import classNames from "classnames";
import MetricChart from "./components/MetricChart";
import {
  importZip,
  startAnalysis,
  startGenerate,
  startPreprocess,
  startRenderFlac,
  startTrain,
  streamLogs,
  streamMetrics,
  streamSystem,
  JobStatusPayload,
  MetricsPayload,
} from "./services/api";

const DEFAULT_MAX_LOG_LINES = 2000;

type PageKey =
  | "import"
  | "preprocess"
  | "train"
  | "analysis"
  | "generate"
  | "arranger"
  | "render"
  | "charts"
  | "terminal"
  | "settings";

type LogKey =
  | "global"
  | "import"
  | "preprocess"
  | "train"
  | "analysis"
  | "generate"
  | "render";

type UiSettings = {
  defaultSoundfont: string;
  maxLogLines: number;
  autoScroll: boolean;
  hudEnabled: boolean;
};

type ArrangerTrack = {
  id: string;
  name: string;
  color: string;
  midi?: string;
};

type ArrangerClip = {
  id: string;
  trackId: string;
  name: string;
  start: number;
  length: number;
  color: string;
};

type MixerChannel = {
  id: string;
  name: string;
  volume: number;
  pan: number;
  mute: boolean;
  solo: boolean;
};

const defaultLogs: Record<LogKey, string[]> = {
  global: [],
  import: [],
  preprocess: [],
  train: [],
  analysis: [],
  generate: [],
  render: [],
};

const PREPROCESS_PROGRESS_RE =
  /(?:\[preprocess\]\s*)?Processing MIDI:\s*(\d+)%\|.*?\|\s*(\d+)\/(\d+)\s*\[([0-9:]+)<([0-9:]+),\s*([\d.]+)\s*(?:file|it)s?\/s\]/i;
const PREPROCESS_PROGRESS_TOKEN = "Processing MIDI:";
const GENERATE_OUTPUT_RE = /\[generate\]\s*saved midi:\s*(.+\.mid)\s*$/i;
const ANSI_RE = /\x1b\[[0-9;]*m/g;
const UI_SETTINGS_KEY = "midi_ui_settings";

const arrangerColors = ["#dfff00", "#20f8b3", "#f31be5", "#cbf716", "#00fff0", "#ff8a00"];

const parseDuration = (value: string | undefined) => {
  if (!value) return null;
  const parts = value.split(":").map((v) => Number(v));
  if (parts.some((v) => Number.isNaN(v))) return null;
  if (parts.length === 2) return parts[0] * 60 + parts[1];
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  if (parts.length === 4) return parts[0] * 86400 + parts[1] * 3600 + parts[2] * 60 + parts[3];
  return null;
};

const loadSettings = (): UiSettings => {
  try {
    const raw = localStorage.getItem(UI_SETTINGS_KEY);
    if (!raw) {
      return {
        defaultSoundfont: "",
        maxLogLines: DEFAULT_MAX_LOG_LINES,
        autoScroll: true,
        hudEnabled: true,
      };
    }
    const parsed = JSON.parse(raw) as Partial<UiSettings>;
    return {
      defaultSoundfont: parsed.defaultSoundfont || "",
      maxLogLines: Number(parsed.maxLogLines) || DEFAULT_MAX_LOG_LINES,
      autoScroll: parsed.autoScroll !== false,
      hudEnabled: parsed.hudEnabled !== false,
    };
  } catch {
    return {
      defaultSoundfont: "",
      maxLogLines: DEFAULT_MAX_LOG_LINES,
      autoScroll: true,
      hudEnabled: true,
    };
  }
};

function useTheme() {
  const [theme, setTheme] = useState<string>(() => {
    const saved = localStorage.getItem("midi_theme");
    return saved === "light" || saved === "dark" ? saved : "dark";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("midi_theme", theme);
  }, [theme]);

  return { theme, setTheme };
}

const makeId = () => `id_${Math.random().toString(36).slice(2, 10)}`;

export default function App() {
  const [page, setPage] = useState<PageKey>("import");
  const [terminalTab, setTerminalTab] = useState<LogKey>("global");
  const { theme, setTheme } = useTheme();

  const [settings, setSettings] = useState<UiSettings>(loadSettings);
  const [cpuPercent, setCpuPercent] = useState<number | null>(null);
  const [logs, setLogs] = useState<Record<LogKey, string[]>>(defaultLogs);
  const logRef = useRef<HTMLDivElement | null>(null);
  const logStreams = useRef<Partial<Record<LogKey, () => void>>>({});
  const metricStream = useRef<null | (() => void)>(null);
  const preMetricStream = useRef<null | (() => void)>(null);

  const [importForm, setImportForm] = useState({
    outDir: "data/midi",
    clearDir: false,
    zipFile: null as File | null,
  });
  const [importStatus, setImportStatus] = useState("");

  const [preForm, setPreForm] = useState({
    dataDir: "data/midi",
    outputDir: "data/cleaned_dataset",
    zipPath: "data/cleaned_dataset.zip",
    minNoteLength: 0.05,
    maxSizeMb: 10,
    workers: 0,
    keepStructure: false,
    skipZip: false,
  });
  const [preStatus, setPreStatus] = useState("");

  const [trainForm, setTrainForm] = useState({
    dataDir: "data/midi",
    useZip: false,
    dataZip: "data/cleaned_dataset.zip",
    extractDir: "data/packed_dataset",
    saveDir: "checkpoints",
    device: "4090",
    epochs: 50,
    batchSize: 0,
    stepsPerBeat: 0,
    maxVoices: 0,
    auxWeight: 0.2,
    saveEverySteps: 500,
    resume: false,
    dashboard: true,
    dynamicLength: true,
    maxSeqLen: 0,
    bucketSize: 64,
    valSplit: 0.05,
    valEveryEpochs: 1,
    evalDataDirs: "",
    evalMaxSamples: 0,
  });
  const [trainStatus, setTrainStatus] = useState("未开始");
  const [trainProgress, setTrainProgress] = useState(0);

  const [analysisForm, setAnalysisForm] = useState({
    dataDir: "data/midi",
    compareDir: "",
    stepsPerBeat: 4,
    maxSeqLen: 0,
    sampleLimit: 0,
    outPath: "dataset_report.json",
  });
  const [analysisStatus, setAnalysisStatus] = useState("");

  const [generateForm, setGenerateForm] = useState({
    checkpoint: "checkpoints/checkpoint_latest.pt",
    outDir: "outputs",
    numSamples: 1,
    seqLen: 256,
    steps: 50,
    maxVoices: 4,
    stepsPerBeat: 4,
    tempo: 120,
    temperature: 1,
  });
  const [generateStatus, setGenerateStatus] = useState("");
  const [generateOutputs, setGenerateOutputs] = useState<string[]>([]);

  const [renderForm, setRenderForm] = useState({
    mode: "file" as "file" | "dir",
    midiPath: "",
    outDir: "outputs_flac",
    soundfont: "",
    sampleRate: 44100,
    normalize: true,
  });
  const [renderStatus, setRenderStatus] = useState("");
  
  const [metrics, setMetrics] = useState({
    loss: [] as number[],
    diffusion: [] as number[],
    lr: [] as number[],
    throughput: [] as number[],
    vram: [] as number[],
  });
  const [preprocessMetrics, setPreprocessMetrics] = useState({
    count: [] as number[],
    speed: [] as number[],
    elapsed: [] as number[],
    remaining: [] as number[],
    cpu: [] as number[],
  });

  const [arrangerName, setArrangerName] = useState("国风流行混音工程");
  const [arrangerBars, setArrangerBars] = useState(64);
  const [arrangerBpm, setArrangerBpm] = useState(124);
  const [arrangerTimeSig, setArrangerTimeSig] = useState("4/4");
  const [arrangerTracks, setArrangerTracks] = useState<ArrangerTrack[]>([
    { id: "trk_drums", name: "Drums", color: arrangerColors[0] },
    { id: "trk_bass", name: "Bass", color: arrangerColors[1] },
    { id: "trk_chords", name: "Chords", color: arrangerColors[2] },
    { id: "trk_lead", name: "Lead", color: arrangerColors[3] },
  ]);
  const [arrangerClips, setArrangerClips] = useState<ArrangerClip[]>([
    { id: "clip_intro", trackId: "trk_chords", name: "Intro Pad", start: 1, length: 8, color: arrangerColors[2] },
    { id: "clip_groove", trackId: "trk_drums", name: "Verse Groove", start: 9, length: 16, color: arrangerColors[0] },
    { id: "clip_bass", trackId: "trk_bass", name: "Bassline", start: 9, length: 16, color: arrangerColors[1] },
    { id: "clip_hook", trackId: "trk_lead", name: "Hook", start: 25, length: 8, color: arrangerColors[3] },
  ]);
  const [mixerChannels, setMixerChannels] = useState<MixerChannel[]>([
    { id: "trk_drums", name: "Drums", volume: 0.78, pan: 0, mute: false, solo: false },
    { id: "trk_bass", name: "Bass", volume: 0.7, pan: 0, mute: false, solo: false },
    { id: "trk_chords", name: "Chords", volume: 0.6, pan: -8, mute: false, solo: false },
    { id: "trk_lead", name: "Lead", volume: 0.64, pan: 8, mute: false, solo: false },
  ]);
  const [clipDraft, setClipDraft] = useState({
    trackId: "trk_drums",
    name: "",
    start: 1,
    length: 4,
  });
  const [newTrackName, setNewTrackName] = useState("");

  useEffect(() => {
    document.documentElement.setAttribute("data-hud", settings.hudEnabled ? "on" : "off");
  }, [settings.hudEnabled]);

  useEffect(() => {
    const stop = streamSystem((payload) => {
      if (typeof payload.cpu_percent === "number") {
        const clamped = Math.max(0, Math.min(100, payload.cpu_percent));
        setCpuPercent(clamped);
      } else {
        setCpuPercent(null);
      }
    });
    return () => stop();
  }, []);

  useEffect(() => {
    localStorage.setItem(UI_SETTINGS_KEY, JSON.stringify(settings));
  }, [settings]);

  useEffect(() => {
    if (!renderForm.soundfont && settings.defaultSoundfont) {
      setRenderForm((prev) => ({ ...prev, soundfont: settings.defaultSoundfont }));
    }
  }, [settings.defaultSoundfont, renderForm.soundfont]);

  useEffect(() => {
    if (!settings.autoScroll || page !== "terminal") return;
    const el = logRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [logs, terminalTab, page, settings.autoScroll]);

    useEffect(() => {
    if (!arrangerTracks.find((track) => track.id === clipDraft.trackId)) {
      setClipDraft((prev) => ({ ...prev, trackId: arrangerTracks[0]?.id || "" }));
    }
  }, [arrangerTracks, clipDraft.trackId]);


    const appendLog = (key: LogKey, line: string) => {
    const cleaned = line.replace(ANSI_RE, "").replace(/\r/g, "");
    const isPreProgress =
      key === "preprocess" && cleaned.includes(PREPROCESS_PROGRESS_TOKEN);
    if (isPreProgress) {
      const match = cleaned.match(PREPROCESS_PROGRESS_RE);
      if (match) {
        const count = Number(match[2]);
        const speed = Number(match[6]);
        const elapsed = parseDuration(match[4]);
        const remaining = parseDuration(match[5]);
        if (Number.isFinite(count) || Number.isFinite(speed)) {
          setPreprocessMetrics((prev) => ({
            count: Number.isFinite(count) ? [...prev.count, count].slice(-400) : prev.count,
            speed: Number.isFinite(speed) ? [...prev.speed, speed].slice(-400) : prev.speed,
            elapsed: Number.isFinite(elapsed as number) ? [...prev.elapsed, elapsed as number].slice(-400) : prev.elapsed,
            remaining: Number.isFinite(remaining as number) ? [...prev.remaining, remaining as number].slice(-400) : prev.remaining,
            cpu: prev.cpu,
          }));
        }
      }
    }
    if (key === "generate") {
      const match = cleaned.match(GENERATE_OUTPUT_RE);
      if (match?.[1]) {
        setGenerateOutputs((prev) => {
          const next = [match[1], ...prev.filter((item) => item !== match[1])];
          return next.slice(0, 20);
        });
      }
    }
    setLogs((prev) => {
      const next = { ...prev };
      const cap = Math.max(200, settings.maxLogLines || DEFAULT_MAX_LOG_LINES);
      let updated = [...prev[key]];
      if (isPreProgress) {
        updated = updated.filter((item) => !item.includes(PREPROCESS_PROGRESS_TOKEN));
        updated.push(cleaned);
      } else {
        updated.push(cleaned);
      }
      next[key] = updated.slice(-cap);
      if (key !== "global") {
        let globalLines = [...prev.global];
        const globalLine = cleaned.startsWith("[") ? cleaned : `[${key}] ${cleaned}`;
        if (isPreProgress) {
          globalLines = globalLines.filter((item) => !item.includes(PREPROCESS_PROGRESS_TOKEN));
          globalLines.push(globalLine);
        } else {
          globalLines.push(globalLine);
        }
        next.global = globalLines.slice(-cap);
      }
      return next;
    });
  };

  const clearLogs = (key: LogKey) => {
    setLogs((prev) => ({ ...prev, [key]: [] }));
  };

  const handleStatus = (key: LogKey, payload: JobStatusPayload) => {
    if (key === "train") {
      if (typeof payload.progress === "number") {
        setTrainProgress(payload.progress);
      }
      if (payload.status) {
        setTrainStatus(payload.status);
      }
    }
    if (key === "preprocess") {
      setPreStatus(payload.status || "");
    }
    if (key === "analysis") {
      setAnalysisStatus(payload.status || "");
    }
    if (key === "generate") {
      setGenerateStatus(payload.status || "");
    }
    if (key === "render") {
      setRenderStatus(payload.status || "");
    }
  };

  const connectLogStream = (key: LogKey, jobId: string) => {
    logStreams.current[key]?.();
    logStreams.current[key] = streamLogs(
      jobId,
      (line) => appendLog(key, line),
      (payload) => handleStatus(key, payload)
    );
  };

  const pushMetric = (payload: MetricsPayload) => {
    setMetrics((prev) => {
      const next = { ...prev };
      if (typeof payload.loss === "number") {
        next.loss = [...prev.loss, payload.loss].slice(-400);
      }
      if (typeof payload.diffusion_loss === "number") {
        next.diffusion = [...prev.diffusion, payload.diffusion_loss].slice(-400);
      }
      if (typeof payload.lr === "number") {
        next.lr = [...prev.lr, payload.lr].slice(-400);
      }
      if (typeof payload.throughput === "number") {
        next.throughput = [...prev.throughput, payload.throughput].slice(-400);
      }
      if (typeof payload.vram_gb === "number") {
        next.vram = [...prev.vram, payload.vram_gb].slice(-400);
      }
      return next;
    });
  };

  const pushPreprocessMetric = (payload: MetricsPayload) => {
    setPreprocessMetrics((prev) => {
      const next = { ...prev };
      if (typeof payload.preprocess_count === "number") {
        next.count = [...prev.count, payload.preprocess_count].slice(-400);
      }
      if (typeof payload.preprocess_speed === "number") {
        next.speed = [...prev.speed, payload.preprocess_speed].slice(-400);
      }
      if (typeof payload.preprocess_elapsed === "number") {
        next.elapsed = [...prev.elapsed, payload.preprocess_elapsed].slice(-400);
      }
      if (typeof payload.preprocess_remaining === "number") {
        next.remaining = [...prev.remaining, payload.preprocess_remaining].slice(-400);
      }
      if (typeof payload.cpu_percent === "number") {
        next.cpu = [...prev.cpu, payload.cpu_percent].slice(-400);
      }
      return next;
    });
  };

  const startImport = async (useExisting: boolean) => {
    setImportStatus("处理中...");
    try {
      const res = await importZip({
        zipFile: useExisting ? null : importForm.zipFile,
        outDir: importForm.outDir,
        clearDir: importForm.clearDir,
      });
      setImportStatus(res.message);
      appendLog("import", res.message);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "导入失败";
      setImportStatus(msg);
      appendLog("import", msg);
    }
  };

  const startPre = async () => {
    clearLogs("preprocess");
    setPreStatus("已启动");
    setPreprocessMetrics({ count: [], speed: [], elapsed: [], remaining: [], cpu: [] });
    try {
      const res = await startPreprocess({
        data_dir: preForm.dataDir,
        output_dir: preForm.outputDir,
        zip_path: preForm.zipPath,
        min_note_length: preForm.minNoteLength,
        max_size_mb: preForm.maxSizeMb,
        workers: preForm.workers,
        keep_structure: preForm.keepStructure,
        skip_zip: preForm.skipZip,
      });
      connectLogStream("preprocess", res.job_id);
      preMetricStream.current?.();
      preMetricStream.current = streamMetrics(res.job_id, pushPreprocessMetric);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "清洗失败";
      setPreStatus(msg);
      appendLog("preprocess", msg);
    }
  };

  const startTrainJob = async () => {
    clearLogs("train");
    setTrainStatus("已启动");
    setTrainProgress(0);
    setMetrics({ loss: [], diffusion: [], lr: [], throughput: [], vram: [] });
    try {
      const res = await startTrain({
        data_dir: trainForm.dataDir,
        data_zip: trainForm.dataZip,
        use_zip: trainForm.useZip,
        extract_dir: trainForm.extractDir,
        save_dir: trainForm.saveDir,
        device: trainForm.device,
        epochs: trainForm.epochs,
        batch_size: trainForm.batchSize,
        steps_per_beat: trainForm.stepsPerBeat,
        max_voices: trainForm.maxVoices,
        aux_weight: trainForm.auxWeight,
        dashboard: trainForm.dashboard,
        resume: trainForm.resume,
        save_every_steps: trainForm.saveEverySteps,
        dynamic_length: trainForm.dynamicLength,
        max_seq_len: trainForm.maxSeqLen,
        bucket_size: trainForm.bucketSize,
        val_split: trainForm.valSplit,
        val_every_epochs: trainForm.valEveryEpochs,
        eval_data_dirs: trainForm.evalDataDirs,
        eval_max_samples: trainForm.evalMaxSamples,
      });
      connectLogStream("train", res.job_id);
      metricStream.current?.();
      metricStream.current = streamMetrics(res.job_id, pushMetric);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "训练启动失败";
      setTrainStatus(msg);
      appendLog("train", msg);
    }
  };

  const startAnalysisJob = async () => {
    clearLogs("analysis");
    setAnalysisStatus("已启动");
    try {
      const res = await startAnalysis({
        data_dir: analysisForm.dataDir,
        compare_dir: analysisForm.compareDir,
        steps_per_beat: analysisForm.stepsPerBeat,
        max_seq_len: analysisForm.maxSeqLen,
        sample_limit: analysisForm.sampleLimit,
        out_path: analysisForm.outPath,
      });
      connectLogStream("analysis", res.job_id);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "分析失败";
      setAnalysisStatus(msg);
      appendLog("analysis", msg);
    }
  };

  const startGenerateJob = async () => {
    clearLogs("generate");
    setGenerateStatus("已启动");
    setGenerateOutputs([]);
    try {
      const res = await startGenerate({
        checkpoint: generateForm.checkpoint,
        out_dir: generateForm.outDir,
        num_samples: generateForm.numSamples,
        seq_len: generateForm.seqLen,
        steps: generateForm.steps,
        max_voices: generateForm.maxVoices,
        steps_per_beat: generateForm.stepsPerBeat,
        tempo: generateForm.tempo,
        temperature: generateForm.temperature,
      });
      connectLogStream("generate", res.job_id);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "生成失败";
      setGenerateStatus(msg);
      appendLog("generate", msg);
    }
  };

  const startRenderJob = async () => {
    clearLogs("render");
    setRenderStatus("已启动");
    try {
      const payload: Record<string, unknown> = {
        out_dir: renderForm.outDir,
        soundfont: renderForm.soundfont || settings.defaultSoundfont,
        sample_rate: renderForm.sampleRate,
        normalize: renderForm.normalize,
      };
      if (renderForm.mode === "file") {
        payload.midi_file = renderForm.midiPath;
      } else {
        payload.midi_dir = renderForm.midiPath;
      }
      const res = await startRenderFlac(payload);
      connectLogStream("render", res.job_id);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "渲染失败";
      setRenderStatus(msg);
      appendLog("render", msg);
    }
  };

    const addClip = () => {
    if (!clipDraft.trackId || !clipDraft.name.trim()) return;
    const bars = Math.max(1, arrangerBars);
    const start = Math.min(bars, Math.max(1, clipDraft.start));
    const length = Math.min(bars - start + 1, Math.max(1, clipDraft.length));
    const track = arrangerTracks.find((t) => t.id === clipDraft.trackId);
    const clip: ArrangerClip = {
      id: makeId(),
      trackId: clipDraft.trackId,
      name: clipDraft.name.trim(),
      start,
      length,
      color: track?.color || arrangerColors[0],
    };
    setArrangerClips((prev) => [...prev, clip]);
    setClipDraft((prev) => ({ ...prev, name: "" }));
  };

  const removeClip = (id: string) => {
    setArrangerClips((prev) => prev.filter((clip) => clip.id !== id));
  };

  const addTrack = () => {
    const name = newTrackName.trim();
    if (!name) return;
    const color = arrangerColors[arrangerTracks.length % arrangerColors.length];
    const id = makeId();
    const newTrack: ArrangerTrack = { id, name, color };
    setArrangerTracks((prev) => [...prev, newTrack]);
    setMixerChannels((prev) => [
      ...prev,
      { id, name, volume: 0.6, pan: 0, mute: false, solo: false },
    ]);
    setNewTrackName("");
  };

  const updateMixer = (id: string, patch: Partial<MixerChannel>) => {
    setMixerChannels((prev) => prev.map((ch) => (ch.id === id ? { ...ch, ...patch } : ch)));
  };

  const exportArranger = () => {
    const payload = {
      name: arrangerName,
      bpm: arrangerBpm,
      timeSignature: arrangerTimeSig,
      bars: arrangerBars,
      tracks: arrangerTracks,
      clips: arrangerClips,
      mixer: mixerChannels,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${arrangerName || "arranger"}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const copyText = async (value: string) => {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(value);
      }
    } catch {
      // ignore
    }
  };

  const themeLabel = useMemo(() => (theme === "dark" ? "切到日间" : "切到夜间"), [theme]);
  const cpuBarWidth = 180;
  const cpuLineWidth = useMemo(() => {
    if (cpuPercent == null || Number.isNaN(cpuPercent)) return 0;
    return Math.max(2, (cpuPercent / 100) * cpuBarWidth);
  }, [cpuBarWidth, cpuPercent]);

  const navItems: { key: PageKey; label: string }[] = [
    { key: "import", label: "数据导入" },
    { key: "preprocess", label: "数据清洗" },
    { key: "train", label: "训练" },
    { key: "analysis", label: "数据分析" },
    { key: "generate", label: "生成 MIDI" },
    { key: "arranger", label: "编曲器" },
    { key: "render", label: "渲染 FLAC" },
    { key: "charts", label: "图表" },
    { key: "terminal", label: "终端" },
    { key: "settings", label: "设置" },
  ];
  return (
    <div className="app">
      <div className="topbar">
        <div>
          <div className="brand">MIDI 集成训练工作流</div>
          <div className="hud-tag">已连接服务器后端 · 版本号0.3beta</div>
        </div>
        <div className="topbar-right">
          <div
            className="hud-barcode"
            style={{ width: `${cpuBarWidth}px` }}
            title={cpuPercent == null ? "CPU: --" : `CPU: ${cpuPercent.toFixed(0)}%`}
          >
            <div className="hud-barcode-line" style={{ width: `${cpuLineWidth}px` }} />
          </div>
          <button
            className="theme-toggle"
            aria-label={themeLabel}
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          >
            {theme === "dark" ? (
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <circle cx="12" cy="12" r="4" />
                <g stroke="currentColor" strokeWidth="2">
                  <line x1="12" y1="1" x2="12" y2="4" />
                  <line x1="12" y1="20" x2="12" y2="23" />
                  <line x1="1" y1="12" x2="4" y2="12" />
                  <line x1="20" y1="12" x2="23" y2="12" />
                  <line x1="4.5" y1="4.5" x2="6.8" y2="6.8" />
                  <line x1="17.2" y1="17.2" x2="19.5" y2="19.5" />
                  <line x1="4.5" y1="19.5" x2="6.8" y2="17.2" />
                  <line x1="17.2" y1="6.8" x2="19.5" y2="4.5" />
                </g>
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M21 14.5a8.5 8.5 0 1 1-11.5-11 7 7 0 0 0 11.5 11z" />
              </svg>
            )}
          </button>
        </div>
      </div>

      <div className="nav">
        {navItems.map((item) => (
          <button
            key={item.key}
            className={classNames({ active: page === item.key })}
            onClick={() => setPage(item.key)}
          >
            {item.label}
          </button>
        ))}
      </div>

      {page === "import" && (
        <div className="page page-animate" key={page}>
          <div className="panel">
            <h2>数据导入</h2>
            <div className="field">
              <label>ZIP 文件</label>
              <input type="file" accept=".zip" onChange={(e) => setImportForm((prev) => ({ ...prev, zipFile: e.target.files?.[0] || null }))} />
            </div>
            <div className="field">
              <label>解压目录</label>
              <input value={importForm.outDir} onChange={(e) => setImportForm((prev) => ({ ...prev, outDir: e.target.value }))} />
            </div>
            <div className="field">
              <label>
                <input type="checkbox" checked={importForm.clearDir} onChange={(e) => setImportForm((prev) => ({ ...prev, clearDir: e.target.checked }))} />
                清空目标目录
              </label>
            </div>
            <div className="actions">
              <button className="btn" onClick={() => startImport(false)}>解压导入</button>
              <button className="btn secondary" onClick={() => startImport(true)}>使用已有目录</button>
            </div>
            <div className="badge" style={{ marginTop: 8 }}>{importStatus || "等待操作"}</div>
          </div>
        </div>
      )}

      {page === "preprocess" && (
        <div className="page page-animate" key={page}>
          <div className="panel">
            <h2>数据清洗</h2>
            <div className="field">
              <label>原始数据目录</label>
              <input value={preForm.dataDir} onChange={(e) => setPreForm((prev) => ({ ...prev, dataDir: e.target.value }))} />
            </div>
            <div className="field">
              <label>输出目录</label>
              <input value={preForm.outputDir} onChange={(e) => setPreForm((prev) => ({ ...prev, outputDir: e.target.value }))} />
            </div>
            <div className="field">
              <label>输出 ZIP 路径</label>
              <input value={preForm.zipPath} onChange={(e) => setPreForm((prev) => ({ ...prev, zipPath: e.target.value }))} />
            </div>
            <div className="grid-2">
              <div className="field">
                <label>最小音符长度</label>
                <input type="number" value={preForm.minNoteLength} onChange={(e) => setPreForm((prev) => ({ ...prev, minNoteLength: Number(e.target.value) }))} />
              </div>
              <div className="field">
                <label>最大文件大小(MB)</label>
                <input type="number" value={preForm.maxSizeMb} onChange={(e) => setPreForm((prev) => ({ ...prev, maxSizeMb: Number(e.target.value) }))} />
              </div>
              <div className="field">
                <label>Workers(0=自动)</label>
                <input type="number" value={preForm.workers} onChange={(e) => setPreForm((prev) => ({ ...prev, workers: Number(e.target.value) }))} />
              </div>
            </div>
            <div className="field">
              <label>
                <input type="checkbox" checked={preForm.keepStructure} onChange={(e) => setPreForm((prev) => ({ ...prev, keepStructure: e.target.checked }))} />
                保留目录结构
              </label>
            </div>
            <div className="field">
              <label>
                <input type="checkbox" checked={preForm.skipZip} onChange={(e) => setPreForm((prev) => ({ ...prev, skipZip: e.target.checked }))} />
                跳过打包 ZIP
              </label>
            </div>
            <div className="actions">
              <button className="btn" onClick={startPre}>开始清洗</button>
            </div>
            <div className="badge" style={{ marginTop: 8 }}>{preStatus || "等待操作"}</div>
          </div>
        </div>
      )}

      {page === "train" && (
        <div className="page page-animate" key={page}>
          <div className="panel">
            <h2>训练</h2>
            <div className="grid-2">
              <div className="field">
                <label>数据目录</label>
                <input value={trainForm.dataDir} onChange={(e) => setTrainForm((prev) => ({ ...prev, dataDir: e.target.value }))} />
              </div>
              <div className="field">
                <label>使用清洗 ZIP</label>
                <select value={trainForm.useZip ? "yes" : "no"} onChange={(e) => setTrainForm((prev) => ({ ...prev, useZip: e.target.value === "yes" }))}>
                  <option value="no">否</option>
                  <option value="yes">是</option>
                </select>
              </div>
              <div className="field">
                <label>ZIP 路径</label>
                <input value={trainForm.dataZip} onChange={(e) => setTrainForm((prev) => ({ ...prev, dataZip: e.target.value }))} />
              </div>
              <div className="field">
                <label>解压目录</label>
                <input value={trainForm.extractDir} onChange={(e) => setTrainForm((prev) => ({ ...prev, extractDir: e.target.value }))} />
              </div>
              <div className="field">
                <label>保存目录</label>
                <input value={trainForm.saveDir} onChange={(e) => setTrainForm((prev) => ({ ...prev, saveDir: e.target.value }))} />
              </div>
              <div className="field">
                <label>设备</label>
                <select value={trainForm.device} onChange={(e) => setTrainForm((prev) => ({ ...prev, device: e.target.value }))}>
                  <option value="4090">4090</option>
                  <option value="4050">4050</option>
                </select>
              </div>
              <div className="field">
                <label>训练轮数</label>
                <input type="number" value={trainForm.epochs} onChange={(e) => setTrainForm((prev) => ({ ...prev, epochs: Number(e.target.value) }))} />
              </div>
              <div className="field">
                <label>批大小(0=自动)</label>
                <input type="number" value={trainForm.batchSize} onChange={(e) => setTrainForm((prev) => ({ ...prev, batchSize: Number(e.target.value) }))} />
              </div>
              <div className="field">
                <label>Steps/Beat(0=自动)</label>
                <input type="number" value={trainForm.stepsPerBeat} onChange={(e) => setTrainForm((prev) => ({ ...prev, stepsPerBeat: Number(e.target.value) }))} />
              </div>
              <div className="field">
                <label>最大声部(0=自动)</label>
                <input type="number" value={trainForm.maxVoices} onChange={(e) => setTrainForm((prev) => ({ ...prev, maxVoices: Number(e.target.value) }))} />
              </div>
              <div className="field">
                <label>Aux 权重</label>
                <input type="number" value={trainForm.auxWeight} onChange={(e) => setTrainForm((prev) => ({ ...prev, auxWeight: Number(e.target.value) }))} />
              </div>
              <div className="field">
                <label>保存步数间隔</label>
                <input type="number" value={trainForm.saveEverySteps} onChange={(e) => setTrainForm((prev) => ({ ...prev, saveEverySteps: Number(e.target.value) }))} />
              </div>
              <div className="field">
                <label>断点恢复</label>
                <select value={trainForm.resume ? "yes" : "no"} onChange={(e) => setTrainForm((prev) => ({ ...prev, resume: e.target.value === "yes" }))}>
                  <option value="no">否</option>
                  <option value="yes">是</option>
                </select>
              </div>
              <div className="field">
                <label>启用训练监控</label>
                <select value={trainForm.dashboard ? "yes" : "no"} onChange={(e) => setTrainForm((prev) => ({ ...prev, dashboard: e.target.value === "yes" }))}>
                  <option value="yes">是</option>
                  <option value="no">否</option>
                </select>
              </div>
              <div className="field">
                <label>动态长度</label>
                <select value={trainForm.dynamicLength ? "yes" : "no"} onChange={(e) => setTrainForm((prev) => ({ ...prev, dynamicLength: e.target.value === "yes" }))}>
                  <option value="yes">是</option>
                  <option value="no">否</option>
                </select>
              </div>
              <div className="field">
                <label>最大序列长度(0=自动)</label>
                <input type="number" value={trainForm.maxSeqLen} onChange={(e) => setTrainForm((prev) => ({ ...prev, maxSeqLen: Number(e.target.value) }))} />
              </div>
              <div className="field">
                <label>分桶粒度</label>
                <input type="number" value={trainForm.bucketSize} onChange={(e) => setTrainForm((prev) => ({ ...prev, bucketSize: Number(e.target.value) }))} />
              </div>
              <div className="field">
                <label>验证集比例</label>
                <input type="number" value={trainForm.valSplit} onChange={(e) => setTrainForm((prev) => ({ ...prev, valSplit: Number(e.target.value) }))} />
              </div>
              <div className="field">
                <label>验证频率(epochs)</label>
                <input type="number" value={trainForm.valEveryEpochs} onChange={(e) => setTrainForm((prev) => ({ ...prev, valEveryEpochs: Number(e.target.value) }))} />
              </div>
              <div className="field">
                <label>额外评估集(逗号分隔)</label>
                <input value={trainForm.evalDataDirs} onChange={(e) => setTrainForm((prev) => ({ ...prev, evalDataDirs: e.target.value }))} />
              </div>
              <div className="field">
                <label>评估批次数上限</label>
                <input type="number" value={trainForm.evalMaxSamples} onChange={(e) => setTrainForm((prev) => ({ ...prev, evalMaxSamples: Number(e.target.value) }))} />
              </div>
            </div>
            <div className="actions" style={{ marginTop: 12 }}>
              <button className="btn" onClick={startTrainJob}>开始训练</button>
            </div>
            <div style={{ marginTop: 12 }}>
              <div className="progress"><span style={{ width: `${Math.round(trainProgress * 100)}%` }} /></div>
              <div className="badge" style={{ marginTop: 8 }}>{trainStatus}</div>
            </div>
          </div>
        </div>
      )}

      {page === "analysis" && (
        <div className="page page-animate" key={page}>
          <div className="panel">
            <h2>数据分析</h2>
            <div className="field">
              <label>数据目录</label>
              <input value={analysisForm.dataDir} onChange={(e) => setAnalysisForm((prev) => ({ ...prev, dataDir: e.target.value }))} />
            </div>
            <div className="field">
              <label>对比目录(可选)</label>
              <input value={analysisForm.compareDir} onChange={(e) => setAnalysisForm((prev) => ({ ...prev, compareDir: e.target.value }))} />
            </div>
            <div className="grid-2">
              <div className="field">
                <label>Steps/Beat</label>
                <input type="number" value={analysisForm.stepsPerBeat} onChange={(e) => setAnalysisForm((prev) => ({ ...prev, stepsPerBeat: Number(e.target.value) }))} />
              </div>
              <div className="field">
                <label>最大序列长度</label>
                <input type="number" value={analysisForm.maxSeqLen} onChange={(e) => setAnalysisForm((prev) => ({ ...prev, maxSeqLen: Number(e.target.value) }))} />
              </div>
            </div>
            <div className="grid-2">
              <div className="field">
                <label>采样上限</label>
                <input type="number" value={analysisForm.sampleLimit} onChange={(e) => setAnalysisForm((prev) => ({ ...prev, sampleLimit: Number(e.target.value) }))} />
              </div>
              <div className="field">
                <label>输出报告路径</label>
                <input value={analysisForm.outPath} onChange={(e) => setAnalysisForm((prev) => ({ ...prev, outPath: e.target.value }))} />
              </div>
            </div>
            <div className="actions">
              <button className="btn" onClick={startAnalysisJob}>开始分析</button>
            </div>
            <div className="badge" style={{ marginTop: 8 }}>{analysisStatus || "等待操作"}</div>
          </div>
        </div>
      )}

      {page === "generate" && (
        <div className="page page-animate" key={page}>
          <div className="panel">
            <h2>生成 MIDI</h2>
            <div className="field">
              <label>Checkpoint</label>
              <input value={generateForm.checkpoint} onChange={(e) => setGenerateForm((prev) => ({ ...prev, checkpoint: e.target.value }))} />
            </div>
            <div className="field">
              <label>输出目录</label>
              <input value={generateForm.outDir} onChange={(e) => setGenerateForm((prev) => ({ ...prev, outDir: e.target.value }))} />
            </div>
            <div className="grid-2">
              <div className="field">
                <label>生成数量</label>
                <input type="number" value={generateForm.numSamples} onChange={(e) => setGenerateForm((prev) => ({ ...prev, numSamples: Number(e.target.value) }))} />
              </div>
              <div className="field">
                <label>Seq Len</label>
                <input type="number" value={generateForm.seqLen} onChange={(e) => setGenerateForm((prev) => ({ ...prev, seqLen: Number(e.target.value) }))} />
              </div>
              <div className="field">
                <label>采样步数</label>
                <input type="number" value={generateForm.steps} onChange={(e) => setGenerateForm((prev) => ({ ...prev, steps: Number(e.target.value) }))} />
              </div>
              <div className="field">
                <label>最大声部</label>
                <input type="number" value={generateForm.maxVoices} onChange={(e) => setGenerateForm((prev) => ({ ...prev, maxVoices: Number(e.target.value) }))} />
              </div>
              <div className="field">
                <label>Steps/Beat</label>
                <input type="number" value={generateForm.stepsPerBeat} onChange={(e) => setGenerateForm((prev) => ({ ...prev, stepsPerBeat: Number(e.target.value) }))} />
              </div>
              <div className="field">
                <label>Tempo</label>
                <input type="number" value={generateForm.tempo} onChange={(e) => setGenerateForm((prev) => ({ ...prev, tempo: Number(e.target.value) }))} />
              </div>
              <div className="field">
                <label>Temperature</label>
                <input type="number" value={generateForm.temperature} onChange={(e) => setGenerateForm((prev) => ({ ...prev, temperature: Number(e.target.value) }))} />
              </div>
            </div>
            <div className="actions">
              <button className="btn" onClick={startGenerateJob}>开始生成</button>
            </div>
            <div className="badge" style={{ marginTop: 8 }}>{generateStatus || "等待操作"}</div>
          </div>
          <div className="panel">
            <h2>输出文件</h2>
            {generateOutputs.length === 0 ? (
              <div className="badge">暂无输出</div>
            ) : (
              <div className="output-list">
                {generateOutputs.map((item) => (
                  <div key={item} className="output-item">
                    <span>{item}</span>
                    <button className="btn secondary" onClick={() => copyText(item)}>复制路径</button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {page === "arranger" && (
        <div className="page page-animate" key={page}>
          <div className="panel">
            <h2>编曲器 · 段落混音</h2>
            <div className="grid-2">
              <div className="field">
                <label>工程名称</label>
                <input value={arrangerName} onChange={(e) => setArrangerName(e.target.value)} />
              </div>
              <div className="field">
                <label>BPM</label>
                <input type="number" value={arrangerBpm} onChange={(e) => setArrangerBpm(Number(e.target.value))} />
              </div>
              <div className="field">
                <label>小节数</label>
                <input type="number" value={arrangerBars} onChange={(e) => setArrangerBars(Math.max(8, Number(e.target.value)))} />
              </div>
              <div className="field">
                <label>拍号</label>
                <input value={arrangerTimeSig} onChange={(e) => setArrangerTimeSig(e.target.value)} />
              </div>
            </div>
            <div className="actions" style={{ marginTop: 8 }}>
              <button className="btn">Play</button>
              <button className="btn secondary">Stop</button>
              <button className="btn ghost">Loop</button>
              <button className="btn ghost" onClick={exportArranger}>导出 JSON</button>
              <div className="badge">{arrangerName} · {arrangerBpm} BPM · {arrangerTimeSig}</div>
            </div>
          </div>

          <div className="panel">
            <h2>段落 / Clips</h2>
            <div className="grid-2">
              <div className="field">
                <label>轨道</label>
                <select value={clipDraft.trackId} onChange={(e) => setClipDraft((prev) => ({ ...prev, trackId: e.target.value }))}>
                  {arrangerTracks.map((track) => (
                    <option key={track.id} value={track.id}>{track.name}</option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label>片段名称</label>
                <input value={clipDraft.name} onChange={(e) => setClipDraft((prev) => ({ ...prev, name: e.target.value }))} />
              </div>
              <div className="field">
                <label>起始小节</label>
                <input type="number" value={clipDraft.start} onChange={(e) => setClipDraft((prev) => ({ ...prev, start: Number(e.target.value) }))} />
              </div>
              <div className="field">
                <label>长度(小节)</label>
                <input type="number" value={clipDraft.length} onChange={(e) => setClipDraft((prev) => ({ ...prev, length: Number(e.target.value) }))} />
              </div>
            </div>
            <div className="actions">
              <button className="btn" onClick={addClip}>添加段落</button>
              <div className="field" style={{ marginBottom: 0 }}>
                <label>新增轨道</label>
                <div className="inline-field">
                  <input value={newTrackName} placeholder="新轨道名称" onChange={(e) => setNewTrackName(e.target.value)} />
                  <button className="btn secondary" onClick={addTrack}>添加轨道</button>
                </div>
              </div>
            </div>
          </div>

          <div className="panel arranger-panel">
            <h2>时间线</h2>
            <div className="arranger-timeline">
              <div className="arranger-ruler" style={{ gridTemplateColumns: `repeat(${arrangerBars}, minmax(18px, 1fr))` }}>
                {Array.from({ length: arrangerBars }).map((_, idx) => (
                  <div key={idx} className="ruler-cell">{idx + 1}</div>
                ))}
              </div>
              {arrangerTracks.map((track) => (
                <div key={track.id} className="arranger-row">
                  <div className="track-label" style={{ borderColor: track.color }}>
                    <span className="track-dot" style={{ background: track.color }} />
                    <div>
                      <div className="track-name">{track.name}</div>
                      <div className="track-meta">BUS A · {track.midi ? track.midi : "No MIDI"}</div>
                    </div>
                  </div>
                  <div className="track-grid" style={{ gridTemplateColumns: `repeat(${arrangerBars}, minmax(18px, 1fr))` }}>
                    {arrangerClips
                      .filter((clip) => clip.trackId === track.id)
                      .map((clip) => (
                        <div
                          key={clip.id}
                          className="arranger-clip"
                          style={{
                            gridColumn: `${clip.start} / span ${clip.length}`,
                            background: clip.color,
                          }}
                        >
                          <span>{clip.name}</span>
                          <button className="clip-remove" onClick={() => removeClip(clip.id)}>
                            ×
                          </button>
                        </div>
                      ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="panel">
            <h2>混音器</h2>
            <div className="mixer">
              {mixerChannels.map((channel) => (
                <div key={channel.id} className="mixer-channel">
                  <div className="mixer-name">{channel.name}</div>
                  <div className="mixer-buttons">
                    <button className={classNames("btn ghost", { active: channel.mute })} onClick={() => updateMixer(channel.id, { mute: !channel.mute })}>M</button>
                    <button className={classNames("btn ghost", { active: channel.solo })} onClick={() => updateMixer(channel.id, { solo: !channel.solo })}>S</button>
                  </div>
                  <div className="mixer-fader">
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.01}
                      value={channel.volume}
                      onChange={(e) => updateMixer(channel.id, { volume: Number(e.target.value) })}
                    />
                  </div>
                  <div className="mixer-pan">
                    <input
                      type="range"
                      min={-50}
                      max={50}
                      step={1}
                      value={channel.pan}
                      onChange={(e) => updateMixer(channel.id, { pan: Number(e.target.value) })}
                    />
                  </div>
                  <div className="mixer-meta">VOL {Math.round(channel.volume * 100)}% · PAN {channel.pan}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {page === "render" && (
        <div className="page page-animate" key={page}>
          <div className="panel">
            <h2>渲染 FLAC</h2>
            <p>需要 SoundFont 与 Fluidsynth 支持，缺失会在日志里提示。</p>
            <div className="grid-2">
              <div className="field">
                <label>输入类型</label>
                <select value={renderForm.mode} onChange={(e) => setRenderForm((prev) => ({ ...prev, mode: e.target.value as "file" | "dir" }))}>
                  <option value="file">单个 MIDI 文件</option>
                  <option value="dir">MIDI 目录</option>
                </select>
              </div>
              <div className="field">
                <label>输入路径</label>
                <input value={renderForm.midiPath} onChange={(e) => setRenderForm((prev) => ({ ...prev, midiPath: e.target.value }))} />
              </div>
              <div className="field">
                <label>输出目录</label>
                <input value={renderForm.outDir} onChange={(e) => setRenderForm((prev) => ({ ...prev, outDir: e.target.value }))} />
              </div>
              <div className="field">
                <label>SoundFont 路径</label>
                <input value={renderForm.soundfont} onChange={(e) => setRenderForm((prev) => ({ ...prev, soundfont: e.target.value }))} />
              </div>
              <div className="field">
                <label>采样率</label>
                <input type="number" value={renderForm.sampleRate} onChange={(e) => setRenderForm((prev) => ({ ...prev, sampleRate: Number(e.target.value) }))} />
              </div>
              <div className="field">
                <label>
                  <input type="checkbox" checked={renderForm.normalize} onChange={(e) => setRenderForm((prev) => ({ ...prev, normalize: e.target.checked }))} />
                  归一化音量
                </label>
              </div>
            </div>
            <div className="actions">
              <button className="btn" onClick={startRenderJob}>开始渲染</button>
            </div>
            <div className="badge" style={{ marginTop: 8 }}>{renderStatus || "等待操作"}</div>
          </div>
        </div>
      )}

      {page === "charts" && (
        <div className="page page-animate" key={page}>
          <div className="panel">
            <h2>清洗进度</h2>
            <div className="chart-grid">
              <MetricChart title="Processed Files" values={preprocessMetrics.count} color="#20f8b3" />
              <MetricChart title="Speed (file/s)" values={preprocessMetrics.speed} color="#f31be5" />
              <MetricChart title="Elapsed (s)" values={preprocessMetrics.elapsed} color="#cbf716" />
              <MetricChart title="Remaining (s)" values={preprocessMetrics.remaining} color="#00fff0" />
              <MetricChart title="CPU (%)" values={preprocessMetrics.cpu} color="#dfff00" />
            </div>
          </div>
          <div className="panel">
            <h2>训练指标曲线</h2>
            <div className="chart-grid">
              <MetricChart title="Loss" values={metrics.loss} color="#dfff00" />
              <MetricChart title="Diffusion" values={metrics.diffusion} color="#ff2df0" />
              <MetricChart title="LR" values={metrics.lr} color="#cbf716" />
              <MetricChart title="Throughput" values={metrics.throughput} color="#00fff0" />
              <MetricChart title="VRAM(GB)" values={metrics.vram} color="#dfff00" />
            </div>
          </div>
        </div>
      )}

      {page === "terminal" && (
        <div className="page page-animate" key={page}>
          <div className="panel">
            <h2>终端日志</h2>
            <div className="terminal-tabs">
              {(["global", "import", "preprocess", "train", "analysis", "generate", "render"] as LogKey[]).map((key) => (
                <button
                  key={key}
                  className={classNames({ active: terminalTab === key })}
                  onClick={() => setTerminalTab(key)}
                >
                  {key === "global" ? "??" : key}
                </button>
              ))}
            </div>
            <div className="actions" style={{ marginBottom: 12 }}>
              <button className="btn secondary" onClick={() => clearLogs(terminalTab)}>清空当前日志</button>
            </div>
            <div className="logs" ref={logRef}>{logs[terminalTab].join("\n") || "暂无日志"}</div>
          </div>
        </div>
      )}

      {page === "settings" && (
        <div className="page page-animate" key={page}>
          <div className="panel">
            <h2>前端设置</h2>
            <div className="grid-2">
              <div className="field">
                <label>默认 SoundFont</label>
                <input
                  value={settings.defaultSoundfont}
                  onChange={(e) => setSettings((prev) => ({ ...prev, defaultSoundfont: e.target.value }))}
                />
              </div>
              <div className="field">
                <label>主题</label>
                <select value={theme} onChange={(e) => setTheme(e.target.value)}>
                  <option value="dark">夜间工业</option>
                  <option value="light">日间</option>
                </select>
              </div>
              <div className="field">
                <label>日志自动滚动</label>
                <select value={settings.autoScroll ? "yes" : "no"} onChange={(e) => setSettings((prev) => ({ ...prev, autoScroll: e.target.value === "yes" }))}>
                  <option value="yes">开启</option>
                  <option value="no">关闭</option>
                </select>
              </div>
              <div className="field">
                <label>日志最大行数</label>
                <input
                  type="number"
                  value={settings.maxLogLines}
                  onChange={(e) => setSettings((prev) => ({ ...prev, maxLogLines: Number(e.target.value) }))}
                />
              </div>
              <div className="field">
                <label>HUD 背景纹理</label>
                <select value={settings.hudEnabled ? "yes" : "no"} onChange={(e) => setSettings((prev) => ({ ...prev, hudEnabled: e.target.value === "yes" }))}>
                  <option value="yes">开启</option>
                  <option value="no">关闭</option>
                </select>
              </div>
            </div>
            <div className="badge">设置将立即生效，仅作用于本地浏览器。</div>
          </div>
        </div>
      )}
    </div>
  );
}

