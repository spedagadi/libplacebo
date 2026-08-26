"""
hdr10_xgb_viewer.py — HDR10 viewer: Spline vs Contrast Recovery (Flask)

Contrast-recovery mode uses the GPU bilateral filter from dv_render:
  decode → tone map → GPU bilateral CR (9x9 edge-preserving blur) → L2 trim → output

Run: python ml/hdr10_xgb_viewer.py
URL: http://localhost:8601
"""
import sys, os, subprocess, tempfile, traceback, io, base64, json, time, struct, threading
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template_string
from pathlib import Path

_HERE  = Path(__file__).parent.resolve()
_TOOLS = _HERE.parent / "tools"
TIMING_LOG = _HERE.parent / "realtime_timing.jsonl"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_TOOLS))
os.environ["PATH"] = (r"C:\Code\libplacebo\build\src;C:\msys64\ucrt64\bin;C:\msys64\usr\bin;"
                      + os.environ.get("PATH", ""))
# Also set DLL directories for MSYS2 runtime
try:
    os.add_dll_directory(r"C:\msys64\ucrt64\bin")
    os.add_dll_directory(r"C:\msys64\usr\bin")
except OSError:
    pass  # Windows < 8.0 — PATH is used instead

BUILD_DIR = _HERE.parent / "build_persistent"
if not (BUILD_DIR / "tools" / "dv_render.exe").exists():
    BUILD_DIR = _HERE.parent / "build"
DV_RENDER = str(BUILD_DIR / "tools" / "dv_render.exe")
DV_SERVER_LOG = _HERE.parent / "persistent_dv_render.log"
BASELINE  = str(BUILD_DIR / "tools" / "libplacebo_baseline_eval.exe")
NATIVE_MODEL_PATH = _HERE.parent / "xgb_gamma_model.plxgb"
W, H      = 1280, 720


class PersistentRenderer:
    """Persistent dv_render process for a single render mode."""

    def __init__(self, mode="spline", iir_alpha=0.10, iir_scene_cut=0.15):
        self.mode = mode
        self.iir_alpha = iir_alpha
        self.iir_scene_cut = iir_scene_cut
        self.process = None
        self.lock = threading.Lock()
        self.stderr_file = None

    def _start(self):
        self.stderr_file = DV_SERVER_LOG.open("ab")
        env = os.environ.copy()
        env["PATH"] = (r"C:\Code\libplacebo\build_persistent\src;"
                r"C:\Code\libplacebo\build_persistent\tools;"
                r"C:\msys64\ucrt64\bin;C:\msys64\usr\bin;"
                + env.get("PATH", ""))
        cmd = [DV_RENDER, "--server", "--mode", self.mode]
        if self.mode != "spline":
            cmd += ["--model", str(NATIVE_MODEL_PATH),
                    "--iir-alpha", str(self.iir_alpha),
                    "--iir-scene-cut", str(self.iir_scene_cut)]
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.stderr_file, env=env, bufsize=0,
        )

    def _stop(self):
        if self.process is not None:
            self.process.kill()
            self.process.wait(timeout=5)
            self.process = None
        if self.stderr_file is not None:
            self.stderr_file.close()
            self.stderr_file = None

    def _send_request(self, request):
        with self.lock:
            for attempt in range(2):
                try:
                    if self.process is None or self.process.poll() is not None:
                        self._start()
                    self.process.stdin.write(request)
                    self.process.stdin.flush()
                    header = self._read_exact(20)
                    if header[:4] != b"DVR2":
                        raise RuntimeError("invalid dv_render server header")
                    version, width, height, count = struct.unpack("<4I", header[4:])
                    if version != 2 or width != W or height != H or count != 2:
                        raise RuntimeError(f"unexpected DVR2 response: {version}, {width}x{height}, {count}")
                    img = None
                    frame_info = None
                    for _ in range(count):
                        mode, length = struct.unpack("<2I", self._read_exact(8))
                        payload = self._read_exact(length)
                        if mode == 3:
                            if length != 52:
                                raise RuntimeError(f"unexpected frame-info payload size: {length}")
                            values = struct.unpack("<I12f", payload)
                            frame_info = dict(zip(
                                ("flags", "gamma", "l1_max_pq", "l1_avg_pq",
                                 "cr_strength", "l2_power", "fire_pop_strength",
                                 "radiance_knee", "radiance_strength",
                                 "chroma_neutral_boost", "chroma_fire_boost",
                                 "chroma_knee", "chroma_skin_protect"), values))
                        else:
                            expected = W * H * 3
                            if length != expected:
                                raise RuntimeError(f"unexpected frame payload size: {length}")
                            img = np.frombuffer(payload, dtype=np.uint8).reshape(H, W, 3).copy()
                    if img is None or frame_info is None:
                        raise RuntimeError("incomplete DVR2 response")
                    return img, frame_info
                except (BrokenPipeError, EOFError, OSError, RuntimeError):
                    self._stop()
                    if attempt:
                        raise
        raise RuntimeError("persistent dv_render failed")

    def _read_exact(self, size):
        data = bytearray()
        while len(data) < size:
            chunk = self.process.stdout.read(size - len(data))
            if not chunk:
                raise EOFError("dv_render server closed stdout")
            data.extend(chunk)
        return bytes(data)


_RENDERER_SPLINE = PersistentRenderer(mode="spline")
_RENDERER_CR     = PersistentRenderer(mode="contrast-recovery")

# ── Core functions ────────────────────────────────────────────────────────────
def render_persistent_pair(video, pts, nits, gamma_mode, gamma, cr_mode,
                            cr_strength, fire_pop_mode, fire_pop, radiance_mode,
                            radiance_knee, radiance_strength,
                            chroma_mode="off", chroma_neutral=1.20, chroma_fire=1.35,
                            chroma_knee=0.55, chroma_skin=0.95):
    """Render spline and CR using two separate processes (no shared renderer state)."""
    spline_request = json.dumps({
        "input": str(video), "pts": float(pts), "width": W, "height": H,
        "out_nits": float(nits),
    }).encode("utf-8") + b"\n"

    cr_request = json.dumps({
        "input": str(video), "pts": float(pts), "width": W, "height": H,
        "out_nits": float(nits),
        "gamma_mode": gamma_mode, "contrast_gamma": float(gamma),
        "cr_mode": cr_mode, "cr_strength": float(cr_strength),
        "fire_pop_mode": fire_pop_mode, "fire_pop_strength": float(fire_pop),
        "radiance_mode": radiance_mode, "radiance_knee": float(radiance_knee),
        "radiance_strength": float(radiance_strength),
        "chroma_mode": chroma_mode,
        "chroma_neutral_boost": float(chroma_neutral),
        "chroma_fire_boost": float(chroma_fire),
        "chroma_knee": float(chroma_knee),
        "chroma_skin_protect": float(chroma_skin),
    }).encode("utf-8") + b"\n"

    img_spl, fi_spl = _RENDERER_SPLINE._send_request(spline_request)
    img_cr,  fi_cr  = _RENDERER_CR._send_request(cr_request)
    # Use CR frame_info (has gamma/CR/chroma values); merge l1 from spline
    fi_cr["l1_max_pq"] = fi_spl["l1_max_pq"]
    fi_cr["l1_avg_pq"] = fi_spl["l1_avg_pq"]
    return img_spl, img_cr, fi_cr

def render_frame(video, pts, nits, maxscl, avg, mode="spline",
                 contrast_gamma=0.0, cr_strength=0.0, fire_pop_strength=0.0):
    """Render frame using dv_render.

    mode: "spline" (pure libplacebo) | "contrast-recovery" (spline + auto L2)
    contrast_gamma: manual gamma override (>0), 0 = auto-predict from scene stats.
    cr_strength: manual CR boost strength (0.0-0.5, 0 = auto from gamma).
    fire_pop_strength: warm chroma reshaping multiplier (0=off, 1.0=default).

    Returns: (frame_array, log_lines_list) — log_lines is always populated.
    """
    log = []
    log.append("=" * 64)
    log.append(f"  PIPELINE LOG")
    log.append(f"  video:  {video}")
    log.append(f"  pts:    {pts:.3f} (probe offsets: {0}, 1, -1)")
    log.append(f"  nits:   {int(nits)} (out-nits)")
    log.append(f"  mode:   {mode}")
    log.append(f"  maxscl: {maxscl:.1f} (L1 max)")
    log.append(f"  avg:    {avg:.1f} (L1 avg)")

    # Build explicit pipeline description
    log.append("")
    log.append("--- Pipeline Description ---")
    if mode == "spline":
        log.append("  [1] Tone-map HDR → SDR (libplacebo pl_render)")
        log.append("  [2] Quantize to uint8 (8-bit)")
        log.append("  NOTE: No secondary processing (CR, fire-pop, L2) in this mode.")
    elif mode == "contrast-recovery":
        log.append("  [1] Tone-map HDR → SDR (libplacebo pl_render)")
        log.append("  [2] GPU Spatial CR (bilateral filter, ~4-8% boost)")
        log.append("  [3] Fire-pop warm chroma reshaping (float pipeline)")
        log.append("  [4] L2 gamma trim (8-bit output with dither)")
        log.append("  [5] Quantize to uint8 (8-bit)")
        log.append("  NOTE: All intermediate processing in float32 to avoid double-quantization.")
    log.append("")

    # Ensure MSYS2 runtime is findable by subprocesses (DLL search path)
    subprocess_env = os.environ.copy()
    msys_paths = [r"C:\msys64\ucrt64\bin", r"C:\msys64\usr\bin"]
    subprocess_env["PATH"] = ";".join(msys_paths) + ";" + subprocess_env.get("PATH", "")

    best_frame = None
    best_stderr = b""
    for off in [0, 1, -1]:
        p = max(0.0, pts + off)
        cmd = [DV_RENDER, "--input", video, "--pts", f"{p:.3f}",
               "--mode", mode, "--width", str(W), "--height", str(H),
               "--out-nits", str(int(nits)),
               "--l1-max", f"{maxscl:.6f}", "--l1-avg", f"{avg:.6f}"]
        params_str = ""
        if contrast_gamma > 0.0:
            cmd += ["--contrast-gamma", f"{contrast_gamma:.3f}"]
            params_str += f"\n  contrast_gamma: {contrast_gamma:.3f} (manual override)"
        else:
            params_str += "\n  contrast_gamma: 0.0 (auto-predict from scene stats)"
        if cr_strength > 0.0:
            cmd += ["--cr-strength", f"{cr_strength:.3f}"]
            params_str += f"\n  cr_strength:    {cr_strength:.3f} (manual override)"
        else:
            params_str += "\n  cr_strength:    0.0 (auto from gamma)"
        if fire_pop_strength > 0.0:
            cmd += ["--fire-pop-strength", f"{fire_pop_strength:.2f}"]
            params_str += f"\n  fire_pop:       {fire_pop_strength:.2f} (1.0=default, 0=off)"
        else:
            params_str += "\n  fire_pop:       0.0 (off)"
        log.append("--- dv_render Command ---")
        log.append(f"  Offset: {off:+d} → pts={p:.3f}")
        log.append(f"  Full command:")
        log.append(f"    {' '.join(cmd)}")
        log.append(f"\n  --- Active Parameters ---{params_str}")
        log.append("")

        try:
            r = subprocess.run(cmd, capture_output=True, env=subprocess_env, timeout=30)
            stderr_text = r.stderr.decode("utf-8", errors="replace")
            log.append("--- dv_render stderr (pts={:.3f}) ---".format(p))
            for line in stderr_text.strip().split("\n"):
                log.append("  " + line)
            log.append(f"--- end stderr (exit_code={r.returncode}) ---")
            log.append("")

            if len(r.stdout) == W * H * 3:
                best_frame = np.frombuffer(r.stdout, dtype=np.uint8).reshape(H, W, 3).copy()
                best_stderr = r.stderr
                log.append(f">>> SUCCESS at offset {off:+d}: {len(r.stdout)} bytes output.")
                break
            else:
                log.append(f">>> BAD OUTPUT SIZE at offset {off:+d}: {len(r.stdout)} bytes (expected {W*H*3}).")
        except subprocess.TimeoutExpired:
            log.append(f">>> TIMEOUT at offset {off:+d} (30s limit).")
        except Exception as e:
            log.append(f">>> EXCEPTION at offset {off:+d}: {e}")

    log.append("")
    log.append("=" * 64)
    log.append("  END LOG")
    log.append("=" * 64)

    if best_frame is not None:
        return best_frame, log
    return None, log

def arr_to_b64(arr):
    buf = io.BytesIO()
    plt.imsave(buf, arr, format="png")
    return base64.b64encode(buf.getvalue()).decode()

def radiance_curve_to_b64(knee, strength, source):
    """Visualize the native radiance hook using its DVR2-applied parameters."""
    x = np.linspace(0.0, 1.0, 256)
    t = np.clip((x - knee) / max(1.0 - knee, 1e-6), 0.0, 1.0)
    envelope = t * t * (3.0 - 2.0 * t)
    y = x + (1.0 - x) * envelope * strength

    figure, axis = plt.subplots(figsize=(4.6, 2.8), dpi=120)
    figure.patch.set_facecolor("#1a1a2e")
    axis.set_facecolor("#10131a")
    axis.plot(x, x, color="#8b96a8", linewidth=1.5, label="Spline output")
    axis.plot(x, y, color="#f05a47", linewidth=2.0, label="Spline + radiance")
    axis.axvline(knee, color="#f2c14e", linewidth=1.0, linestyle="--")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Post-spline luma", color="#cbd5e1", fontsize=8)
    axis.set_ylabel("Output luma", color="#cbd5e1", fontsize=8)
    axis.tick_params(colors="#9ca3af", labelsize=7)
    axis.grid(color="#334155", alpha=0.45, linewidth=0.5)
    axis.legend(loc="upper left", frameon=False, labelcolor="#e5e7eb", fontsize=7)
    axis.set_title(f"Native radiance: {source}, knee={knee:.3f}, strength={strength:.3f}",
                   color="#e5e7eb", fontsize=8)
    figure.tight_layout(pad=0.8)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", facecolor=figure.get_facecolor())
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode()

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html>
<head>
<title>HDR10 Viewer — Spline vs Contrast Recovery</title>
<style>
  body{background:#0e1117;color:#eee;font-family:monospace;padding:20px;margin:0}
  h2{color:#fff;margin-bottom:16px}
  .row{display:flex;gap:12px;align-items:flex-end;margin-bottom:16px;flex-wrap:wrap}
  label{display:block;font-size:12px;color:#aaa;margin-bottom:4px}
  input[type=text]{width:420px;padding:8px;background:#1e222d;border:1px solid #333;color:#fff;border-radius:4px;font-size:13px}
    input[type=number],select{width:120px;padding:8px;background:#1e222d;border:1px solid #333;color:#fff;border-radius:4px;font-size:13px}
  button{padding:9px 24px;background:#e74c3c;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:14px;font-weight:bold}
  button:disabled{background:#555;cursor:not-allowed}
    .diagnostics{display:grid;grid-template-columns:minmax(0,1fr) minmax(280px,380px);gap:12px;margin-top:16px}
    #log{background:#1a1a2e;padding:12px;border-radius:4px;font-size:12px;white-space:pre-wrap;min-height:60px;color:#9cf}
    #curve-panel{background:#1a1a2e;border-radius:4px;padding:6px;display:flex;align-items:center}
    #curve{width:100%;display:none}
  .imgs{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
  .imgs img{width:100%;border-radius:4px}
  .caption{font-size:11px;color:#888;text-align:center;margin-top:4px}
    @media(max-width:900px){.diagnostics{grid-template-columns:1fr}}
</style>
</head>
<body>
<h2>HDR10 — Spline vs Contrast Recovery</h2>
<div class="row">
  <div><label>Video path</label><input type="text" id="video" value="G:/28.Years.Later.The.Bone.Temple.mkv"></div>
    <div><label>Current frame</label><input type="number" id="frame" value="72500" min="0"></div>
    <div><label>End frame</label><input type="number" id="end-frame" value="72510" min="0"></div>
  <div><label>Target nits</label><input type="number" id="nits" value="50" min="10" max="4000"></div>
    <div><label>Gamma mode</label><select id="gamma-mode"><option value="auto">Auto</option><option value="manual">Manual</option><option value="off">Off</option></select></div>
    <div><label>Manual gamma</label><input type="number" id="gamma" value="1.0" min="0.5" max="1.5" step="0.05"></div>
    <div><label>CR mode</label><select id="cr-mode"><option value="auto">Auto</option><option value="manual">Manual</option><option value="off">Off</option></select></div>
    <div><label>Manual CR strength</label><input type="number" id="cr-strength" value="0.3" min="0" max="0.5" step="0.05"></div>
    <div><label>Fire-pop mode</label><select id="fire-pop-mode"><option value="off">Off</option><option value="manual">Manual</option><option value="auto">Auto</option></select></div>
    <div><label>Manual fire-pop strength</label><input type="number" id="fire-pop" value="1.0" min="0" max="2.0" step="0.1"></div>
    <div><label>Radiance mode</label><select id="radiance-mode"><option value="off">Off</option><option value="auto">Auto</option><option value="manual">Manual</option></select></div>
    <div><label>Manual radiance knee</label><input type="number" id="radiance-knee" value="0.60" min="0.40" max="0.90" step="0.05"></div>
    <div><label>Manual radiance strength</label><input type="number" id="radiance-strength" value="0.30" min="0" max="0.60" step="0.05"></div>
    <div style="width:100%;margin-top:10px;padding-top:10px;border-top:1px solid #444"><label style="color:#f39c12;font-weight:bold">Chroma Vector Tuner</label></div>
    <div><label>Chroma mode</label><select id="chroma-mode"><option value="off">Off</option><option value="auto">Auto</option><option value="manual">Manual</option></select></div>
    <div><label>Neutral boost</label><input type="number" id="chroma-neutral" value="1.20" min="1.00" max="1.50" step="0.05"></div>
    <div><label>Fire boost</label><input type="number" id="chroma-fire" value="1.35" min="1.00" max="1.50" step="0.05"></div>
    <div><label>Knee point</label><input type="number" id="chroma-knee" value="0.55" min="0.10" max="0.90" step="0.05"></div>
    <div><label>Skin protect</label><input type="number" id="chroma-skin" value="0.95" min="0.50" max="1.00" step="0.05"></div>
    <div><button id="forward-btn" onclick="run()">&#9654; Step +1 Frame</button></div>
</div>
<div class="imgs">
  <div><img id="img_spl" src="" style="display:none"><div id="cap_spl" class="caption"></div></div>
  <div><img id="img_l2" src="" style="display:none"><div id="cap_l2" class="caption"></div></div>
</div>
<div class="diagnostics">
    <div id="log">Ready. Left: Spline. Right: native libplacebo contrast recovery with GPU feature extraction and model inference.</div>
    <div id="curve-panel"><img id="curve" alt="Native radiance transfer curve"></div>
</div>
<script>
function run(){
    const forward=document.getElementById('forward-btn'); forward.disabled=true;
    let stopped=false;
  document.getElementById('log').textContent='Starting...';
  ['img_spl','img_l2'].forEach(id=>{document.getElementById(id).style.display='none';});
  fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({video:document.getElementById('video').value,
      frame:parseInt(document.getElementById('frame').value),
            end_frame:parseInt(document.getElementById('end-frame').value),
      nits:parseInt(document.getElementById('nits').value),
    gamma_mode:document.getElementById('gamma-mode').value,
      gamma:parseFloat(document.getElementById('gamma').value),
    cr_mode:document.getElementById('cr-mode').value,
      cr_strength:parseFloat(document.getElementById('cr-strength').value),
    fire_pop_mode:document.getElementById('fire-pop-mode').value,
            fire_pop:parseFloat(document.getElementById('fire-pop').value),
        radiance_mode:document.getElementById('radiance-mode').value,
        radiance_knee:parseFloat(document.getElementById('radiance-knee').value),
        radiance_strength:parseFloat(document.getElementById('radiance-strength').value),
        chroma_mode:document.getElementById('chroma-mode').value,
        chroma_neutral:parseFloat(document.getElementById('chroma-neutral').value),
        chroma_fire:parseFloat(document.getElementById('chroma-fire').value),
        chroma_knee:parseFloat(document.getElementById('chroma-knee').value),
        chroma_skin:parseFloat(document.getElementById('chroma-skin').value)})})
  .then(r=>r.json()).then(d=>{
    document.getElementById('log').textContent=d.log;
        if(d.next_frame !== undefined){
            document.getElementById('frame').value=d.next_frame;
            if(d.stopped){
                stopped=true;
                forward.disabled=true;
            }
        }
    if(d.img_spl){const i=document.getElementById('img_spl');i.src='data:image/png;base64,'+d.img_spl;i.style.display='block';
      document.getElementById('cap_spl').textContent='Spline ('+d.nits+' nits)';}
    if(d.img_l2){const i=document.getElementById('img_l2');i.src='data:image/png;base64,'+d.img_l2;i.style.display='block';
            document.getElementById('cap_l2').textContent='CR (gamma='+d.gamma+' '+d.gamma_source+', CR='+d.cr_strength+' '+d.cr_source+', FP='+d.fire_pop+' '+d.fire_source+', R='+d.radiance_strength+' '+d.radiance_source+')';}
    if(d.curve){const i=document.getElementById('curve');i.src='data:image/png;base64,'+d.curve;i.style.display='block';}
    }).catch(e=>{document.getElementById('log').textContent+='\\nFetch error: '+e;})
    .finally(()=>{
        if(!stopped){ forward.disabled=false; }
    });
}
</script>
</body>
</html>
"""

@app.get("/")
def index():
    return render_template_string(HTML)

@app.post("/run")
def run_pipeline():
    data  = request.json
    video = data["video"]
    frame = int(data["frame"])
    end_frame = int(data.get("end_frame", frame))
    nits  = int(data["nits"])
    gamma_mode = data.get("gamma_mode", "auto")
    gamma = float(data.get("gamma", 1.0))
    cr_mode = data.get("cr_mode", "auto")
    cr_strength = float(data.get("cr_strength", 0.3))
    fire_pop_mode = data.get("fire_pop_mode", "off")
    fire_pop = float(data.get("fire_pop", 1.0))
    radiance_mode = data.get("radiance_mode", "off")
    radiance_knee = float(data.get("radiance_knee", 0.60))
    radiance_strength = float(data.get("radiance_strength", 0.30))
    chroma_mode = data.get("chroma_mode", "off")
    chroma_neutral = float(data.get("chroma_neutral", 1.20))
    chroma_fire = float(data.get("chroma_fire", 1.35))
    chroma_knee = float(data.get("chroma_knee", 0.55))
    chroma_skin = float(data.get("chroma_skin", 0.95))
    log   = []
    L     = lambda msg: (log.append(msg), print(msg, flush=True))

    # Ensure MSYS2 runtime is findable by subprocesses (DLL search path)
    msys_paths = [r"C:\msys64\ucrt64\bin", r"C:\msys64\usr\bin"]
    subprocess_env = os.environ.copy()
    subprocess_env["PATH"] = ";".join(msys_paths) + ";" + subprocess_env.get("PATH", "")

    try:
        # fps
        ffprobe = r"C:\msys64\ucrt64\bin\ffprobe.exe"
        at_end = frame > end_frame
        L(f"video={video!r}  frame={frame}  end={end_frame}  nits={nits}")
        if at_end:
            L("Sequence end reached")
            return jsonify(log="\n".join(log), img_spl=None, img_l2=None,
                           nits=nits, power=0, gamma=0,
                           next_frame=frame, stopped=True)
        frame_indices = [frame]
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", video],
            capture_output=True, text=True, timeout=30, env=subprocess_env)
        L(f"ffprobe stdout={r.stdout.strip()!r} stderr={r.stderr.strip()!r} retcode={r.returncode}")
        tok = r.stdout.strip().split(",")[0]
        if not tok:
            L("ffprobe returned empty — check video path")
            return jsonify(log="\n".join(log), img_spl=None, img_l2=None,
                          nits=nits, power=0, gamma=0)
        fps = float(tok.split("/")[0]) / float(tok.split("/")[1]) if "/" in tok else float(tok)
        range_start = time.perf_counter()
        img_spl = img_cr = frame_info = None
        timing_records = []
        for index, current_frame in enumerate(frame_indices):
            pts = current_frame / fps
            frame_start = time.perf_counter()
            L(f"\nFrame {current_frame}  pts={pts:.3f}s ({pts/60:.1f} min)")
            L(f"Gamma: {gamma_mode}" + (f" ({gamma:.3f})" if gamma_mode == "manual" else ""))
            L(f"CR: {cr_mode}" + (f" ({cr_strength:.3f})" if cr_mode == "manual" else ""))
            L(f"Fire pop: {fire_pop_mode}" + (f" ({fire_pop:.2f}x)" if fire_pop_mode == "manual" else ""))
            L(f"Radiance: {radiance_mode}" + (f" (knee={radiance_knee:.2f} strength={radiance_strength:.2f})" if radiance_mode == "manual" else ""))
            L(f"Chroma: {chroma_mode}" + (f" (neutral={chroma_neutral:.2f} fire={chroma_fire:.2f} knee={chroma_knee:.2f} skin={chroma_skin:.2f})" if chroma_mode == "manual" else ""))

            render_start = time.perf_counter()
            L("Rendering spline and contrast-recovery...")
            img_spl, img_cr, frame_info = render_persistent_pair(
                video, pts, nits, gamma_mode, gamma, cr_mode, cr_strength,
                fire_pop_mode, fire_pop, radiance_mode, radiance_knee,
                radiance_strength, chroma_mode, chroma_neutral, chroma_fire,
                chroma_knee, chroma_skin
            )
            render_ms = (time.perf_counter() - render_start) * 1000.0
            if img_spl is None or img_cr is None:
                L(f"Persistent render FAILED at frame {current_frame}")
                continue
            L(f"Spline OK shape={img_spl.shape}")
            L(f"Contrast recovery OK shape={img_cr.shape}")
            gamma_source = "model" if frame_info["flags"] & 2 else (
                "manual" if frame_info["flags"] & 4 else "fallback")
            cr_source = "manual" if frame_info["flags"] & 64 else (
                "off" if frame_info["flags"] & 128 else "auto")
            fire_source = "manual" if frame_info["flags"] & 256 else (
                "auto fallback" if frame_info["flags"] & 1024 else "off")
            radiance_source = "manual" if frame_info["flags"] & 4096 else (
                "auto" if frame_info["flags"] & 2048 else "off")
            L("Applied by native libplacebo (DVR2 readback):")
            L("  gamma={:.3f} source={}  CR={:.3f} source={}  "
              "fire-pop={:.2f} source={}".format(
                frame_info["gamma"], gamma_source, frame_info["cr_strength"],
                cr_source, frame_info["fire_pop_strength"], fire_source))
            L("  radiance knee={:.3f} strength={:.3f} source={}".format(
                frame_info["radiance_knee"], frame_info["radiance_strength"],
                radiance_source))
            L("  L1 max={:.4f} avg={:.4f}  L2 power={:.0f}".format(
                frame_info["l1_max_pq"], frame_info["l1_avg_pq"],
                frame_info["l2_power"]))
            L("  chroma neutral={:.2f} fire={:.2f} knee={:.2f} skin={:.2f}".format(
                frame_info["chroma_neutral_boost"], frame_info["chroma_fire_boost"],
                frame_info["chroma_knee"], frame_info["chroma_skin_protect"]))
            total_ms = (time.perf_counter() - frame_start) * 1000.0
            L(f"Frame {current_frame} complete in {total_ms:.1f} ms")
            record = {
                "frame": current_frame, "pts": round(pts, 6),
                "render_pair_ms": round(render_ms, 3),
                "total_ms": round(total_ms, 3), "direction": "forward",
                "width": W, "height": H, "fire_pop": fire_pop,
                "cr_strength": frame_info["cr_strength"],
                "gamma": frame_info["gamma"],
                "native_flags": frame_info["flags"],
                "gamma_mode": gamma_source,
                "cr_mode": cr_source,
                "fire_pop_mode": fire_source,
                "radiance_mode": radiance_source,
                "radiance_knee": frame_info["radiance_knee"],
                "radiance_strength": frame_info["radiance_strength"],
                "l1_max_pq": frame_info["l1_max_pq"],
                "l1_avg_pq": frame_info["l1_avg_pq"],
            }
            timing_records.append(record)
            with TIMING_LOG.open("a", encoding="utf-8") as timing_file:
                timing_file.write(json.dumps(record) + "\n")

        L(f"\nFrame complete in {(time.perf_counter() - range_start):.3f} s")
        if img_spl is None or img_cr is None:
            L("No complete frame rendered")
            return jsonify(log="\n".join(log), img_spl=None, img_l2=None,
                     nits=nits, power=0, gamma=frame_info["gamma"] if frame_info else gamma,
                   next_frame=frame, stopped=False)
        L("DONE")
        all_log = log
        next_frame = frame + 1
        stopped = next_frame > end_frame
        curve = radiance_curve_to_b64(
            frame_info["radiance_knee"], frame_info["radiance_strength"],
            radiance_source)
        return jsonify(
            log         = "\n".join(all_log),
            img_spl     = arr_to_b64(img_spl),
            img_l2      = arr_to_b64(img_cr),
            nits        = nits,
            gamma       = round(frame_info["gamma"], 3),
            gamma_source= gamma_source,
            cr_strength = round(frame_info["cr_strength"], 3),
            cr_source   = cr_source,
            fire_pop    = round(frame_info["fire_pop_strength"], 2),
            fire_source = fire_source,
            radiance_knee = round(frame_info["radiance_knee"], 3),
            radiance_strength = round(frame_info["radiance_strength"], 3),
            radiance_source = radiance_source,
            curve       = curve,
            timing      = timing_records,
            next_frame  = next_frame,
            stopped     = stopped,
        )
    except Exception:
        L("EXCEPTION:\n" + traceback.format_exc())
        return jsonify(log="\n".join(log), img_spl=None, img_l2=None,
                      nits=nits, power=0, gamma=0)


if __name__ == "__main__":
    import waitress
    print("Starting on http://localhost:8601 (waitress)", flush=True)
    waitress.serve(app, host="0.0.0.0", port=8601, threads=4, channel_timeout=120)
