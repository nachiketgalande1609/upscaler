"""Gradio web UI for the Real-ESRGAN image upscaler."""

from __future__ import annotations

import math
import os
import tempfile
import threading
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

# ── Hardware setup ────────────────────────────────────────────────────────────

if torch.cuda.is_available():
    _DEVICE = "cuda"
    _USE_HALF = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    _gpu_name = torch.cuda.get_device_name(0)
    _vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    _DEVICE_LABEL = f"{_gpu_name}  ·  {_vram_gb:.0f} GB VRAM  ·  fp16"
    _DEVICE_SHORT = f"GPU · {_gpu_name}"
    print(f"Device: {_DEVICE_LABEL}")
else:
    _DEVICE = "cpu"
    _USE_HALF = False
    torch.set_num_threads(os.cpu_count() or 8)
    _DEVICE_LABEL = f"CPU  ·  {os.cpu_count()} threads  ·  fp32"
    _DEVICE_SHORT = f"CPU · {os.cpu_count()} threads"
    print(f"Device: {_DEVICE_LABEL}")

MODEL_CACHE: dict = {}
MODEL_DIR = Path("models")
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}

MODEL_URLS = {
    4: (
        "RealESRGAN_x4plus",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    ),
    2: (
        "RealESRGAN_x2plus",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    ),
}


# ── Tile-counting wrapper ─────────────────────────────────────────────────────

class _CountingWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, counter: list):
        super().__init__()
        self._inner = model
        self._counter = counter

    def forward(self, *args, **kwargs):
        out = self._inner(*args, **kwargs)
        self._counter[0] += 1
        return out


# ── Model helpers ─────────────────────────────────────────────────────────────

def _download_model(model_name: str, url: str) -> Path:
    import urllib.request
    path = MODEL_DIR / f"{model_name}.pth"
    if path.exists():
        return path
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    def _hook(n, bs, total):
        if total > 0:
            print(f"\rDownloading {model_name}: {min(int(n*bs*100/total),100)}%",
                  end="", flush=True)
    urllib.request.urlretrieve(url, path, _hook)
    print()
    return path


def _get_upsampler(scale: int):
    if scale not in MODEL_CACHE:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model_name, url = MODEL_URLS[scale]
        model_path = _download_model(model_name, url)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=scale)
        upsampler = RealESRGANer(
            scale=scale,
            model_path=str(model_path),
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=_USE_HALF,
        )
        MODEL_CACHE[scale] = upsampler
        print(f"Model loaded on {_DEVICE.upper()}" + (" (fp16)" if _USE_HALF else " (fp32)"))
    return MODEL_CACHE[scale]


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _progress_html(pct: int, label: str, done: bool = False) -> str:
    clamped = max(0, min(100, pct))
    fill_class = "prog-fill-done" if done else "prog-fill"
    return f"""
<div class="prog-card">
  <div class="prog-top">
    <span class="prog-label">{label}</span>
    <span class="prog-num {'prog-done' if done else ''}">{clamped}%</span>
  </div>
  <div class="prog-track">
    <div class="{fill_class}" style="width:{clamped}%"></div>
  </div>
</div>"""


def _idle_html() -> str:
    return '<div class="prog-card prog-idle">Ready — upload an image and press Upscale</div>'


def _info_html(w: int, h: int, ow: int, oh: int, scale: int) -> str:
    return f"""
<div class="result-info">
  <div class="ri-item"><span class="ri-k">Input</span><span class="ri-v">{w} × {h}</span></div>
  <div class="ri-sep">→</div>
  <div class="ri-item"><span class="ri-k">Output</span><span class="ri-v">{ow} × {oh}</span></div>
  <div class="ri-badge">{scale}×</div>
</div>"""


# ── Processing ────────────────────────────────────────────────────────────────

def upscale_single(image, scale_choice: str, tile_choice: str):
    if image is None:
        gr.Warning("Upload an image first.")
        yield gr.update(), gr.update(), _idle_html(), ""
        return

    scale = int(scale_choice[0])
    tile = 0 if tile_choice == "None" else int(tile_choice)

    yield gr.update(), gr.update(), _progress_html(5, "Loading model…"), ""
    upsampler = _get_upsampler(scale)
    upsampler.tile_size = tile

    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    if tile > 0:
        total_tiles = math.ceil(w / tile) * math.ceil(h / tile)
        tile_counter = [0]
        original_model = upsampler.model
        upsampler.model = _CountingWrapper(original_model, tile_counter)

        result: list = [None]
        exc_holder: list = [None]

        def _run():
            try:
                result[0], _ = upsampler.enhance(img_bgr, outscale=scale)
            except Exception as e:
                exc_holder[0] = e
            finally:
                upsampler.model = original_model

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        try:
            while thread.is_alive():
                n = tile_counter[0]
                pct = 10 + int(85 * n / total_tiles) if total_tiles else 50
                yield (gr.update(), gr.update(),
                       _progress_html(pct, f"Upscaling… {n} / {total_tiles} tiles"), "")
                time.sleep(0.25)
            yield gr.update(), gr.update(), _progress_html(95, "Finalising…"), ""
        finally:
            upsampler.model = original_model
        thread.join()

        if exc_holder[0] is not None:
            raise gr.Error(f"Upscaling failed: {exc_holder[0]}")
        output_bgr = result[0]
    else:
        yield (gr.update(), gr.update(),
               _progress_html(20, f"Upscaling {w}×{h} at {scale}×…"), "")
        try:
            output_bgr, _ = upsampler.enhance(img_bgr, outscale=scale)
        except Exception as exc:
            raise gr.Error(f"Upscaling failed: {exc}")

    oh, ow = output_bgr.shape[:2]
    yield gr.update(), gr.update(), _progress_html(98, "Saving…"), ""

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmp.name, output_bgr)

    yield (
        cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB),
        tmp.name,
        _progress_html(100, "Complete!", done=True),
        _info_html(w, h, ow, oh, scale),
    )


def upscale_batch(input_folder: str, output_folder: str,
                  scale_choice: str, tile_choice: str):
    lines: list[str] = []
    def emit(msg: str) -> str:
        lines.append(msg)
        return "\n".join(lines)

    if not input_folder.strip():
        yield "Error: Input folder path is required."; return
    if not output_folder.strip():
        yield "Error: Output folder path is required."; return

    src = Path(input_folder.strip())
    dst = Path(output_folder.strip())
    if not src.is_dir():
        yield f"Error: Folder not found: {src}"; return

    images = sorted(p for p in src.iterdir()
                    if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not images:
        yield f"No supported images found in: {src}"; return

    dst.mkdir(parents=True, exist_ok=True)
    scale = int(scale_choice[0])
    tile = 0 if tile_choice == "None" else int(tile_choice)

    yield emit("Loading model…")
    try:
        upsampler = _get_upsampler(scale)
    except Exception as exc:
        yield emit(f"Error loading model: {exc}"); return
    upsampler.tile_size = tile
    yield emit(f"Ready on {_DEVICE.upper()} — {len(images)} image(s) queued\n")

    ok = 0
    for i, img_path in enumerate(images):
        yield emit(f"[{i+1}/{len(images)}]  {img_path.name}")
        try:
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img_bgr is None:
                raise ValueError("Could not decode image")
            out_bgr, _ = upsampler.enhance(img_bgr, outscale=scale)
            ext = ".png" if img_path.suffix.lower() == ".webp" else img_path.suffix.lower()
            out_path = dst / (img_path.stem + ext)
            cv2.imwrite(str(out_path), out_bgr)
            yield emit(f"    ✓  saved → {out_path.name}")
            ok += 1
        except Exception as exc:
            yield emit(f"    ✗  {exc}")

    status = "✓ All done" if ok == len(images) else f"⚠ {ok}/{len(images)} succeeded"
    yield emit(f"\n{status} — {ok} image(s) upscaled successfully.")


# ── CSS ───────────────────────────────────────────────────────────────────────

_is_gpu = _DEVICE == "cuda"
_dot_color  = "#4ade80" if _is_gpu else "#fbbf24"
_dot_bg     = "rgba(74,222,128,0.12)" if _is_gpu else "rgba(251,191,36,0.12)"
_dot_border = "rgba(74,222,128,0.3)"  if _is_gpu else "rgba(251,191,36,0.3)"

CSS = """
/* ─── Tokens ───────────────────────────────────────────────── */
:root {
    --bg:         #09090c;
    --panel:      #111115;
    --card:       #16161c;
    --border:     #252530;
    --border-hi:  #3a3a50;
    --accent:     #7c6ef5;
    --accent-hi:  #a78bfa;
    --accent-glow:rgba(124,110,245,0.25);
    --text:       #dde1f0;
    --sub:        #7878a0;
    --success:    #4ade80;
    --r:          14px;
    --r-sm:       8px;
}

/* ─── Global reset ──────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container, .main, .wrap, .app {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}
footer { display: none !important; }
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; padding: 0 1.5rem 3rem !important; }

/* ─── Scrollbar ─────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 99px; }

/* ─── Panels & groups ───────────────────────────────────────── */
.gr-box, .gr-form, .gr-group, .gr-panel {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
}

/* ─── Tabs ──────────────────────────────────────────────────── */
.tab-nav {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 0 !important;
    margin-bottom: 1.5rem !important;
}
.tab-nav button {
    background: transparent !important;
    color: var(--sub) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    padding: 0.7rem 1.4rem !important;
    transition: color 0.2s, border-color 0.2s !important;
    letter-spacing: 0.01em !important;
}
.tab-nav button.selected {
    color: var(--accent-hi) !important;
    border-bottom-color: var(--accent-hi) !important;
}
.tab-nav button:hover:not(.selected) { color: var(--text) !important; }

/* ─── Inputs ────────────────────────────────────────────────── */
input[type=text], textarea {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-sm) !important;
    color: var(--text) !important;
    font-size: 0.875rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
input[type=text]:focus, textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
    outline: none !important;
}
label span, .gr-form label {
    color: var(--sub) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

/* ─── Radio & Dropdown ──────────────────────────────────────── */
.gr-radio, .gr-dropdown { background: transparent !important; }
.gr-radio label { text-transform: none !important; font-size: 0.85rem !important; font-weight: 400 !important; color: var(--text) !important; letter-spacing: 0 !important; }
input[type=radio] { accent-color: var(--accent-hi) !important; }
.gr-dropdown select, select {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-sm) !important;
    color: var(--text) !important;
}

/* ─── Sidebar card ──────────────────────────────────────────── */
.sidebar {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}
.ctrl-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1.25rem 1.25rem 1rem;
}
.ctrl-card-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--sub);
    margin-bottom: 0.85rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
}

/* ─── Image panels ──────────────────────────────────────────── */
.img-well {
    position: relative;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--r);
    overflow: hidden;
    transition: border-color 0.25s;
    min-height: 420px;
    display: flex;
    flex-direction: column;
}
.img-well:hover { border-color: var(--border-hi); }
.img-well-label {
    position: absolute;
    top: 10px; left: 12px;
    z-index: 10;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--sub);
    background: rgba(9,9,12,0.7);
    padding: 0.2rem 0.6rem;
    border-radius: 99px;
    backdrop-filter: blur(6px);
    border: 1px solid var(--border);
}

/* Gradio image component inside img-well */
.img-well .gr-image, .img-well > div { height: 100% !important; }

/* ─── Upscale button ────────────────────────────────────────── */
.upscale-btn > button {
    width: 100% !important;
    background: linear-gradient(135deg, #6d5ef0, #9d7df5, #c4a8fc) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--r-sm) !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    height: 2.8rem !important;
    cursor: pointer !important;
    box-shadow: 0 0 0 0 var(--accent-glow) !important;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.25s !important;
    position: relative;
    overflow: hidden;
}
.upscale-btn > button::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, transparent 40%, rgba(255,255,255,0.12) 100%);
    pointer-events: none;
}
.upscale-btn > button:hover {
    opacity: 0.92 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 24px var(--accent-glow) !important;
}
.upscale-btn > button:active { transform: translateY(0) !important; }

/* ─── Progress card ─────────────────────────────────────────── */
.prog-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--r-sm);
    padding: 0.9rem 1rem;
}
.prog-idle {
    font-size: 0.8rem;
    color: var(--sub);
    text-align: center;
}
.prog-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.55rem;
}
.prog-label {
    font-size: 0.82rem;
    color: var(--text);
    font-family: ui-monospace, Consolas, monospace;
}
.prog-num {
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--accent-hi);
    font-family: ui-monospace, Consolas, monospace;
    min-width: 3rem;
    text-align: right;
}
.prog-done { color: var(--success) !important; }
.prog-track {
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 99px;
    overflow: hidden;
}
.prog-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--accent), var(--accent-hi), #c4b5fd);
    background-size: 300% 100%;
    transition: width 0.35s cubic-bezier(0.4,0,0.2,1);
    animation: shimmer 2.5s linear infinite;
}
.prog-fill-done {
    height: 100%;
    border-radius: 99px;
    background: var(--success);
    transition: width 0.35s cubic-bezier(0.4,0,0.2,1);
}
@keyframes shimmer {
    0%   { background-position: 200% center; }
    100% { background-position: -200% center; }
}

/* ─── Result info bar ───────────────────────────────────────── */
.result-info {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--r-sm);
    padding: 0.65rem 1rem;
    font-size: 0.82rem;
    animation: fadeUp 0.4s ease both;
}
@keyframes fadeUp { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }
.ri-item { display: flex; flex-direction: column; gap: 0.1rem; }
.ri-k { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--sub); font-weight: 700; }
.ri-v { font-family: ui-monospace, Consolas, monospace; color: var(--text); font-size: 0.84rem; }
.ri-sep { color: var(--sub); font-size: 1rem; padding: 0 0.25rem; }
.ri-badge {
    margin-left: auto;
    background: rgba(124,110,245,0.15);
    border: 1px solid rgba(124,110,245,0.3);
    color: var(--accent-hi);
    font-size: 0.78rem;
    font-weight: 700;
    border-radius: 99px;
    padding: 0.15rem 0.65rem;
}

/* ─── App header ────────────────────────────────────────────── */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.75rem 0 1.25rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
    animation: fadeDown 0.45s ease both;
}
@keyframes fadeDown { from { opacity:0; transform:translateY(-10px); } to { opacity:1; transform:translateY(0); } }
.app-logo { display: flex; align-items: center; gap: 0.75rem; }
.app-logo-icon {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #6d5ef0, #a78bfa);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    box-shadow: 0 4px 16px rgba(124,110,245,0.35);
}
.app-logo-text h1 {
    font-size: 1.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0;
    color: var(--text);
}
.app-logo-text p {
    font-size: 0.72rem;
    color: var(--sub);
    margin: 0;
}
.device-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.73rem;
    font-family: ui-monospace, Consolas, monospace;
    border-radius: 99px;
    padding: 0.35rem 0.9rem;
    border: 1px solid;
    animation: fadeIn 0.6s ease 0.2s both;
}
.device-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}
@keyframes fadeIn { from { opacity:0; } to { opacity:1; } }

/* ─── Download file widget ──────────────────────────────────── */
.gr-file { background: var(--card) !important; border-color: var(--border) !important; border-radius: var(--r-sm) !important; }

/* ─── Batch log ─────────────────────────────────────────────── */
.batch-log textarea {
    font-family: ui-monospace, 'Cascadia Code', Consolas, monospace !important;
    font-size: 0.78rem !important;
    line-height: 1.65 !important;
    background: var(--panel) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* ─── Markdown body ─────────────────────────────────────────── */
.gr-markdown p { color: var(--sub) !important; font-size: 0.85rem !important; }
"""

# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="4K Upscaler") as app:

    # Header
    gr.HTML(f"""
    <div class="app-header">
        <div class="app-logo">
            <div class="app-logo-icon">✦</div>
            <div class="app-logo-text">
                <h1>4K Upscaler</h1>
                <p>Real-ESRGAN · AI image enhancement</p>
            </div>
        </div>
        <span class="device-pill"
              style="background:{_dot_bg};color:{_dot_color};border-color:{_dot_border}">
            <span class="device-dot" style="background:{_dot_color}"></span>
            {_DEVICE_SHORT}
        </span>
    </div>
    """)

    with gr.Tabs():

        # ── Single Image ──────────────────────────────────────────────────────
        with gr.Tab("Single Image"):

            with gr.Row(equal_height=False):

                # Left: controls sidebar
                with gr.Column(scale=1, min_width=240):
                    gr.HTML('<div class="ctrl-card-title" style="margin-top:0.25rem">Settings</div>')

                    scale_radio = gr.Radio(
                        choices=["2x", "4x"],
                        value="4x",
                        label="Scale factor",
                    )

                    tile_drop = gr.Dropdown(
                        choices=["None", "512", "256"],
                        value="None",
                        label="Tile size",
                        info="None = full image (fast on GPU). Use 512 if VRAM overflows.",
                    )

                    run_btn = gr.Button(
                        "Upscale",
                        variant="primary",
                        size="lg",
                        elem_classes="upscale-btn",
                    )

                    progress_html = gr.HTML(
                        value='<div class="prog-card prog-idle">Ready — upload an image and press Upscale</div>',
                    )

                    result_info = gr.HTML(value="")

                    download_file = gr.File(label="Download result")

                # Right: before / after images
                with gr.Column(scale=3):
                    with gr.Row(equal_height=True):
                        with gr.Column(elem_classes="img-well"):
                            gr.HTML('<div class="img-well-label">Before</div>')
                            input_img = gr.Image(
                                type="pil",
                                show_label=False,
                                height=500,
                                container=False,
                            )
                        with gr.Column(elem_classes="img-well"):
                            gr.HTML('<div class="img-well-label">After</div>')
                            output_img = gr.Image(
                                show_label=False,
                                height=500,
                                interactive=False,
                                container=False,
                            )

            run_btn.click(
                fn=upscale_single,
                inputs=[input_img, scale_radio, tile_drop],
                outputs=[output_img, download_file, progress_html, result_info],
            )

        # ── Batch ─────────────────────────────────────────────────────────────
        with gr.Tab("Batch Processing"):

            with gr.Row():

                with gr.Column(scale=1, min_width=300):
                    gr.Markdown("Process all images in a folder. Enter full paths on this machine.")

                    in_folder = gr.Textbox(
                        label="Input folder",
                        placeholder=r"C:\Users\you\photos",
                    )
                    out_folder = gr.Textbox(
                        label="Output folder",
                        placeholder=r"C:\Users\you\photos_4k",
                    )
                    with gr.Row():
                        batch_scale = gr.Radio(["2x", "4x"], value="4x", label="Scale")
                        batch_tile  = gr.Dropdown(["None", "512", "256"], value="None", label="Tile size")

                    batch_btn = gr.Button("Run Batch", variant="primary", size="lg",
                                          elem_classes="upscale-btn")

                with gr.Column(scale=2):
                    batch_log = gr.Textbox(
                        label="Log",
                        lines=24,
                        interactive=False,
                        autoscroll=True,
                        elem_classes="batch-log",
                    )

            batch_btn.click(
                fn=upscale_batch,
                inputs=[in_folder, out_folder, batch_scale, batch_tile],
                outputs=batch_log,
            )


if __name__ == "__main__":
    app.launch(inbrowser=True, theme=gr.themes.Base(), css=CSS)
