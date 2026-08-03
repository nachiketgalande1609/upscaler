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
    print(f"Device: {_DEVICE_LABEL}")
else:
    _DEVICE = "cpu"
    _USE_HALF = False
    torch.set_num_threads(os.cpu_count() or 8)
    _DEVICE_LABEL = f"CPU  ·  {os.cpu_count()} threads  ·  fp32"
    print(f"Device: {_DEVICE_LABEL}  — install PyTorch CUDA for GPU acceleration")

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


# ── Processing ────────────────────────────────────────────────────────────────

def _pct_html(pct: int, label: str) -> str:
    """Return an HTML progress bar at the given percentage (0-100)."""
    clamped = max(0, min(100, pct))
    return f"""
<div class="prog-wrap">
  <div class="prog-header">
    <span class="prog-label">{label}</span>
    <span class="prog-pct">{clamped}%</span>
  </div>
  <div class="prog-track">
    <div class="prog-fill" style="width:{clamped}%"></div>
  </div>
</div>
"""


def upscale_single(image, scale_choice: str, tile_choice: str):
    if image is None:
        gr.Warning("Please upload an image first.")
        yield gr.update(), gr.update(), _pct_html(0, "Waiting for image…")
        return

    scale = int(scale_choice[0])
    tile = 0 if tile_choice == "None" else int(tile_choice)

    yield gr.update(), gr.update(), _pct_html(5, "Loading model…")
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
                # Map tile progress to 10–95% range
                pct = 10 + int(85 * n / total_tiles) if total_tiles else 50
                yield gr.update(), gr.update(), _pct_html(pct, f"Upscaling… tile {n}/{total_tiles}")
                time.sleep(0.25)
            yield gr.update(), gr.update(), _pct_html(95, "Finalising…")
        finally:
            upsampler.model = original_model
        thread.join()

        if exc_holder[0] is not None:
            raise gr.Error(f"Upscaling failed: {exc_holder[0]}")
        output_bgr = result[0]

    else:
        yield gr.update(), gr.update(), _pct_html(20, f"Upscaling {w}×{h} at {scale}×…")
        try:
            output_bgr, _ = upsampler.enhance(img_bgr, outscale=scale)
        except Exception as exc:
            raise gr.Error(f"Upscaling failed: {exc}")

    oh, ow = output_bgr.shape[:2]
    yield gr.update(), gr.update(), _pct_html(98, "Saving…")
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmp.name, output_bgr)

    yield (
        cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB),
        tmp.name,
        _pct_html(100, f"Done  ·  {w}×{h} → {ow}×{oh}  ({scale}×)"),
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
    yield emit(f"Ready on {_DEVICE.upper()} — processing {len(images)} image(s)…\n")

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
            yield emit(f"    ✓  {out_path}")
            ok += 1
        except Exception as exc:
            yield emit(f"    ✗  Error: {exc}")

    yield emit(f"\n{'✓' if ok == len(images) else '!'} Done — {ok}/{len(images)} upscaled successfully.")


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
/* ── Base / reset ──────────────────────────────────── */
:root {
    --bg:       #0d0d0f;
    --surface:  #141418;
    --border:   #2a2a32;
    --accent:   #7c6ef5;
    --accent2:  #a78bfa;
    --text:     #e8e8f0;
    --muted:    #6b6b80;
    --success:  #4ade80;
    --radius:   14px;
}

/* Force dark background everywhere */
body, .gradio-container, .main, footer {
    background: var(--bg) !important;
    color: var(--text) !important;
}
footer { display: none !important; }

/* Panels */
.gr-box, .gr-form, .gr-group, .wrap {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* Tabs */
.tab-nav button {
    background: transparent !important;
    color: var(--muted) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    font-weight: 600 !important;
    transition: color 0.2s, border-color 0.2s !important;
    padding: 0.6rem 1.2rem !important;
}
.tab-nav button.selected {
    color: var(--accent2) !important;
    border-bottom-color: var(--accent2) !important;
}
.tab-nav { border-bottom: 1px solid var(--border) !important; background: transparent !important; }

/* Inputs & textareas */
input, textarea, select, .gr-textbox textarea {
    background: #1a1a22 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    transition: border-color 0.2s !important;
}
input:focus, textarea:focus {
    border-color: var(--accent) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(124,110,245,0.15) !important;
}

/* Radio / dropdown labels */
label, .gr-radio label, .gr-dropdown label {
    color: var(--text) !important;
    font-size: 0.85rem !important;
}
.gr-radio input[type=radio] { accent-color: var(--accent2) !important; }

/* Image panels */
.img-panel {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
    background: #10101a !important;
    transition: border-color 0.3s !important;
}
.img-panel:hover { border-color: var(--accent) !important; }

/* Primary button */
.upscale-btn button, button.primary, button[variant=primary] {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    height: 3rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s !important;
    box-shadow: 0 4px 20px rgba(124,110,245,0.35) !important;
}
.upscale-btn button:hover, button.primary:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(124,110,245,0.5) !important;
}
.upscale-btn button:active { transform: translateY(0) !important; }

/* Batch log */
.batch-log textarea {
    font-family: ui-monospace, 'Cascadia Code', Consolas, monospace !important;
    font-size: 0.78rem !important;
    line-height: 1.6 !important;
    border-radius: 8px !important;
    background: #0f0f16 !important;
}

/* ── App header ─────────────────────────────────────── */
.app-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    animation: fadeDown 0.5s ease both;
}
@keyframes fadeDown {
    from { opacity: 0; transform: translateY(-14px); }
    to   { opacity: 1; transform: translateY(0); }
}
.app-header h1 {
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    margin: 0 0 0.4rem;
    background: linear-gradient(135deg, #c4b5fd, #818cf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.app-header p {
    font-size: 0.95rem;
    color: var(--muted);
    margin: 0 0 0.75rem;
}
.device-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.72rem;
    font-family: ui-monospace, Consolas, monospace;
    border-radius: 999px;
    padding: 0.25rem 0.9rem;
    border: 1px solid;
    animation: fadeIn 0.8s ease 0.3s both;
}
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

/* ── Progress bar ───────────────────────────────────── */
.prog-wrap {
    padding: 0.6rem 0;
    animation: fadeIn 0.3s ease both;
}
.prog-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
}
.prog-label {
    font-size: 0.82rem;
    color: var(--text);
    font-family: ui-monospace, Consolas, monospace;
}
.prog-pct {
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--accent2);
    font-family: ui-monospace, Consolas, monospace;
    min-width: 3rem;
    text-align: right;
}
.prog-track {
    width: 100%;
    height: 8px;
    background: rgba(255,255,255,0.07);
    border-radius: 999px;
    overflow: hidden;
}
.prog-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--accent), var(--accent2), #c4b5fd);
    background-size: 200% 100%;
    transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    animation: shimmer 2s linear infinite;
}
@keyframes shimmer {
    0%   { background-position: 200% center; }
    100% { background-position: -200% center; }
}

/* Subtle slide-in for images */
.img-panel img { animation: imgReveal 0.4s ease both; }
@keyframes imgReveal {
    from { opacity: 0; transform: scale(0.98); }
    to   { opacity: 1; transform: scale(1); }
}

/* Settings group */
.settings-card {
    border-radius: var(--radius) !important;
    padding: 1rem 1.25rem !important;
    margin-top: 0.5rem !important;
}
"""

# ── Device badge colours ──────────────────────────────────────────────────────
if _DEVICE == "cuda":
    _badge_bg     = "rgba(74,222,128,0.1)"
    _badge_border = "rgba(74,222,128,0.35)"
    _badge_color  = "#4ade80"
    _badge_dot    = "●"
else:
    _badge_bg     = "rgba(251,191,36,0.1)"
    _badge_border = "rgba(251,191,36,0.35)"
    _badge_color  = "#fbbf24"
    _badge_dot    = "●"

# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="4K Upscaler") as app:

    gr.HTML(f"""
    <div class="app-header">
        <h1>4K Image Upscaler</h1>
        <p>AI-powered enhancement using Real-ESRGAN</p>
        <span class="device-badge"
              style="background:{_badge_bg};color:{_badge_color};border-color:{_badge_border}">
            <span>{_badge_dot}</span>
            <span>{_DEVICE_LABEL}</span>
        </span>
    </div>
    """)

    with gr.Tabs():

        # ── Single Image ──────────────────────────────────────────────────────
        with gr.Tab("Single Image"):

            with gr.Row(equal_height=True):
                with gr.Column():
                    input_img = gr.Image(
                        label="Input Image",
                        type="pil",
                        height=440,
                        elem_classes="img-panel",
                    )
                with gr.Column():
                    output_img = gr.Image(
                        label="Output Image",
                        height=440,
                        interactive=False,
                        elem_classes="img-panel",
                    )

            with gr.Group(elem_classes="settings-card"):
                with gr.Row():
                    scale_radio = gr.Radio(
                        choices=["2x", "4x"],
                        value="4x",
                        label="Scale",
                        scale=1,
                    )
                    tile_drop = gr.Dropdown(
                        choices=["None", "512", "256"],
                        value="None",
                        label="Tile size",
                        info="None = full image; 512/256 if VRAM runs out",
                        scale=1,
                    )

            with gr.Row():
                run_btn = gr.Button(
                    "⬆  Upscale Image",
                    variant="primary",
                    size="lg",
                    scale=3,
                    elem_classes="upscale-btn",
                )
                download_file = gr.File(
                    label="Download PNG",
                    scale=2,
                )

            progress_html = gr.HTML(value="", visible=False)

            run_btn.click(
                fn=upscale_single,
                inputs=[input_img, scale_radio, tile_drop],
                outputs=[output_img, download_file, progress_html],
            ).then(
                fn=lambda: gr.update(visible=True),
                outputs=progress_html,
            )

        # ── Batch ─────────────────────────────────────────────────────────────
        with gr.Tab("Batch Processing"):
            gr.Markdown(
                "Process every image in a folder. Enter full paths on this machine.",
                elem_id="batch-desc",
            )
            with gr.Row():
                with gr.Column(scale=1):
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
                        batch_tile  = gr.Dropdown(
                            ["None", "512", "256"], value="None", label="Tile size",
                        )
                    batch_btn = gr.Button("⬆  Run Batch", variant="primary", size="lg")

                with gr.Column(scale=1):
                    batch_log = gr.Textbox(
                        label="Progress log",
                        lines=22,
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
