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

_physical_cores = os.cpu_count() or 4
_torch_threads = max(1, _physical_cores // 2)
torch.set_num_threads(_torch_threads)
print(f"Using {_torch_threads}/{_physical_cores} CPU threads for inference.")

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


# ── Tile-counting model wrapper ───────────────────────────────────────────────

class _CountingWrapper(torch.nn.Module):
    """Replaces upsampler.model temporarily to count per-tile forward passes."""
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
        MODEL_CACHE[scale] = RealESRGANer(
            scale=scale, model_path=str(model_path), model=model,
            tile=0, tile_pad=10, pre_pad=0, half=False,
        )
    return MODEL_CACHE[scale]


# ── Processing ────────────────────────────────────────────────────────────────

def _bar(n: int, total: int, width: int = 24) -> str:
    filled = int(width * n / total) if total else 0
    return f"[{'#' * filled}{'-' * (width - filled)}] {n}/{total}"


def upscale_single(image, scale_choice: str, tile_choice: str):
    if image is None:
        gr.Warning("Please upload an image first.")
        yield gr.update(), gr.update(), "Waiting for image..."
        return

    scale = int(scale_choice[0])
    tile = 0 if tile_choice == "None" else int(tile_choice)

    yield gr.update(), gr.update(), "Loading model..."
    upsampler = _get_upsampler(scale)
    upsampler.tile = tile

    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    if tile > 0:
        total_tiles = math.ceil(w / tile) * math.ceil(h / tile)
        tile_counter = [0]

        # Swap in counting wrapper so every tile forward increments counter.
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
                upsampler.model = original_model  # always restore

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        try:
            while thread.is_alive():
                n = tile_counter[0]
                yield gr.update(), gr.update(), f"Upscaling  {_bar(n, total_tiles)}"
                time.sleep(0.25)
            # Final tick
            yield gr.update(), gr.update(), f"Upscaling  {_bar(total_tiles, total_tiles)}"
        finally:
            upsampler.model = original_model  # safety restore if generator is cancelled

        thread.join()
        if exc_holder[0] is not None:
            raise gr.Error(f"Upscaling failed: {exc_holder[0]}")
        output_bgr = result[0]

    else:
        yield gr.update(), gr.update(), f"Upscaling {w}x{h} at {scale}x — please wait..."
        try:
            output_bgr, _ = upsampler.enhance(img_bgr, outscale=scale)
        except Exception as exc:
            raise gr.Error(f"Upscaling failed: {exc}")

    oh, ow = output_bgr.shape[:2]
    yield gr.update(), gr.update(), "Saving output..."
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmp.name, output_bgr)

    yield (cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB),
           tmp.name,
           f"Done    {w} x {h}  ->  {ow} x {oh}  ({scale}x)")


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

    yield emit("Loading model...")
    try:
        upsampler = _get_upsampler(scale)
    except Exception as exc:
        yield emit(f"Error loading model: {exc}"); return
    upsampler.tile = tile
    yield emit(f"Model ready — processing {len(images)} image(s)...\n")

    ok = 0
    for i, img_path in enumerate(images):
        yield emit(f"[{i+1}/{len(images)}] {img_path.name}")
        try:
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img_bgr is None:
                raise ValueError("Could not decode image")
            out_bgr, _ = upsampler.enhance(img_bgr, outscale=scale)
            ext = ".png" if img_path.suffix.lower() == ".webp" else img_path.suffix.lower()
            out_path = dst / (img_path.stem + ext)
            cv2.imwrite(str(out_path), out_bgr)
            yield emit(f"    -> {out_path}")
            ok += 1
        except Exception as exc:
            yield emit(f"    Error: {exc}")

    yield emit(f"\nDone — {ok}/{len(images)} upscaled successfully.")


# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
footer { display: none !important; }

/* ── App shell ── */
.app-header {
    text-align: center;
    padding: 2rem 1rem 1rem;
}
.app-header h1 {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0 0 0.3rem;
}
.app-header p {
    font-size: 0.95rem;
    opacity: 0.6;
    margin: 0;
}

/* ── Image panels ── */
.img-panel { border-radius: 12px !important; overflow: hidden; }

/* ── Settings card ── */
.settings-card {
    border-radius: 12px;
    padding: 1rem 1.25rem !important;
}

/* ── Status ── */
.status-row textarea {
    font-family: ui-monospace, 'Cascadia Code', Consolas, monospace !important;
    font-size: 0.82rem !important;
    border-radius: 8px !important;
    resize: none;
}

/* ── Upscale button ── */
.upscale-btn button {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    height: 3rem !important;
    border-radius: 10px !important;
}

/* ── Batch log ── */
.batch-log textarea {
    font-family: ui-monospace, 'Cascadia Code', Consolas, monospace !important;
    font-size: 0.78rem !important;
    border-radius: 8px !important;
}
"""

with gr.Blocks(title="4K Image Upscaler") as app:

    gr.HTML("""
    <div class="app-header">
        <h1>4K Image Upscaler</h1>
        <p>AI-powered enhancement using Real-ESRGAN &nbsp;&middot;&nbsp;
           Model weights downloaded automatically on first use</p>
    </div>
    """)

    with gr.Tabs():

        # ── Single Image ──────────────────────────────────────────────────────
        with gr.Tab("Single Image"):

            with gr.Row(equal_height=True):
                with gr.Column():
                    input_img = gr.Image(
                        label="Input",
                        type="pil",
                        height=460,
                        elem_classes="img-panel",
                    )
                with gr.Column():
                    output_img = gr.Image(
                        label="Output",
                        height=460,
                        interactive=False,
                        elem_classes="img-panel",
                    )

            with gr.Group(elem_classes="settings-card"):
                with gr.Row():
                    scale_radio = gr.Radio(
                        choices=["2x", "4x"],
                        value="4x",
                        label="Scale factor",
                        scale=1,
                    )
                    tile_drop = gr.Dropdown(
                        choices=["None", "512", "256"],
                        value="None",
                        label="Tile size",
                        info="Lower = less memory used",
                        scale=1,
                    )
                    status_box = gr.Textbox(
                        label="Status",
                        value="Ready",
                        interactive=False,
                        max_lines=1,
                        scale=3,
                        elem_classes="status-row",
                    )

            with gr.Row():
                run_btn = gr.Button(
                    "Upscale",
                    variant="primary",
                    size="lg",
                    scale=3,
                    elem_classes="upscale-btn",
                )
                download_file = gr.File(
                    label="Download PNG",
                    scale=2,
                )

            run_btn.click(
                fn=upscale_single,
                inputs=[input_img, scale_radio, tile_drop],
                outputs=[output_img, download_file, status_box],
            )

        # ── Batch ─────────────────────────────────────────────────────────────
        with gr.Tab("Batch Processing"):
            gr.Markdown("Process all images in a folder. Enter full paths on this machine.")
            with gr.Row():
                with gr.Column(scale=1):
                    in_folder = gr.Textbox(
                        label="Input folder",
                        placeholder=r"C:\Users\you\wallpapers",
                    )
                    out_folder = gr.Textbox(
                        label="Output folder",
                        placeholder=r"C:\Users\you\wallpapers_4k",
                    )
                    with gr.Row():
                        batch_scale = gr.Radio(["2x", "4x"], value="4x", label="Scale factor")
                        batch_tile = gr.Dropdown(
                            ["None", "512", "256"], value="None", label="Tile size",
                        )
                    batch_btn = gr.Button("Run Batch", variant="primary", size="lg")

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
    app.launch(inbrowser=True, theme=gr.themes.Soft(), css=CSS)
