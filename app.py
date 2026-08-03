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

# Limit PyTorch to half the CPU cores so the laptop stays responsive.
# Change this number if you want faster processing (more cores = more heat/lag).
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


# ── Helpers ───────────────────────────────────────────────────────────────────


def _download_model(model_name: str, url: str) -> Path:
    import urllib.request

    path = MODEL_DIR / f"{model_name}.pth"
    if path.exists():
        return path
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def _hook(n, bs, total):
        if total > 0:
            pct = min(int(n * bs * 100 / total), 100)
            print(f"\rDownloading {model_name}: {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, path, _hook)
    print()
    return path


def _get_upsampler(scale: int):
    if scale not in MODEL_CACHE:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model_name, url = MODEL_URLS[scale]
        model_path = _download_model(model_name, url)
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=scale,
        )
        MODEL_CACHE[scale] = RealESRGANer(
            scale=scale,
            model_path=str(model_path),
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
        )
    return MODEL_CACHE[scale]


# ── Processing ────────────────────────────────────────────────────────────────


def upscale_single(image, scale_choice: str, tile_choice: str):
    if image is None:
        gr.Warning("Please upload an image first.")
        yield gr.update(), gr.update(), ""
        return

    scale = int(scale_choice[0])
    tile = 0 if tile_choice == "None" else int(tile_choice)

    yield gr.update(), gr.update(), "Step 1/3 — Loading model..."
    upsampler = _get_upsampler(scale)
    upsampler.tile = tile

    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    if tile > 0:
        # Calculate total tiles so we can show Tile X/N progress.
        total_tiles = math.ceil(w / tile) * math.ceil(h / tile)

        # Forward hook fires once per tile inside enhance().
        tile_counter = [0]
        def _hook(module, inp, out):
            tile_counter[0] += 1
        hook = upsampler.model.register_forward_hook(_hook)

        # Run enhance() in a background thread so the generator keeps yielding.
        result: list = [None]
        exc_holder: list = [None]
        def _run():
            try:
                result[0], _ = upsampler.enhance(img_bgr, outscale=scale)
            except Exception as e:
                exc_holder[0] = e

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        try:
            while thread.is_alive():
                n = tile_counter[0]
                yield gr.update(), gr.update(), f"Step 2/3 — Tile {n}/{total_tiles}..."
                time.sleep(0.3)
            yield gr.update(), gr.update(), f"Step 2/3 — Tile {total_tiles}/{total_tiles}..."
        finally:
            hook.remove()
        thread.join()

        if exc_holder[0] is not None:
            raise gr.Error(f"Upscaling failed: {exc_holder[0]}")
        output_bgr = result[0]

    else:
        yield gr.update(), gr.update(), f"Step 2/3 — Upscaling {w}x{h} at {scale}x (may take a minute on CPU)..."
        try:
            output_bgr, _ = upsampler.enhance(img_bgr, outscale=scale)
        except Exception as exc:
            raise gr.Error(f"Upscaling failed: {exc}")

    oh, ow = output_bgr.shape[:2]
    yield gr.update(), gr.update(), "Step 3/3 — Saving..."
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmp.name, output_bgr)

    info = f"Input: {w} x {h}   Output: {ow} x {oh}   ({scale}x)"
    yield cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB), tmp.name, info


def upscale_batch(
    input_folder: str,
    output_folder: str,
    scale_choice: str,
    tile_choice: str,
):
    lines: list[str] = []

    def emit(msg: str) -> str:
        lines.append(msg)
        return "\n".join(lines)

    if not input_folder.strip():
        yield "Error: Input folder path is required."
        return
    if not output_folder.strip():
        yield "Error: Output folder path is required."
        return

    src = Path(input_folder.strip())
    dst = Path(output_folder.strip())

    if not src.is_dir():
        yield f"Error: Folder not found: {src}"
        return

    images = sorted(
        p for p in src.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )
    if not images:
        yield f"No supported images found in: {src}"
        return

    dst.mkdir(parents=True, exist_ok=True)
    scale = int(scale_choice[0])
    tile = 0 if tile_choice == "None" else int(tile_choice)

    yield emit("Loading model...")
    try:
        upsampler = _get_upsampler(scale)
    except Exception as exc:
        yield emit(f"Error loading model: {exc}")
        return
    upsampler.tile = tile
    yield emit(f"Model ready. Processing {len(images)} image(s)...\n")

    ok = 0
    for i, img_path in enumerate(images):
        yield emit(f"[{i + 1}/{len(images)}] {img_path.name}")
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

    yield emit(f"\nDone — {ok}/{len(images)} image(s) upscaled successfully.")


# ── Layout ────────────────────────────────────────────────────────────────────


CSS = """
footer { display: none !important; }
.tab-nav button { font-size: 1rem; font-weight: 600; }
"""

with gr.Blocks(title="4K Image Upscaler") as app:
    gr.Markdown(
        "# 4K Image Upscaler\n"
        "AI-powered upscaling using **Real-ESRGAN**. "
        "Model weights are downloaded automatically on first use."
    )

    with gr.Tabs():

        # ── Single image tab ──────────────────────────────────────────────────
        with gr.Tab("Single Image"):
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(
                        label="Input Image",
                        type="pil",
                        height=420,
                    )
                    with gr.Row():
                        scale_radio = gr.Radio(
                            choices=["2x", "4x"],
                            value="4x",
                            label="Scale factor",
                        )
                        tile_drop = gr.Dropdown(
                            choices=["None", "512", "256"],
                            value="None",
                            label="Tile size (reduce if out of memory)",
                        )
                    run_btn = gr.Button("Upscale", variant="primary", size="lg")

                with gr.Column():
                    output_img = gr.Image(
                        label="Upscaled Output",
                        height=420,
                    )
                    res_info = gr.Textbox(
                        label="Resolution",
                        interactive=False,
                        max_lines=1,
                    )
                    download_file = gr.File(label="Download PNG")

            run_btn.click(
                fn=upscale_single,
                inputs=[input_img, scale_radio, tile_drop],
                outputs=[output_img, download_file, res_info],
            )

        # ── Batch tab ─────────────────────────────────────────────────────────
        with gr.Tab("Batch Processing"):
            gr.Markdown(
                "Process all images in a folder at once. "
                "Enter full paths on this machine."
            )
            with gr.Row():
                with gr.Column():
                    in_folder = gr.Textbox(
                        label="Input folder",
                        placeholder=r"C:\Users\you\wallpapers",
                    )
                    out_folder = gr.Textbox(
                        label="Output folder",
                        placeholder=r"C:\Users\you\wallpapers_4k",
                    )
                    with gr.Row():
                        batch_scale = gr.Radio(
                            ["2x", "4x"],
                            value="4x",
                            label="Scale factor",
                        )
                        batch_tile = gr.Dropdown(
                            ["None", "512", "256"],
                            value="None",
                            label="Tile size",
                        )
                    batch_btn = gr.Button("Run Batch", variant="primary", size="lg")

                with gr.Column():
                    batch_log = gr.Textbox(
                        label="Progress log",
                        lines=20,
                        interactive=False,
                        autoscroll=True,
                    )

            batch_btn.click(
                fn=upscale_batch,
                inputs=[in_folder, out_folder, batch_scale, batch_tile],
                outputs=batch_log,
            )


if __name__ == "__main__":
    app.launch(inbrowser=True, theme=gr.themes.Soft(), css=CSS)
