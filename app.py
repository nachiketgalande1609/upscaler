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
    _DEVICE_LABEL = f"{_gpu_name} · {_vram_gb:.0f} GB VRAM · fp16"
    _DEVICE_SHORT = f"GPU · {_gpu_name}"
    _DOT = "#4ade80"; _DOT_BG = "rgba(74,222,128,0.12)"; _DOT_BD = "rgba(74,222,128,0.3)"
    print(f"Device: {_DEVICE_LABEL}")
else:
    _DEVICE = "cpu"
    _USE_HALF = False
    torch.set_num_threads(os.cpu_count() or 8)
    _DEVICE_LABEL = f"CPU · {os.cpu_count()} threads · fp32"
    _DEVICE_SHORT = f"CPU · {os.cpu_count()} threads"
    _DOT = "#fbbf24"; _DOT_BG = "rgba(251,191,36,0.12)"; _DOT_BD = "rgba(251,191,36,0.3)"
    print(f"Device: {_DEVICE_LABEL}")

MODEL_CACHE: dict = {}
MODEL_DIR = Path("models")
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}
MODEL_URLS = {
    4: ("RealESRGAN_x4plus",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"),
    2: ("RealESRGAN_x2plus",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"),
}


class _CountingWrapper(torch.nn.Module):
    def __init__(self, model, counter):
        super().__init__()
        self._inner = model
        self._counter = counter
    def forward(self, *a, **kw):
        out = self._inner(*a, **kw)
        self._counter[0] += 1
        return out


def _download_model(model_name, url):
    import urllib.request
    path = MODEL_DIR / f"{model_name}.pth"
    if path.exists():
        return path
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    def _hook(n, bs, total):
        if total > 0:
            print(f"\rDownloading {model_name}: {min(int(n*bs*100/total),100)}%", end="", flush=True)
    urllib.request.urlretrieve(url, path, _hook)
    print()
    return path


def _get_upsampler(scale):
    if scale not in MODEL_CACHE:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        name, url = MODEL_URLS[scale]
        path = _download_model(name, url)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
        MODEL_CACHE[scale] = RealESRGANer(
            scale=scale, model_path=str(path), model=model,
            tile=0, tile_pad=10, pre_pad=0, half=_USE_HALF)
    return MODEL_CACHE[scale]


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _prog(pct, label, done=False):
    p = max(0, min(100, pct))
    bar_style = (
        'background:#4ade80;transition:width .35s ease'
        if done else
        'background:linear-gradient(90deg,#7c6ef5,#a78bfa,#c4b5fd);background-size:300% 100%;'
        'animation:shimmer 2s linear infinite;transition:width .35s ease'
    )
    num_color = "#4ade80" if done else "#a78bfa"
    return f"""
<div style="background:#16161c;border:1px solid #252530;border-radius:10px;padding:12px 14px;margin-top:4px">
  <div style="display:flex;justify-content:space-between;margin-bottom:8px;font-size:13px;font-family:ui-monospace,Consolas,monospace">
    <span style="color:#dde1f0">{label}</span>
    <span style="color:{num_color};font-weight:700">{p}%</span>
  </div>
  <div style="height:6px;background:rgba(255,255,255,0.07);border-radius:99px;overflow:hidden">
    <div style="height:100%;width:{p}%;border-radius:99px;{bar_style}"></div>
  </div>
</div>"""


def _idle():
    return '<div style="background:#16161c;border:1px solid #252530;border-radius:10px;padding:12px 14px;margin-top:4px;font-size:12px;color:#6b6b80;text-align:center">Ready — upload an image and click Upscale</div>'


def _info(w, h, ow, oh, scale):
    return f"""
<div style="display:flex;align-items:center;gap:12px;background:#16161c;border:1px solid #252530;
            border-radius:10px;padding:10px 14px;margin-top:6px;font-size:12px">
  <div><div style="color:#6b6b80;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em">Input</div>
       <div style="font-family:ui-monospace,Consolas,monospace;color:#dde1f0">{w} × {h}</div></div>
  <div style="color:#6b6b80;font-size:16px">→</div>
  <div><div style="color:#6b6b80;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em">Output</div>
       <div style="font-family:ui-monospace,Consolas,monospace;color:#dde1f0">{ow} × {oh}</div></div>
  <div style="margin-left:auto;background:rgba(124,110,245,.15);border:1px solid rgba(124,110,245,.3);
              color:#a78bfa;font-size:11px;font-weight:700;border-radius:99px;padding:2px 10px">{scale}×</div>
</div>"""


# ── Processing ────────────────────────────────────────────────────────────────

def upscale_single(image, scale_choice, tile_choice):
    if image is None:
        gr.Warning("Upload an image first.")
        yield gr.update(), gr.update(), _idle(), ""
        return

    scale = int(scale_choice[0])
    tile = 0 if tile_choice == "None" else int(tile_choice)

    yield gr.update(), gr.update(), _prog(5, "Loading model…"), ""
    upsampler = _get_upsampler(scale)
    upsampler.tile_size = tile

    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    if tile > 0:
        total = math.ceil(w / tile) * math.ceil(h / tile)
        counter = [0]
        orig = upsampler.model
        upsampler.model = _CountingWrapper(orig, counter)
        result, exc_h = [None], [None]

        def _run():
            try:
                result[0], _ = upsampler.enhance(img_bgr, outscale=scale)
            except Exception as e:
                exc_h[0] = e
            finally:
                upsampler.model = orig

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        try:
            while t.is_alive():
                n = counter[0]
                pct = 10 + int(85 * n / total) if total else 50
                yield gr.update(), gr.update(), _prog(pct, f"Upscaling… {n}/{total} tiles"), ""
                time.sleep(0.25)
            yield gr.update(), gr.update(), _prog(95, "Finalising…"), ""
        finally:
            upsampler.model = orig
        t.join()
        if exc_h[0]:
            raise gr.Error(f"Failed: {exc_h[0]}")
        out_bgr = result[0]
    else:
        yield gr.update(), gr.update(), _prog(20, f"Upscaling {w}×{h} → {scale}×…"), ""
        try:
            out_bgr, _ = upsampler.enhance(img_bgr, outscale=scale)
        except Exception as e:
            raise gr.Error(f"Failed: {e}")

    oh, ow = out_bgr.shape[:2]
    yield gr.update(), gr.update(), _prog(98, "Saving…"), ""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmp.name, out_bgr)

    yield (
        cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB),
        tmp.name,
        _prog(100, "Done!", done=True),
        _info(w, h, ow, oh, scale),
    )


def upscale_batch(in_folder, out_folder, scale_choice, tile_choice):
    lines = []
    def emit(m):
        lines.append(m); return "\n".join(lines)

    if not in_folder.strip(): yield "Error: input folder required"; return
    if not out_folder.strip(): yield "Error: output folder required"; return
    src = Path(in_folder.strip()); dst = Path(out_folder.strip())
    if not src.is_dir(): yield f"Error: not found: {src}"; return

    imgs = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not imgs: yield f"No supported images in {src}"; return

    dst.mkdir(parents=True, exist_ok=True)
    scale = int(scale_choice[0])
    tile = 0 if tile_choice == "None" else int(tile_choice)

    yield emit("Loading model…")
    try: up = _get_upsampler(scale)
    except Exception as e: yield emit(f"Error: {e}"); return
    up.tile_size = tile
    yield emit(f"Ready on {_DEVICE.upper()} — {len(imgs)} image(s)\n")

    ok = 0
    for i, p in enumerate(imgs):
        yield emit(f"[{i+1}/{len(imgs)}]  {p.name}")
        try:
            bgr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if bgr is None: raise ValueError("Could not read file")
            out, _ = up.enhance(bgr, outscale=scale)
            ext = ".png" if p.suffix.lower() == ".webp" else p.suffix.lower()
            op = dst / (p.stem + ext)
            cv2.imwrite(str(op), out)
            yield emit(f"    ✓  {op.name}"); ok += 1
        except Exception as e:
            yield emit(f"    ✗  {e}")

    yield emit(f"\n{'✓' if ok==len(imgs) else '⚠'} {ok}/{len(imgs)} done.")


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@keyframes shimmer { 0%{background-position:200% center} 100%{background-position:-200% center} }
@keyframes fadeIn  { from{opacity:0} to{opacity:1} }
@keyframes slideUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }

/* Force dark base */
body { background:#09090c !important; }
.gradio-container { background:#09090c !important; max-width:100% !important; padding:0 !important; }
.main { background:#09090c !important; }
footer { display:none !important; }

/* All panels */
.gr-form, .gr-box, .wrap { background:#111115 !important; }

/* Tabs */
.tab-nav { background:#09090c !important; border-bottom:1px solid #252530 !important; padding:0 24px !important; }
.tab-nav button {
    background:transparent !important; color:#555570 !important;
    border:none !important; border-bottom:2px solid transparent !important;
    border-radius:0 !important; font-size:13px !important; font-weight:600 !important;
    padding:10px 18px !important; transition:color .2s,border-color .2s !important;
}
.tab-nav button.selected { color:#a78bfa !important; border-bottom-color:#a78bfa !important; }
.tab-nav button:hover:not(.selected) { color:#aaa !important; }

/* Input fields */
.gr-textbox textarea, input[type=text] {
    background:#1a1a22 !important; border:1px solid #2a2a38 !important;
    color:#dde1f0 !important; border-radius:8px !important;
    font-size:13px !important; transition:border-color .2s,box-shadow .2s !important;
}
.gr-textbox textarea:focus, input[type=text]:focus {
    border-color:#7c6ef5 !important;
    box-shadow:0 0 0 3px rgba(124,110,245,.2) !important;
}

/* Labels */
label > span, .gr-form label > span { color:#888 !important; font-size:11px !important; font-weight:600 !important; letter-spacing:.05em !important; text-transform:uppercase !important; }

/* Radio */
.gr-radio { background:transparent !important; }
.gr-radio label { color:#ccc !important; font-size:13px !important; font-weight:400 !important; text-transform:none !important; letter-spacing:0 !important; }
input[type=radio] { accent-color:#a78bfa !important; }

/* Dropdown */
.gr-dropdown > label > div { background:#1a1a22 !important; border:1px solid #2a2a38 !important; color:#dde1f0 !important; border-radius:8px !important; }

/* Primary button — target every possible Gradio button selector */
button.primary, button[variant="primary"], .gr-button-primary,
#single-run-btn button, #batch-run-btn button {
    background: linear-gradient(135deg,#5b4de0,#7c6ef5,#a78bfa) !important;
    color: #fff !important; border: none !important;
    border-radius: 9px !important; font-size: 14px !important;
    font-weight: 700 !important; letter-spacing: .04em !important;
    height: 44px !important; width: 100% !important;
    cursor: pointer !important;
    box-shadow: 0 4px 20px rgba(124,110,245,.35) !important;
    transition: opacity .2s, transform .15s, box-shadow .2s !important;
}
button.primary:hover, button[variant="primary"]:hover {
    opacity:.88 !important; transform:translateY(-1px) !important;
    box-shadow:0 6px 28px rgba(124,110,245,.5) !important;
}

/* Image panels */
#input-panel, #output-panel {
    background:#111115 !important;
    border:1px solid #252530 !important;
    border-radius:14px !important;
    overflow:hidden !important;
    transition:border-color .25s !important;
}
#input-panel:hover, #output-panel:hover { border-color:#3a3a52 !important; }

/* Sidebar */
#sidebar { background:#111115; border:1px solid #252530; border-radius:14px; padding:20px; }

/* File widget */
.gr-file { background:#1a1a22 !important; border:1px solid #2a2a38 !important; border-radius:8px !important; }
.gr-file label > span { color:#888 !important; font-size:11px !important; }

/* Batch log */
#batch-log textarea {
    font-family:ui-monospace,Consolas,monospace !important;
    font-size:12px !important; line-height:1.7 !important;
    background:#0f0f16 !important; border:1px solid #252530 !important;
    color:#c8c8e0 !important; border-radius:10px !important;
}

/* Markdown */
.gr-markdown, .gr-markdown p { color:#555570 !important; font-size:13px !important; }
"""

# ── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="4K Upscaler", theme=gr.themes.Base(), css=CSS) as app:

    # ── Top bar ───────────────────────────────────────────────────────────────
    gr.HTML(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
                padding:18px 28px 14px;border-bottom:1px solid #1e1e28;
                background:#09090c;animation:fadeIn .5s ease both">
      <div style="display:flex;align-items:center;gap:12px">
        <div style="width:38px;height:38px;background:linear-gradient(135deg,#5b4de0,#a78bfa);
                    border-radius:10px;display:flex;align-items:center;justify-content:center;
                    font-size:18px;box-shadow:0 4px 16px rgba(124,110,245,.4)">✦</div>
        <div>
          <div style="font-size:17px;font-weight:800;color:#e8e8f4;letter-spacing:-.02em">4K Upscaler</div>
          <div style="font-size:11px;color:#555570">Real-ESRGAN · AI image enhancement</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:7px;background:{_DOT_BG};
                  border:1px solid {_DOT_BD};border-radius:99px;padding:5px 14px;
                  font-size:11px;font-family:ui-monospace,Consolas,monospace;color:{_DOT};
                  animation:fadeIn .6s ease .2s both">
        <span style="width:6px;height:6px;background:{_DOT};border-radius:50%;
                     animation:pulse 2s ease-in-out infinite"></span>
        {_DEVICE_SHORT}
      </div>
    </div>
    <style>
    @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.35}} }}
    </style>
    """)

    with gr.Tabs():

        # ── Single Image tab ──────────────────────────────────────────────────
        with gr.Tab("  Single Image  "):
            with gr.Row(equal_height=False):

                # Sidebar
                with gr.Column(scale=1, min_width=260, elem_id="sidebar"):
                    gr.HTML('<div style="font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#444;margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid #1e1e28">Settings</div>')

                    scale_radio = gr.Radio(
                        choices=["2x", "4x"], value="4x", label="Scale factor",
                    )
                    gr.HTML('<div style="height:4px"></div>')
                    tile_drop = gr.Dropdown(
                        choices=["None", "512", "256"], value="None", label="Tile size",
                        info="None = full image. Use 512/256 if VRAM overflows.",
                    )
                    gr.HTML('<div style="height:12px"></div>')

                    run_btn = gr.Button("Upscale", variant="primary", elem_id="single-run-btn")

                    gr.HTML('<div style="height:8px"></div>')
                    progress_html = gr.HTML(value=_idle())
                    result_info   = gr.HTML(value="")

                    gr.HTML('<div style="height:8px"></div>')
                    download_file = gr.File(label="Download result")

                # Images — Before / After
                with gr.Column(scale=3):
                    with gr.Row(equal_height=True):
                        with gr.Column(elem_id="input-panel"):
                            gr.HTML('<div style="padding:10px 14px 4px;font-size:10px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:#555570">Before</div>')
                            input_img = gr.Image(
                                type="pil", show_label=False,
                                height=520, container=False,
                            )
                        with gr.Column(elem_id="output-panel"):
                            gr.HTML('<div style="padding:10px 14px 4px;font-size:10px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:#555570">After</div>')
                            output_img = gr.Image(
                                show_label=False, height=520,
                                interactive=False, container=False,
                            )

            run_btn.click(
                fn=upscale_single,
                inputs=[input_img, scale_radio, tile_drop],
                outputs=[output_img, download_file, progress_html, result_info],
            )

        # ── Batch tab ─────────────────────────────────────────────────────────
        with gr.Tab("  Batch Processing  "):
            with gr.Row():
                with gr.Column(scale=1, min_width=300, elem_id="sidebar"):
                    gr.HTML('<div style="font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#444;margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid #1e1e28">Batch Settings</div>')
                    in_folder  = gr.Textbox(label="Input folder",  placeholder=r"C:\Users\you\photos")
                    out_folder = gr.Textbox(label="Output folder", placeholder=r"C:\Users\you\photos_4k")
                    gr.HTML('<div style="height:4px"></div>')
                    with gr.Row():
                        batch_scale = gr.Radio(["2x","4x"], value="4x", label="Scale")
                        batch_tile  = gr.Dropdown(["None","512","256"], value="None", label="Tile size")
                    gr.HTML('<div style="height:12px"></div>')
                    batch_btn = gr.Button("Run Batch", variant="primary", elem_id="batch-run-btn")
                    gr.Markdown("Processes every JPG, PNG, and WEBP in the input folder.")

                with gr.Column(scale=2):
                    batch_log = gr.Textbox(
                        label="Progress log", lines=26,
                        interactive=False, autoscroll=True, elem_id="batch-log",
                    )

            batch_btn.click(
                fn=upscale_batch,
                inputs=[in_folder, out_folder, batch_scale, batch_tile],
                outputs=batch_log,
            )


if __name__ == "__main__":
    app.launch(inbrowser=True, css=CSS)
