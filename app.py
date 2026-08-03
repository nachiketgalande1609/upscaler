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

# model_key -> (filename, url, num_blocks_or_None, scale, is_hat)
MODELS = {
    "Real-HAT 4x  (best quality)": (
        "Real_HAT_GAN_SRx4",
        "https://huggingface.co/hfmaster/models-moved/resolve/main/upscalers/Real_HAT_GAN_SRx4.pth",
        None, 4, True,
    ),
    "Real-ESRGAN 4x  (fast)": (
        "RealESRGAN_x4plus",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        23, 4, False,
    ),
    "Real-ESRGAN 4x Lite  (fastest)": (
        "RealESRGAN_x4plus_anime_6B",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        6, 4, False,
    ),
    "Real-ESRGAN 2x": (
        "RealESRGAN_x2plus",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        23, 2, False,
    ),
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
    path = MODEL_DIR / f"{model_name}.pth"
    if path.exists():
        return path
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_name}...")

    if "huggingface.co" in url:
        # HuggingFace CDN requires requests for proper redirect + streaming
        import requests
        headers = {"User-Agent": "Mozilla/5.0"}
        with requests.get(url, stream=True, headers=headers, allow_redirects=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = min(int(downloaded * 100 / total), 100)
                        print(f"\r  {pct}%  ({downloaded >> 20} / {total >> 20} MB)", end="", flush=True)
        print()
    else:
        import urllib.request
        def _hook(n, bs, total):
            if total > 0:
                print(f"\r  {min(int(n*bs*100/total),100)}%", end="", flush=True)
        urllib.request.urlretrieve(url, path, _hook)
        print()

    return path


class _HATUpsampler:
    """Minimal upsampler for HAT — same interface as RealESRGANer."""

    def __init__(self, hat_model: torch.nn.Module, scale: int):
        self.model = hat_model
        self._window_size = hat_model.window_size  # cache before model may be swapped
        self.scale = scale
        self.tile_size = 0   # set by caller, same as RealESRGANer
        self.tile_pad = 32

    def enhance(self, img_bgr, outscale=None):
        scale = outscale or self.scale
        h, w = img_bgr.shape[:2]
        max_val = 65535.0 if img_bgr.max() > 256 else 255.0

        # BGR uint8/16 → RGB float [0,1] NCHW
        img_f = img_bgr.astype(np.float32) / max_val
        img_rgb = img_f[:, :, ::-1].copy()
        t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0)
        t = t.to(_DEVICE)  # HAT runs fp32 — no half() conversion

        ws = self._window_size  # 16 — cached at init, survives _CountingWrapper swap

        with torch.no_grad():
            if self.tile_size > 0:
                out = self._tile_process(t, ws)
            else:
                # Pad to window_size multiple, run, unpad
                ph = (ws - h % ws) % ws
                pw = (ws - w % ws) % ws
                if ph or pw:
                    t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode='reflect')
                out = self.model(t)
                out = out[:, :, :h * scale, :w * scale]

        # NCHW → HWC BGR uint8/16
        out_np = out.squeeze(0).float().clamp(0, 1).cpu().numpy()
        out_np = (out_np.transpose(1, 2, 0)[:, :, ::-1] * max_val).round()
        dtype = np.uint16 if max_val > 255 else np.uint8
        return out_np.clip(0, max_val).astype(dtype), None

    def _tile_process(self, t, ws):
        _, _, h, w = t.shape
        scale = self.scale
        tile = self.tile_size
        pad = self.tile_pad
        out = torch.zeros(1, 3, h * scale, w * scale, dtype=t.dtype, device=t.device)

        tiles_x = math.ceil(w / tile)
        tiles_y = math.ceil(h / tile)
        for yi in range(tiles_y):
            for xi in range(tiles_x):
                x0 = max(xi * tile - pad, 0)
                x1 = min((xi + 1) * tile + pad, w)
                y0 = max(yi * tile - pad, 0)
                y1 = min((yi + 1) * tile + pad, h)
                tile_in = t[:, :, y0:y1, x0:x1]

                # Pad tile to window_size multiple
                th, tw = tile_in.shape[2], tile_in.shape[3]
                tph = (ws - th % ws) % ws
                tpw = (ws - tw % ws) % ws
                if tph or tpw:
                    tile_in = torch.nn.functional.pad(tile_in, (0, tpw, 0, tph), mode='reflect')

                tile_out = self.model(tile_in)
                tile_out = tile_out[:, :, :th * scale, :tw * scale]

                # Destination (strip padding contribution)
                ox0 = (x0 if xi == 0 else x0 + pad) * scale
                oy0 = (y0 if yi == 0 else y0 + pad) * scale
                sx0 = 0 if xi == 0 else pad * scale
                sy0 = 0 if yi == 0 else pad * scale
                ox1 = ox0 + tile_out.shape[3] - sx0
                oy1 = oy0 + tile_out.shape[2] - sy0
                out[:, :, oy0:oy1, ox0:ox1] = tile_out[:, :, sy0:, sx0:]
        return out


def _get_upsampler(model_key: str):
    if model_key not in MODEL_CACHE:
        name, url, num_blocks, scale, is_hat = MODELS[model_key]
        path = _download_model(name, url)

        if is_hat:
            from hat_arch import HAT
            hat = HAT(
                upscale=4, in_chans=3, img_size=64, window_size=16,
                compress_ratio=3, squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
                upsampler='pixelshuffle', resi_connection='1conv',
            )
            loadnet = torch.load(str(path), map_location='cpu')
            key = 'params_ema' if 'params_ema' in loadnet else ('params' if 'params' in loadnet else None)
            hat.load_state_dict(loadnet[key] if key else loadnet, strict=True)
            hat.eval()
            hat = hat.to(_DEVICE)
            # HAT uses float32 internally for attention masks — fp16 causes type mismatch
            MODEL_CACHE[model_key] = _HATUpsampler(hat, scale)
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=num_blocks, num_grow_ch=32, scale=scale)
            MODEL_CACHE[model_key] = RealESRGANer(
                scale=scale, model_path=str(path), model=model,
                tile=0, tile_pad=10, pre_pad=0, half=_USE_HALF)
    return MODEL_CACHE[model_key]


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

def upscale_single(image, model_key, tile_choice):
    if image is None:
        gr.Warning("Upload an image first.")
        yield gr.update(), gr.update(), _idle(), ""
        return

    _, _, _, scale, _ = MODELS[model_key]
    tile = 0 if tile_choice == "None" else int(tile_choice)

    yield gr.update(), gr.update(), _prog(5, "Loading model…"), ""
    upsampler = _get_upsampler(model_key)
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


def upscale_batch(in_folder, out_folder, model_key, tile_choice):
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
    _, _, _, scale, _ = MODELS[model_key]
    tile = 0 if tile_choice == "None" else int(tile_choice)

    yield emit("Loading model…")
    try: up = _get_upsampler(model_key)
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

with gr.Blocks(title="4K Upscaler") as app:

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

                    model_drop = gr.Dropdown(
                        choices=list(MODELS.keys()),
                        value="Real-HAT 4x  (best quality)",
                        label="Model",
                        info="Fast 4x is ~4× quicker with nearly identical quality.",
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
                inputs=[input_img, model_drop, tile_drop],
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
                    batch_model = gr.Dropdown(
                        choices=list(MODELS.keys()),
                        value="Real-HAT 4x  (best quality)",
                        label="Model",
                    )
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
                inputs=[in_folder, out_folder, batch_model, batch_tile],
                outputs=batch_log,
            )


if __name__ == "__main__":
    app.launch(inbrowser=True, theme=gr.themes.Base(), css=CSS)
