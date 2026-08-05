"""FastAPI web UI for Real-ESRGAN / Real-HAT image upscaler."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import tempfile
import threading
import time
import uuid
import webbrowser
from pathlib import Path

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

# ── Hardware ──────────────────────────────────────────────────────────────────

if torch.cuda.is_available():
    _DEVICE = "cuda"
    _USE_HALF = True
    _USE_BF16_HAT = torch.cuda.get_device_capability()[0] >= 8
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    _gpu_name  = torch.cuda.get_device_name(0)
    _vram_gb   = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    _hat_prec  = "bf16" if _USE_BF16_HAT else "fp16"
    _DEVICE_LABEL = f"{_gpu_name} · {_vram_gb:.0f} GB VRAM · fp16"
    print(f"Device: {_DEVICE_LABEL} (HAT: {_hat_prec})")
else:
    _DEVICE = "cpu"
    _USE_HALF = False
    _USE_BF16_HAT = False
    torch.set_num_threads(os.cpu_count() or 8)
    _DEVICE_LABEL = f"CPU · {os.cpu_count()} threads · fp32"
    print(f"Device: {_DEVICE_LABEL}")

MODEL_CACHE: dict = {}
MODEL_DIR   = Path("models")
SUPPORTED   = {".jpg", ".jpeg", ".png", ".webp"}

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

# ── Model helpers ─────────────────────────────────────────────────────────────

class _CountingWrapper(torch.nn.Module):
    def __init__(self, model, counter):
        super().__init__()
        self._inner   = model
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
        import requests
        headers = {"User-Agent": "Mozilla/5.0"}
        with requests.get(url, stream=True, headers=headers, allow_redirects=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
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
        self.model        = hat_model
        self._window_size = hat_model.window_size
        self.scale        = scale
        self.tile_size    = 0
        self.tile_pad     = 32

    def enhance(self, img_bgr, outscale=None):
        scale   = outscale or self.scale
        h, w    = img_bgr.shape[:2]
        max_val = 65535.0 if img_bgr.max() > 256 else 255.0

        img_f   = img_bgr.astype(np.float32) / max_val
        img_rgb = img_f[:, :, ::-1].copy()
        t       = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0)
        if _USE_BF16_HAT:
            t = t.to(torch.bfloat16)
        t = t.to(_DEVICE)

        ws = self._window_size

        with torch.no_grad():
            if self.tile_size > 0:
                out = self._tile_process(t, ws)
            else:
                ph = (ws - h % ws) % ws
                pw = (ws - w % ws) % ws
                if ph or pw:
                    t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode='reflect')
                out = self.model(t)
                out = out[:, :, :h * scale, :w * scale]

        out_np = out.squeeze(0).float().clamp(0, 1).cpu().numpy()
        out_np = (out_np.transpose(1, 2, 0)[:, :, ::-1] * max_val).round()
        dtype  = np.uint16 if max_val > 255 else np.uint8
        return out_np.clip(0, max_val).astype(dtype), None

    def _tile_process(self, t, ws):
        _, _, h, w = t.shape
        scale = self.scale
        tile  = self.tile_size
        pad   = self.tile_pad
        out   = torch.zeros(1, 3, h * scale, w * scale, dtype=t.dtype, device=t.device)

        for yi in range(math.ceil(h / tile)):
            for xi in range(math.ceil(w / tile)):
                x0 = max(xi * tile - pad, 0);  x1 = min((xi + 1) * tile + pad, w)
                y0 = max(yi * tile - pad, 0);  y1 = min((yi + 1) * tile + pad, h)
                tile_in = t[:, :, y0:y1, x0:x1]

                th, tw = tile_in.shape[2], tile_in.shape[3]
                tph = (ws - th % ws) % ws;  tpw = (ws - tw % ws) % ws
                if tph or tpw:
                    tile_in = torch.nn.functional.pad(tile_in, (0, tpw, 0, tph), mode='reflect')

                tile_out = self.model(tile_in)
                tile_out = tile_out[:, :, :th * scale, :tw * scale]
                del tile_in
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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
            if _USE_BF16_HAT:
                hat = hat.to(torch.bfloat16)
            hat = hat.to(_DEVICE)
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


def _make_thumb_b64(bgr: np.ndarray, max_dim: int = 480) -> str:
    h, w = bgr.shape[:2]
    s = min(1.0, max_dim / max(h, w, 1))
    small = bgr if s >= 1.0 else cv2.resize(bgr, (int(w * s), int(h * s)))
    _, enc = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(enc.tobytes()).decode()


# ── FastAPI ────────────────────────────────────────────────────────────────────

webapp = FastAPI()
_tasks: dict = {}


def _run_upscale(task_id: str, img_bytes: bytes, model_key: str, tile_str: str):
    def upd(**kw):
        _tasks[task_id].update(kw)

    try:
        upd(progress=5, label="Decoding image…")
        arr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            raise ValueError("Could not decode image")

        _, _, _, scale, is_hat = MODELS[model_key]
        tile = int(tile_str)
        if is_hat and tile == 0:
            tile = 512

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        upd(progress=10, label="Loading model…")
        upsampler       = _get_upsampler(model_key)
        upsampler.tile_size = tile

        h, w = img_bgr.shape[:2]

        if tile > 0:
            total   = math.ceil(w / tile) * math.ceil(h / tile)
            counter = [0]
            orig    = upsampler.model
            upsampler.model = _CountingWrapper(orig, counter)
            res_box, err_box = [None], [None]

            def _run():
                try:
                    res_box[0], _ = upsampler.enhance(img_bgr, outscale=scale)
                except Exception as e:
                    err_box[0] = e
                finally:
                    upsampler.model = orig

            th = threading.Thread(target=_run, daemon=True)
            th.start()
            while th.is_alive():
                n   = counter[0]
                pct = 15 + int(78 * n / total) if total else 50
                upd(progress=pct,
                    label=f"Upscaling… tile {n}/{total}",
                    current_tile=n,
                    total_tiles=total)
                time.sleep(0.25)
            th.join()
            upsampler.model = orig
            if err_box[0]:
                raise err_box[0]
            out_bgr = res_box[0]
        else:
            upd(progress=20, label=f"Upscaling {w}×{h}…", current_tile=0, total_tiles=0)
            out_bgr, _ = upsampler.enhance(img_bgr, outscale=scale)

        upd(progress=96, label="Saving…")
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, out_bgr)
        oh, ow = out_bgr.shape[:2]

        upd(progress=100, label="Done!", done=True,
            result=tmp.name, input_size=f"{w}×{h}",
            output_size=f"{ow}×{oh}", scale=scale,
            current_tile=total if tile > 0 else 0,
            total_tiles=total if tile > 0 else 0)

    except Exception as e:
        upd(progress=0, label=str(e), done=True, error=True)


def _run_batch(task_id: str, in_folder: str, out_folder: str, model_key: str, tile_str: str):
    def upd(**kw):
        _tasks[task_id].update(kw)

    def log(msg: str, status: str = "info"):
        entry = {"msg": msg, "status": status, "ts": time.time()}
        _tasks[task_id]["log"].append(entry)

    try:
        src = Path(in_folder.strip())
        dst = Path(out_folder.strip())
        if not src.is_dir():
            raise ValueError(f"Input folder not found: {src}")

        imgs = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)
        if not imgs:
            raise ValueError(f"No supported images in {src}")

        dst.mkdir(parents=True, exist_ok=True)
        _, _, _, scale, is_hat = MODELS[model_key]
        tile = int(tile_str)
        if is_hat and tile == 0:
            tile = 512

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log("Loading model…")
        upd(progress=2, label="Loading model…", total=len(imgs), done_count=0,
            current_tile=0, total_tiles=0, current_img_b64=None, current_img_name="")
        upsampler = _get_upsampler(model_key)
        upsampler.tile_size = tile
        log(f"Ready — {len(imgs)} image(s) found", "ok")

        ok = 0
        for i, p in enumerate(imgs):
            img_pct_start = 5 + int(90 * i / len(imgs))
            img_pct_end   = 5 + int(90 * (i + 1) / len(imgs))
            upd(progress=img_pct_start,
                label=f"Processing {i+1}/{len(imgs)}: {p.name}",
                done_count=i,
                current_tile=0, total_tiles=0,
                current_img_b64=None, current_img_name=p.name)
            log(p.name, "processing")
            t0 = time.time()
            try:
                bgr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if bgr is None:
                    raise ValueError("Could not read file")

                # Thumbnail preview for the UI
                upd(current_img_b64=_make_thumb_b64(bgr), current_img_name=p.name)

                if tile > 0:
                    bh, bw = bgr.shape[:2]
                    total_t = math.ceil(bw / tile) * math.ceil(bh / tile)
                    counter = [0]
                    orig = upsampler.model
                    upsampler.model = _CountingWrapper(orig, counter)
                    res_box, err_box = [None], [None]
                    _bgr = bgr  # capture for closure

                    def _do():
                        try:
                            res_box[0], _ = upsampler.enhance(_bgr, outscale=scale)
                        except Exception as ex:
                            err_box[0] = ex
                        finally:
                            upsampler.model = orig

                    th = threading.Thread(target=_do, daemon=True)
                    th.start()
                    while th.is_alive():
                        n = counter[0]
                        inner_pct = img_pct_start + int(
                            (img_pct_end - img_pct_start) * n / total_t
                        ) if total_t else img_pct_start
                        upd(current_tile=n, total_tiles=total_t, progress=inner_pct)
                        time.sleep(0.3)
                    th.join()
                    upsampler.model = orig
                    if err_box[0]:
                        raise err_box[0]
                    out = res_box[0]
                else:
                    upd(current_tile=0, total_tiles=0)
                    out, _ = upsampler.enhance(bgr, outscale=scale)

                ext = ".png" if p.suffix.lower() == ".webp" else p.suffix.lower()
                op  = dst / (p.stem + ext)
                cv2.imwrite(str(op), out)
                elapsed = time.time() - t0
                log(f"{p.name}  ({elapsed:.1f}s)", "ok")
                ok += 1
            except Exception as e:
                log(f"{p.name}  ✗  {e}", "error")

            upd(current_tile=0, total_tiles=0)

        upd(progress=100, label=f"Done — {ok}/{len(imgs)} succeeded", done=True,
            done_count=len(imgs), current_img_b64=None, current_img_name="",
            current_tile=0, total_tiles=0)

    except Exception as e:
        upd(progress=0, label=str(e), done=True, error=True)


# ── Routes ────────────────────────────────────────────────────────────────────

@webapp.get("/")
async def index():
    model_opts = "".join(f'<option value="{k}">{k}</option>' for k in MODELS)
    html = _HTML.replace("{{MODEL_OPTIONS}}", model_opts) \
                .replace("{{DEVICE_LABEL}}", _DEVICE_LABEL)
    return HTMLResponse(html)


@webapp.post("/upscale")
async def start_upscale(
    file:  UploadFile = File(...),
    model: str        = Form(...),
    tile:  str        = Form("0"),
):
    if model not in MODELS:
        return JSONResponse({"error": "Unknown model"}, status_code=400)
    task_id   = str(uuid.uuid4())
    img_bytes = await file.read()
    _tasks[task_id] = {
        "progress": 0, "label": "Queued", "done": False, "error": False,
        "result": None, "current_tile": 0, "total_tiles": 0,
    }
    threading.Thread(target=_run_upscale, args=(task_id, img_bytes, model, tile), daemon=True).start()
    return {"task_id": task_id}


@webapp.get("/progress/{task_id}")
async def stream_progress(task_id: str):
    async def _gen():
        while True:
            t = _tasks.get(task_id)
            if not t:
                break
            payload = {k: v for k, v in t.items() if k != "result"}
            yield f"data: {json.dumps(payload)}\n\n"
            if t.get("done"):
                break
            await asyncio.sleep(0.2)
    return StreamingResponse(_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@webapp.get("/result/{task_id}")
async def get_result(task_id: str):
    t    = _tasks.get(task_id, {})
    path = t.get("result")
    if path and Path(path).exists():
        return FileResponse(path, media_type="image/png", filename="upscaled.png")
    return JSONResponse({"error": "not found"}, status_code=404)


@webapp.post("/batch")
async def start_batch(
    in_folder:  str = Form(...),
    out_folder: str = Form(...),
    model:      str = Form(...),
    tile:       str = Form("0"),
):
    if model not in MODELS:
        return JSONResponse({"error": "Unknown model"}, status_code=400)
    task_id = str(uuid.uuid4())
    _tasks[task_id] = {
        "progress": 0, "label": "Queued", "done": False, "error": False,
        "log": [], "total": 0, "done_count": 0,
        "current_tile": 0, "total_tiles": 0,
        "current_img_b64": None, "current_img_name": "",
    }
    threading.Thread(target=_run_batch,
                     args=(task_id, in_folder, out_folder, model, tile),
                     daemon=True).start()
    return {"task_id": task_id}


@webapp.get("/batch-progress/{task_id}")
async def stream_batch_progress(task_id: str):
    async def _gen():
        last_log_len = 0
        while True:
            t = _tasks.get(task_id)
            if not t:
                break
            new_entries = t["log"][last_log_len:]
            last_log_len = len(t["log"])
            payload = {
                "progress":        t["progress"],
                "label":           t["label"],
                "done":            t["done"],
                "error":           t["error"],
                "total":           t["total"],
                "done_count":      t["done_count"],
                "current_tile":    t.get("current_tile", 0),
                "total_tiles":     t.get("total_tiles", 0),
                "current_img_b64": t.get("current_img_b64"),
                "current_img_name": t.get("current_img_name", ""),
                "new_logs":        new_entries,
            }
            yield f"data: {json.dumps(payload)}\n\n"
            if t.get("done"):
                break
            await asyncio.sleep(0.25)
    return StreamingResponse(_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── HTML template ─────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Upscaler</title>
<style>
:root {
  --bg:      #06060f;
  --bg2:     #0d0d1a;
  --glass:   rgba(255,255,255,.03);
  --border:  rgba(255,255,255,.07);
  --accent:  #7c3aed;
  --accent2: #a78bfa;
  --green:   #4ade80;
  --red:     #f87171;
  --amber:   #fbbf24;
  --text:    #f1f5f9;
  --text2:   #94a3b8;
  --text3:   #334155;
  --sw:      310px;
}

* { box-sizing:border-box; margin:0; padding:0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  background: var(--bg); color: var(--text);
  height: 100vh; overflow: hidden;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,.08); border-radius: 99px; }

/* ── Header ── */
header {
  height: 52px; display: flex; align-items: center;
  justify-content: space-between; padding: 0 20px;
  border-bottom: 1px solid var(--border);
  background: rgba(6,6,15,.9); backdrop-filter: blur(20px);
  position: relative; z-index: 50;
}
.logo {
  display: flex; align-items: center; gap: 10px;
}
.logo-icon {
  width: 30px; height: 30px; border-radius: 9px;
  background: linear-gradient(135deg, #4c1d95, #7c3aed);
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 4px 14px rgba(124,58,237,.4);
}
.logo-text {
  font-weight: 700; font-size: 15px; letter-spacing: -.02em;
}
.device-badge {
  display: flex; align-items: center; gap: 7px;
  padding: 5px 12px; border-radius: 99px;
  background: rgba(255,255,255,.03); border: 1px solid var(--border);
}
.dot-green {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--green); box-shadow: 0 0 8px var(--green);
  flex-shrink: 0;
}
.device-text {
  font-size: 11px; color: var(--text3);
  font-family: ui-monospace, 'Cascadia Code', Consolas, monospace;
}

/* ── Layout ── */
.layout {
  display: flex; height: calc(100vh - 52px);
}

/* ── Sidebar ── */
aside {
  width: var(--sw); flex-shrink: 0;
  border-right: 1px solid var(--border);
  display: flex; flex-direction: column; overflow: hidden;
  background: rgba(13,13,26,.5);
}

/* ── Tabs ── */
.tab-nav {
  display: flex; gap: 3px; padding: 10px 12px 8px;
  border-bottom: 1px solid var(--border); flex-shrink: 0;
}
.tab-btn {
  flex: 1; padding: 8px; font-size: 11px; font-weight: 700;
  letter-spacing: .07em; text-transform: uppercase;
  border: 1px solid transparent; border-radius: 8px;
  cursor: pointer; transition: all .2s;
  background: transparent; color: var(--text3);
}
.tab-btn.active {
  background: rgba(124,58,237,.14);
  border-color: rgba(124,58,237,.28);
  color: var(--accent2);
}
.tab-btn:hover:not(.active) { color: var(--text2); }

/* ── Panel ── */
.panel {
  flex: 1; overflow-y: auto;
  padding: 14px 14px 20px; display: flex; flex-direction: column; gap: 13px;
}

/* ── Field label ── */
.lbl {
  font-size: 10px; font-weight: 700; letter-spacing: .1em;
  text-transform: uppercase; color: var(--text3); margin-bottom: 5px; display: block;
}
.hint { font-size: 11px; color: var(--text3); margin-top: 4px; }

/* ── Drop zone ── */
.drop-zone {
  border: 2px dashed rgba(124,58,237,.3); border-radius: 12px;
  padding: 22px 12px; text-align: center; cursor: pointer;
  transition: all .2s; position: relative;
}
.drop-zone:hover, .drop-zone.drag-over {
  border-color: var(--accent); background: rgba(124,58,237,.05);
}
.drop-zone.has-file {
  border-style: solid; border-color: rgba(124,58,237,.4); padding: 10px 12px;
}

/* ── Select ── */
.sel-wrap { position: relative; }
.sel-wrap select {
  width: 100%; padding: 9px 32px 9px 11px; appearance: none;
  background: rgba(255,255,255,.04); border: 1px solid var(--border);
  border-radius: 9px; color: var(--text); font-size: 13px; cursor: pointer;
  transition: border-color .2s; outline: none;
}
.sel-wrap select:hover, .sel-wrap select:focus { border-color: rgba(124,58,237,.5); }
.sel-wrap select option { background: #131320; }
.sel-arrow {
  position: absolute; right: 10px; top: 50%; transform: translateY(-50%);
  pointer-events: none; color: var(--text3);
}

/* ── Text input ── */
.txt-input {
  width: 100%; padding: 9px 11px;
  background: rgba(255,255,255,.04); border: 1px solid var(--border);
  border-radius: 9px; color: var(--text); font-size: 12px;
  font-family: ui-monospace, Consolas, monospace;
  transition: border-color .2s; outline: none;
}
.txt-input:focus { border-color: rgba(124,58,237,.5); }
.txt-input::placeholder { color: var(--text3); }

/* ── Buttons ── */
.btn-primary {
  width: 100%; padding: 12px; border: none; border-radius: 10px;
  cursor: pointer; font-size: 14px; font-weight: 700; color: #fff;
  background: linear-gradient(135deg, #4c1d95, #7c3aed);
  box-shadow: 0 4px 20px rgba(124,58,237,.3);
  transition: transform .15s, box-shadow .15s, opacity .15s;
  letter-spacing: .02em;
}
.btn-primary:hover:not(:disabled) {
  transform: translateY(-1px); box-shadow: 0 8px 32px rgba(124,58,237,.5);
}
.btn-primary:active:not(:disabled) { transform: translateY(0); }
.btn-primary:disabled { opacity: .3; cursor: not-allowed; }

.btn-dl {
  display: flex; align-items: center; justify-content: center; gap: 8px;
  width: 100%; padding: 10px; border-radius: 10px; font-size: 13px;
  font-weight: 600; color: var(--green); border: 1px solid rgba(74,222,128,.25);
  background: rgba(74,222,128,.05); text-decoration: none; transition: all .2s;
}
.btn-dl:hover { background: rgba(74,222,128,.1); border-color: rgba(74,222,128,.4); }

/* ── Progress bar ── */
.prog-track {
  height: 4px; background: rgba(255,255,255,.05);
  border-radius: 99px; overflow: hidden;
}
@keyframes shimmer {
  0%   { background-position: 200% center; }
  100% { background-position: -200% center; }
}
.prog-fill {
  height: 100%; border-radius: 99px; transition: width .3s ease;
  background: linear-gradient(90deg,#4c1d95,#7c3aed,#a78bfa,#7c3aed,#4c1d95);
  background-size: 300% 100%; animation: shimmer 2s linear infinite;
}
.prog-fill.done  { background: var(--green); animation: none; }
.prog-fill.error { background: var(--red);   animation: none; }

/* ── Tile grid ── */
.tile-grid-card {
  background: rgba(0,0,0,.25); border: 1px solid var(--border);
  border-radius: 10px; padding: 10px; margin-top: 10px;
}
.tile-grid-header {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 8px;
}
.tile-grid-title {
  font-size: 10px; font-weight: 700; letter-spacing: .09em;
  text-transform: uppercase; color: var(--text3);
}
.tile-counter {
  font-size: 11px; font-weight: 700;
  font-family: ui-monospace, Consolas, monospace; color: var(--accent2);
}
#tile-grid, #b-tile-grid {
  display: grid; gap: 2px;
}
.tile-cell {
  aspect-ratio: 1; border-radius: 2px;
  background: rgba(255,255,255,.06); transition: background .12s;
}
.tile-cell.done { background: rgba(124,58,237,.6); }
.tile-cell.active {
  background: var(--accent2);
  animation: tpulse .6s ease infinite alternate;
}
@keyframes tpulse { to { opacity: .5; } }

/* ── Stat chips ── */
.stats-row { display: flex; gap: 7px; align-items: center; }
.stat-chip {
  flex: 1; display: flex; flex-direction: column; align-items: center;
  padding: 8px 6px; background: rgba(255,255,255,.03);
  border: 1px solid var(--border); border-radius: 9px;
}
.stat-val {
  font-size: 12px; font-family: ui-monospace, Consolas, monospace;
  font-weight: 700; color: var(--text);
}
.stat-key {
  font-size: 9px; color: var(--text3); margin-top: 2px;
  text-transform: uppercase; letter-spacing: .07em;
}
.arrow-sep { color: var(--text3); font-size: 16px; flex-shrink: 0; }

/* ── Spinner ── */
@keyframes spin { to { transform: rotate(360deg); } }
.spinner {
  width: 14px; height: 14px; border-radius: 50%;
  border: 2px solid rgba(124,58,237,.2);
  border-top-color: var(--accent);
  animation: spin .65s linear infinite; flex-shrink: 0;
}

/* ── Skeleton ── */
@keyframes skel {
  0%   { background-position: 200% center; }
  100% { background-position: -200% center; }
}
.skeleton {
  background: linear-gradient(90deg,
    rgba(255,255,255,.03), rgba(255,255,255,.07), rgba(255,255,255,.03));
  background-size: 300% 100%; animation: skel 1.8s linear infinite;
}

/* ── Comparison slider ── */
.compare-wrap {
  position: relative; overflow: hidden; user-select: none;
  cursor: col-resize; border-radius: 10px; width: 100%; height: 100%;
}
.cmp-layer {
  position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
}
.cmp-layer img { max-width: 100%; max-height: 100%; object-fit: contain; }
#cmp-after-clip { clip-path: inset(0 50% 0 0); }
.cmp-handle {
  position: absolute; top: 0; bottom: 0; width: 2px; left: 50%;
  background: rgba(255,255,255,.9); transform: translateX(-50%);
  z-index: 10; pointer-events: none;
}
.cmp-handle::before {
  content: ''; position: absolute; top: 50%; left: 50%;
  transform: translate(-50%,-50%);
  width: 34px; height: 34px; border-radius: 50%;
  background: white; box-shadow: 0 4px 20px rgba(0,0,0,.7);
}
.cmp-badge {
  position: absolute; bottom: 12px; padding: 3px 10px;
  background: rgba(0,0,0,.6); backdrop-filter: blur(8px);
  border-radius: 99px; font-size: 10px; font-weight: 700;
  letter-spacing: .08em; text-transform: uppercase;
  color: rgba(255,255,255,.6); pointer-events: none; z-index: 5;
}
.toggle-btn {
  position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
  background: rgba(0,0,0,.65); backdrop-filter: blur(12px);
  border: 1px solid rgba(255,255,255,.1); border-radius: 99px;
  padding: 6px 18px; font-size: 11px; color: var(--text2);
  cursor: pointer; transition: all .2s; z-index: 20; white-space: nowrap;
}
.toggle-btn:hover { border-color: rgba(255,255,255,.2); color: var(--text); }

/* ── Batch log ── */
.batch-log { overflow-y: auto; display: flex; flex-direction: column; gap: 3px; }
.log-row {
  display: flex; align-items: center; gap: 8px;
  padding: 7px 10px; border-radius: 7px;
  background: rgba(255,255,255,.02); border: 1px solid rgba(255,255,255,.04);
  font-size: 11px; font-family: ui-monospace, Consolas, monospace;
}
@keyframes rowIn {
  from { opacity:0; transform: translateX(-6px); }
  to   { opacity:1; transform: translateX(0); }
}
.log-row { animation: rowIn .18s ease; }
.log-row.ok         { border-color: rgba(74,222,128,.15); }
.log-row.error      { border-color: rgba(241,68,68,.15); color: var(--red); }
.log-row.processing { border-color: rgba(124,58,237,.2); }
.log-row .log-name  { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text2); }

/* ── Empty state ── */
.empty-state {
  flex: 1; display: flex; align-items: center; justify-content: center;
}
.empty-icon {
  width: 68px; height: 68px; border-radius: 18px;
  background: rgba(255,255,255,.02); border: 1px solid var(--border);
  display: flex; align-items: center; justify-content: center;
  margin: 0 auto 16px;
}

@keyframes fadeUp {
  from { opacity:0; transform:translateY(8px); }
  to   { opacity:1; transform:translateY(0); }
}
.fade-up { animation: fadeUp .3s ease both; }

/* ── Panel section header ── */
.section-hdr {
  flex-shrink: 0; padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
}
.section-title {
  font-size: 10px; font-weight: 700; letter-spacing: .1em;
  text-transform: uppercase; color: var(--text3);
}
</style>
</head>
<body>

<!-- Header -->
<header>
  <div class="logo">
    <div class="logo-icon">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5">
        <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
      </svg>
    </div>
    <span class="logo-text">AI Upscaler</span>
  </div>
  <div class="device-badge">
    <span class="dot-green"></span>
    <span class="device-text">{{DEVICE_LABEL}}</span>
  </div>
</header>

<!-- Layout -->
<div class="layout">

<!-- ══ SIDEBAR ══ -->
<aside>
  <div class="tab-nav">
    <button class="tab-btn active" id="tab-single" onclick="switchTab('single')">Single</button>
    <button class="tab-btn"        id="tab-batch"  onclick="switchTab('batch')">Batch</button>
  </div>

  <!-- Single panel -->
  <div class="panel" id="panel-single">

    <!-- Upload -->
    <div>
      <span class="lbl">Image</span>
      <div id="drop-zone" class="drop-zone" onclick="document.getElementById('fi').click()">
        <input id="fi" type="file" accept="image/*" style="display:none">
        <div id="dz-idle">
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="var(--text3)"
               stroke-width="1.5" style="margin:0 auto 10px;display:block;">
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
            <polyline points="17 8 12 3 7 8"/>
            <line x1="12" y1="3" x2="12" y2="15"/>
          </svg>
          <p style="font-size:13px;color:var(--text2);margin:0 0 3px;">Drop image here</p>
          <p style="font-size:11px;color:var(--text3);">or click to browse · JPG PNG WebP</p>
        </div>
        <div id="dz-file" style="display:none;">
          <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:40px;height:40px;border-radius:7px;overflow:hidden;flex-shrink:0;background:rgba(124,58,237,.15);">
              <img id="thumb" style="width:100%;height:100%;object-fit:cover;">
            </div>
            <div style="min-width:0;flex:1;">
              <div id="fname" style="font-size:12px;color:var(--text);font-weight:600;
                   white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"></div>
              <div id="fsize" style="font-size:11px;color:var(--text3);margin-top:2px;"></div>
            </div>
            <button onclick="resetFile(event)"
              style="background:none;border:none;cursor:pointer;color:var(--text3);
                     padding:4px;border-radius:5px;flex-shrink:0;transition:color .15s;"
              onmouseover="this.style.color='var(--red)'"
              onmouseout="this.style.color='var(--text3)'">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" stroke-width="2.5">
                <path d="M18 6L6 18M6 6l12 12"/>
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Model -->
    <div>
      <span class="lbl">Model</span>
      <div class="sel-wrap">
        <select id="model-sel">{{MODEL_OPTIONS}}</select>
        <svg class="sel-arrow" width="12" height="12" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2.5"><path d="M6 9l6 6 6-6"/></svg>
      </div>
    </div>

    <!-- Tile size -->
    <div>
      <span class="lbl">Tile size
        <span style="font-weight:400;text-transform:none;letter-spacing:0;color:var(--text3);font-size:10px;">
          — lower = less VRAM
        </span>
      </span>
      <div class="sel-wrap">
        <select id="tile-sel">
          <option value="0">Auto</option>
          <option value="256">256 px</option>
          <option value="512" selected>512 px</option>
          <option value="1024">1024 px</option>
        </select>
        <svg class="sel-arrow" width="12" height="12" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2.5"><path d="M6 9l6 6 6-6"/></svg>
      </div>
    </div>

    <!-- Upscale button -->
    <button id="btn-up" class="btn-primary" disabled onclick="doUpscale()">
      Upscale Image
    </button>

    <!-- Progress -->
    <div id="prog-wrap" style="display:none;">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
        <div style="display:flex;align-items:center;gap:8px;">
          <div id="prog-spinner" class="spinner" style="display:none;"></div>
          <span id="prog-lbl" style="font-size:12px;color:var(--text2);"></span>
        </div>
        <span id="prog-pct"
          style="font-size:12px;font-family:ui-monospace,Consolas,monospace;font-weight:700;color:var(--accent2);"></span>
      </div>
      <div class="prog-track"><div id="prog-fill" class="prog-fill" style="width:0%"></div></div>

      <!-- Tile grid (shown when tiling) -->
      <div id="tile-prog-wrap" class="tile-grid-card" style="display:none;">
        <div class="tile-grid-header">
          <span class="tile-grid-title">Tile Progress</span>
          <span id="tile-counter" class="tile-counter"></span>
        </div>
        <div id="tile-grid"></div>
      </div>
    </div>

    <!-- Stats -->
    <div id="stats-wrap" style="display:none;" class="fade-up">
      <div class="stats-row">
        <div class="stat-chip">
          <span id="stat-in"    class="stat-val">–</span>
          <span class="stat-key">Input</span>
        </div>
        <span class="arrow-sep">→</span>
        <div class="stat-chip">
          <span id="stat-out"   class="stat-val">–</span>
          <span class="stat-key">Output</span>
        </div>
        <div class="stat-chip" style="flex:0 0 auto;padding:8px 12px;">
          <span id="stat-scale" class="stat-val" style="color:var(--accent2);">–</span>
          <span class="stat-key">Scale</span>
        </div>
      </div>
    </div>

    <!-- Download -->
    <a id="btn-dl" href="#" class="btn-dl" style="display:none;" download="upscaled.png">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/>
      </svg>
      Download Result
    </a>

  </div><!-- /panel-single -->

  <!-- Batch panel -->
  <div class="panel" id="panel-batch" style="display:none;">

    <div>
      <span class="lbl">Input folder</span>
      <input id="b-in" class="txt-input" type="text" placeholder="C:\images\input" spellcheck="false">
      <p class="hint">Full path · JPG, PNG, WebP files</p>
    </div>

    <div>
      <span class="lbl">Output folder</span>
      <input id="b-out" class="txt-input" type="text" placeholder="C:\images\output" spellcheck="false">
      <p class="hint">Created automatically if missing</p>
    </div>

    <div>
      <span class="lbl">Model</span>
      <div class="sel-wrap">
        <select id="b-model">{{MODEL_OPTIONS}}</select>
        <svg class="sel-arrow" width="12" height="12" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2.5"><path d="M6 9l6 6 6-6"/></svg>
      </div>
    </div>

    <div>
      <span class="lbl">Tile size</span>
      <div class="sel-wrap">
        <select id="b-tile">
          <option value="0">Auto</option>
          <option value="256">256 px</option>
          <option value="512" selected>512 px</option>
          <option value="1024">1024 px</option>
        </select>
        <svg class="sel-arrow" width="12" height="12" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2.5"><path d="M6 9l6 6 6-6"/></svg>
      </div>
    </div>

    <button id="btn-batch" class="btn-primary" onclick="doBatch()">Start Batch</button>

    <!-- Batch overall progress -->
    <div id="b-prog-wrap" style="display:none;">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
        <div style="display:flex;align-items:center;gap:8px;">
          <div id="b-prog-spinner" class="spinner" style="display:none;"></div>
          <span id="b-prog-lbl" style="font-size:12px;color:var(--text2);"></span>
        </div>
        <span id="b-count-lbl"
          style="font-size:12px;font-family:ui-monospace,Consolas,monospace;font-weight:700;color:var(--accent2);"></span>
      </div>
      <div class="prog-track"><div id="b-prog-fill" class="prog-fill" style="width:0%"></div></div>

      <!-- Per-image tile progress -->
      <div id="b-tile-prog-wrap" class="tile-grid-card" style="display:none;">
        <div class="tile-grid-header">
          <span class="tile-grid-title">Image Tiles</span>
          <span id="b-tile-counter" class="tile-counter"></span>
        </div>
        <div id="b-tile-grid"></div>
      </div>
    </div>

  </div><!-- /panel-batch -->
</aside>

<!-- ══ SINGLE MAIN ══ -->
<main id="main-single" style="flex:1;display:flex;flex-direction:column;overflow:hidden;">

  <!-- Empty state -->
  <div id="empty-state" class="empty-state">
    <div style="text-align:center;">
      <div class="empty-icon">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--text3)" stroke-width="1.5">
          <rect x="3" y="3" width="18" height="18" rx="2"/>
          <circle cx="8.5" cy="8.5" r="1.5"/>
          <path d="M21 15l-5-5L5 21"/>
        </svg>
      </div>
      <p style="color:var(--text2);font-size:15px;font-weight:600;margin-bottom:6px;">No image selected</p>
      <p style="color:var(--text3);font-size:13px;">Drop an image in the sidebar to get started</p>
    </div>
  </div>

  <!-- Column headers -->
  <div id="img-headers" style="display:none;flex-shrink:0;border-bottom:1px solid var(--border);">
    <div style="display:flex;">
      <div style="flex:1;text-align:center;padding:10px 0;font-size:10px;font-weight:700;
                  letter-spacing:.1em;text-transform:uppercase;color:var(--text3);">Before</div>
      <div style="width:1px;background:var(--border);"></div>
      <div style="flex:1;text-align:center;padding:10px 0;font-size:10px;font-weight:700;
                  letter-spacing:.1em;text-transform:uppercase;color:var(--text3);">After</div>
    </div>
  </div>

  <!-- Image area -->
  <div id="img-area" style="display:none;flex:1;overflow:hidden;position:relative;">

    <!-- Side-by-side -->
    <div id="side-by-side" style="display:flex;height:100%;">
      <div style="flex:1;display:flex;align-items:center;justify-content:center;padding:20px;overflow:hidden;">
        <img id="before-img" style="max-width:100%;max-height:100%;object-fit:contain;border-radius:8px;">
      </div>
      <div style="width:1px;background:var(--border);flex-shrink:0;"></div>
      <div style="flex:1;display:flex;align-items:center;justify-content:center;padding:20px;overflow:hidden;">
        <div id="after-placeholder" style="color:var(--text3);font-size:14px;text-align:center;">
          <div style="font-size:32px;margin-bottom:8px;opacity:.5;">→</div>
          <span>Click Upscale</span>
        </div>
        <div id="after-skeleton" style="display:none;width:80%;height:72%;border-radius:8px;" class="skeleton"></div>
        <img id="after-img-side" style="display:none;max-width:100%;max-height:100%;object-fit:contain;border-radius:8px;">
      </div>
    </div>

    <!-- Comparison slider -->
    <div id="compare-view" style="display:none;height:100%;padding:20px;position:relative;">
      <div id="compare-wrap" class="compare-wrap">
        <div class="cmp-layer">
          <img id="cmp-before" style="max-width:100%;max-height:100%;object-fit:contain;">
          <span class="cmp-badge" style="left:14px;">Before</span>
        </div>
        <div class="cmp-layer" id="cmp-after-clip">
          <img id="cmp-after" style="max-width:100%;max-height:100%;object-fit:contain;">
          <span class="cmp-badge" style="right:14px;">After</span>
        </div>
        <div class="cmp-handle" id="cmp-handle"></div>
      </div>
      <button class="toggle-btn" id="toggle-btn" onclick="toggleView()">
        ⇄ Side-by-side
      </button>
    </div>

  </div>
</main>

<!-- ══ BATCH MAIN ══ -->
<main id="main-batch" style="flex:1;display:none;flex-direction:column;overflow:hidden;">
  <div style="display:flex;height:100%;overflow:hidden;">

    <!-- Preview pane -->
    <div style="flex:1;display:flex;flex-direction:column;overflow:hidden;border-right:1px solid var(--border);">
      <div class="section-hdr">
        <span class="section-title">Current Image</span>
        <span id="b-current-name"
          style="font-size:11px;color:var(--text2);font-family:ui-monospace,Consolas,monospace;
                 max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"></span>
      </div>
      <div id="batch-preview-area"
        style="flex:1;display:flex;align-items:center;justify-content:center;padding:24px;overflow:hidden;">
        <div id="b-idle-msg" style="text-align:center;">
          <div style="width:60px;height:60px;border-radius:16px;background:rgba(255,255,255,.02);
                      border:1px solid var(--border);display:flex;align-items:center;
                      justify-content:center;margin:0 auto 14px;">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="var(--text3)" stroke-width="1.5">
              <rect x="3" y="3" width="18" height="18" rx="2"/>
              <circle cx="8.5" cy="8.5" r="1.5"/>
              <path d="M21 15l-5-5L5 21"/>
            </svg>
          </div>
          <p style="color:var(--text3);font-size:13px;" id="b-idle-text">Enter folders and start</p>
        </div>
        <div id="b-img-wrap" style="display:none;width:100%;height:100%;
             display:none;align-items:center;justify-content:center;">
          <img id="b-current-img" style="max-width:100%;max-height:100%;
               object-fit:contain;border-radius:10px;box-shadow:0 8px 40px rgba(0,0,0,.5);">
        </div>
      </div>
    </div>

    <!-- Log pane -->
    <div style="width:300px;flex-shrink:0;display:flex;flex-direction:column;overflow:hidden;">
      <div class="section-hdr">
        <span class="section-title">Batch Log</span>
        <span id="b-count-lbl-main"
          style="font-size:11px;color:var(--accent2);font-family:ui-monospace,Consolas,monospace;font-weight:700;"></span>
      </div>
      <div id="batch-log" class="batch-log" style="flex:1;padding:8px 10px 10px;">
        <div style="padding:40px 0;text-align:center;color:var(--text3);font-size:12px;">
          Waiting…
        </div>
      </div>
    </div>

  </div>
</main>

</div><!-- /layout -->

<script>
// ── State ──────────────────────────────────────────────────────────────────
let selectedFile = null, taskId = null, evtSrc = null;
let hasResult = false, viewMode = 'side';
let batchEvtSrc = null;
let _sliderDragging = false;

// ── File handling ──────────────────────────────────────────────────────────
const dropZone = document.getElementById('drop-zone');
const fi = document.getElementById('fi');

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0]; if (f) setFile(f);
});
fi.addEventListener('change', () => { if (fi.files[0]) setFile(fi.files[0]); });

function setFile(f) {
  selectedFile = f;
  const url = URL.createObjectURL(f);
  document.getElementById('thumb').src = url;
  document.getElementById('fname').textContent = f.name;
  document.getElementById('fsize').textContent = fmtSize(f.size);
  document.getElementById('dz-idle').style.display = 'none';
  document.getElementById('dz-file').style.display = '';
  dropZone.classList.add('has-file');
  document.getElementById('before-img').src = url;
  document.getElementById('cmp-before').src = url;
  showImgArea();
  resetResult();
  document.getElementById('btn-up').disabled = false;
}

function resetFile(e) {
  e.stopPropagation();
  selectedFile = null; fi.value = '';
  document.getElementById('dz-idle').style.display = '';
  document.getElementById('dz-file').style.display = 'none';
  dropZone.classList.remove('has-file');
  document.getElementById('btn-up').disabled = true;
  document.getElementById('empty-state').style.display = '';
  document.getElementById('img-headers').style.display = 'none';
  document.getElementById('img-area').style.display = 'none';
  resetResult();
}

function showImgArea() {
  document.getElementById('empty-state').style.display = 'none';
  document.getElementById('img-headers').style.display = '';
  document.getElementById('img-area').style.display = '';
  document.getElementById('side-by-side').style.display = 'flex';
  document.getElementById('compare-view').style.display = 'none';
  viewMode = 'side';
}

function resetResult() {
  hasResult = false;
  if (evtSrc) { evtSrc.close(); evtSrc = null; }
  document.getElementById('after-placeholder').style.display = '';
  document.getElementById('after-skeleton').style.display = 'none';
  document.getElementById('after-img-side').style.display = 'none';
  document.getElementById('prog-wrap').style.display = 'none';
  document.getElementById('tile-prog-wrap').style.display = 'none';
  document.getElementById('stats-wrap').style.display = 'none';
  document.getElementById('btn-dl').style.display = 'none';
  setProgress(0, '', false);
}

// ── Upscale ────────────────────────────────────────────────────────────────
async function doUpscale() {
  if (!selectedFile) return;
  document.getElementById('btn-up').disabled = true;
  resetResult();
  document.getElementById('prog-wrap').style.display = '';
  document.getElementById('prog-spinner').style.display = '';
  document.getElementById('after-placeholder').style.display = 'none';
  document.getElementById('after-skeleton').style.display = '';
  setProgress(3, 'Starting…', false);

  const fd = new FormData();
  fd.append('file',  selectedFile);
  fd.append('model', document.getElementById('model-sel').value);
  fd.append('tile',  document.getElementById('tile-sel').value);

  const res = await fetch('/upscale', { method:'POST', body:fd });
  const { task_id } = await res.json();
  taskId = task_id;

  if (evtSrc) evtSrc.close();
  evtSrc = new EventSource('/progress/' + taskId);
  evtSrc.onmessage = e => {
    const d = JSON.parse(e.data);

    // Tile grid
    if (d.total_tiles > 0) {
      document.getElementById('tile-prog-wrap').style.display = '';
      const ct = d.current_tile || 0, tt = d.total_tiles;
      document.getElementById('tile-counter').textContent = ct + ' / ' + tt;
      updateTileGrid('tile-grid', ct, tt);
    }

    setProgress(d.progress, d.label, d.done && !d.error);

    if (d.done) {
      evtSrc.close();
      document.getElementById('prog-spinner').style.display = 'none';
      document.getElementById('btn-up').disabled = false;
      document.getElementById('after-skeleton').style.display = 'none';

      if (d.error) {
        document.getElementById('prog-fill').className = 'prog-fill error';
        document.getElementById('prog-fill').style.width = '100%';
        document.getElementById('prog-lbl').textContent = '✗  ' + d.label;
        document.getElementById('prog-pct').textContent = '';
        document.getElementById('after-placeholder').style.display = '';
        document.getElementById('after-placeholder').innerHTML =
          '<div style="color:var(--red);font-size:13px;">Error — check console</div>';
      } else {
        showResult(task_id, d);
      }
    }
  };
}

function showResult(id, d) {
  hasResult = true;
  const url = '/result/' + id + '?t=' + Date.now();
  const sideAfter = document.getElementById('after-img-side');
  sideAfter.src = url; sideAfter.style.display = '';
  document.getElementById('cmp-after').src = url;
  document.getElementById('stat-in').textContent    = d.input_size;
  document.getElementById('stat-out').textContent   = d.output_size;
  document.getElementById('stat-scale').textContent = d.scale + '×';
  document.getElementById('stats-wrap').style.display = '';
  const dl = document.getElementById('btn-dl');
  dl.href = url; dl.style.display = '';
  setTimeout(switchToCompare, 500);
}

function switchToCompare() {
  if (!hasResult) return;
  document.getElementById('side-by-side').style.display = 'none';
  document.getElementById('compare-view').style.display = '';
  document.getElementById('img-headers').style.display = 'none';
  document.getElementById('toggle-btn').textContent = '⇄ Side-by-side';
  viewMode = 'compare';
  initSlider();
}

function toggleView() {
  if (viewMode === 'compare') {
    document.getElementById('compare-view').style.display = 'none';
    document.getElementById('side-by-side').style.display = 'flex';
    document.getElementById('img-headers').style.display = '';
    document.getElementById('toggle-btn').textContent = '⇄ Comparison slider';
    viewMode = 'side';
  } else {
    switchToCompare();
  }
}

// ── Tile grid ─────────────────────────────────────────────────────────────
function updateTileGrid(gridId, completed, total) {
  const grid = document.getElementById(gridId);
  const cols = Math.ceil(Math.sqrt(total));
  const rows = Math.ceil(total / cols);
  const needed = rows * cols;

  if (grid.children.length !== needed) {
    grid.style.gridTemplateColumns = 'repeat(' + cols + ', 1fr)';
    grid.innerHTML = '';
    for (let i = 0; i < needed; i++) {
      const c = document.createElement('div');
      c.className = 'tile-cell';
      grid.appendChild(c);
    }
  }

  const cells = grid.children;
  for (let i = 0; i < cells.length; i++) {
    if (i < completed)      cells[i].className = 'tile-cell done';
    else if (i === completed) cells[i].className = 'tile-cell active';
    else                    cells[i].className = 'tile-cell';
  }
}

// ── Progress helpers ───────────────────────────────────────────────────────
function setProgress(pct, label, done) {
  const fill = document.getElementById('prog-fill');
  fill.style.width = pct + '%';
  fill.className = 'prog-fill' + (done ? ' done' : '');
  document.getElementById('prog-lbl').textContent = label;
  document.getElementById('prog-pct').textContent = pct > 0 ? pct + '%' : '';
}

function fmtSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(0) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

// ── Comparison slider ──────────────────────────────────────────────────────
function initSlider() {
  const wrap   = document.getElementById('compare-wrap');
  const clip   = document.getElementById('cmp-after-clip');
  const handle = document.getElementById('cmp-handle');

  function setPos(x) {
    const r = wrap.getBoundingClientRect();
    const p = Math.max(5, Math.min(95, (x - r.left) / r.width * 100));
    clip.style.clipPath = 'inset(0 ' + (100 - p) + '% 0 0)';
    handle.style.left = p + '%';
  }

  wrap.onmousedown = e => { _sliderDragging = true; setPos(e.clientX); e.preventDefault(); };
  window.onmousemove = e => { if (_sliderDragging) setPos(e.clientX); };
  window.onmouseup = () => { _sliderDragging = false; };
  wrap.ontouchstart = e => { _sliderDragging = true; setPos(e.touches[0].clientX); };
  window.ontouchmove = e => { if (_sliderDragging) setPos(e.touches[0].clientX); };
  window.ontouchend = () => { _sliderDragging = false; };

  const r = wrap.getBoundingClientRect();
  setPos(r.left + r.width * 0.5);
}

// ── Tab switching ──────────────────────────────────────────────────────────
function switchTab(tab) {
  const s = tab === 'single';
  document.getElementById('tab-single').className = 'tab-btn' + (s ? ' active' : '');
  document.getElementById('tab-batch').className  = 'tab-btn' + (s ? '' : ' active');
  document.getElementById('panel-single').style.display = s ? '' : 'none';
  document.getElementById('panel-batch').style.display  = s ? 'none' : '';
  document.getElementById('main-single').style.display  = s ? 'flex' : 'none';
  document.getElementById('main-batch').style.display   = s ? 'none' : 'flex';
}

// ── Batch ──────────────────────────────────────────────────────────────────
async function doBatch() {
  const inF  = document.getElementById('b-in').value.trim();
  const outF = document.getElementById('b-out').value.trim();
  if (!inF)  { alert('Enter an input folder path.'); return; }
  if (!outF) { alert('Enter an output folder path.'); return; }

  // Reset UI
  document.getElementById('batch-log').innerHTML =
    '<div style="padding:40px 0;text-align:center;color:var(--text3);font-size:12px;">Starting…</div>';
  document.getElementById('b-count-lbl').textContent      = '';
  document.getElementById('b-count-lbl-main').textContent = '';
  document.getElementById('b-current-name').textContent = '';
  document.getElementById('btn-batch').disabled = true;
  document.getElementById('b-prog-wrap').style.display = '';
  document.getElementById('b-prog-spinner').style.display = '';
  document.getElementById('b-tile-prog-wrap').style.display = 'none';
  document.getElementById('b-img-wrap').style.display = 'none';
  document.getElementById('b-idle-msg').style.display = '';
  document.getElementById('b-idle-text').textContent = 'Starting batch…';
  setBatchProgress(3, 'Starting…', false);

  const fd = new FormData();
  fd.append('in_folder',  inF);
  fd.append('out_folder', outF);
  fd.append('model', document.getElementById('b-model').value);
  fd.append('tile',  document.getElementById('b-tile').value);

  const res = await fetch('/batch', { method:'POST', body:fd });
  const { task_id } = await res.json();

  if (batchEvtSrc) batchEvtSrc.close();
  batchEvtSrc = new EventSource('/batch-progress/' + task_id);
  batchEvtSrc.onmessage = e => {
    const d = JSON.parse(e.data);
    setBatchProgress(d.progress, d.label, d.done && !d.error);

    if (d.total > 0) {
      document.getElementById('b-count-lbl').textContent      = d.done_count + ' / ' + d.total;
      document.getElementById('b-count-lbl-main').textContent = d.done_count + ' / ' + d.total;
    }

    // Tile grid for current image
    if (d.total_tiles > 0) {
      document.getElementById('b-tile-prog-wrap').style.display = '';
      const ct = d.current_tile || 0, tt = d.total_tiles;
      document.getElementById('b-tile-counter').textContent = ct + ' / ' + tt;
      updateTileGrid('b-tile-grid', ct, tt);
    } else {
      document.getElementById('b-tile-prog-wrap').style.display = 'none';
    }

    // Current image preview
    if (d.current_img_b64) {
      document.getElementById('b-current-img').src = 'data:image/jpeg;base64,' + d.current_img_b64;
      document.getElementById('b-img-wrap').style.display = 'flex';
      document.getElementById('b-idle-msg').style.display = 'none';
    }
    if (d.current_img_name) {
      document.getElementById('b-current-name').textContent = d.current_img_name;
    }

    // Log entries
    const logEl = document.getElementById('batch-log');
    if (d.new_logs && d.new_logs.length > 0) {
      // Clear "Starting..." placeholder on first real entry
      if (logEl.children.length === 1 && logEl.children[0].tagName !== 'DIV'.toUpperCase() ||
          logEl.textContent.includes('Starting') || logEl.textContent.includes('Waiting')) {
        logEl.innerHTML = '';
      }
    }
    (d.new_logs || []).forEach(entry => {
      // Transition existing processing row to ok/error
      if (entry.status === 'ok' || entry.status === 'error') {
        const processing = logEl.querySelectorAll('.log-row.processing');
        if (processing.length > 0) {
          const row = processing[processing.length - 1];
          row.className = 'log-row ' + entry.status;
          row.innerHTML = mkRowHtml(entry);
          logEl.scrollTop = logEl.scrollHeight;
          return;
        }
      }
      const row = document.createElement('div');
      row.className = 'log-row ' + entry.status;
      row.innerHTML = mkRowHtml(entry);
      logEl.appendChild(row);
      logEl.scrollTop = logEl.scrollHeight;
    });

    if (d.done) {
      batchEvtSrc.close();
      document.getElementById('btn-batch').disabled = false;
      document.getElementById('b-prog-spinner').style.display = 'none';
      document.getElementById('b-tile-prog-wrap').style.display = 'none';
      document.getElementById('b-img-wrap').style.display = 'none';
      document.getElementById('b-idle-msg').style.display = '';
      document.getElementById('b-idle-text').textContent = d.error ? 'Batch failed.' : 'Batch complete!';
      document.getElementById('b-current-name').textContent = '';
      if (d.error) {
        document.getElementById('b-prog-fill').className = 'prog-fill error';
        document.getElementById('b-prog-fill').style.width = '100%';
      }
    }
  };
}

function mkRowHtml(entry) {
  let icon = '';
  if (entry.status === 'ok') {
    icon = '<span style="color:var(--green);font-size:12px;flex-shrink:0;">✓</span>';
  } else if (entry.status === 'error') {
    icon = '<span style="color:var(--red);font-size:12px;flex-shrink:0;">✗</span>';
  } else if (entry.status === 'processing') {
    icon = '<div class="spinner" style="width:11px;height:11px;flex-shrink:0;border-width:1.5px;"></div>';
  } else {
    icon = '<span style="color:var(--text3);flex-shrink:0;font-size:14px;line-height:1;">·</span>';
  }
  return icon + '<span class="log-name">' + escHtml(entry.msg) + '</span>';
}

function setBatchProgress(pct, label, done) {
  const fill = document.getElementById('b-prog-fill');
  fill.style.width = pct + '%';
  fill.className = 'prog-fill' + (done ? ' done' : '');
  document.getElementById('b-prog-lbl').textContent = label;
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
</script>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:7860")
    uvicorn.run(webapp, host="127.0.0.1", port=7860, log_level="warning")
