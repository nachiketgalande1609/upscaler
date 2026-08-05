# AI Image Upscaler

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A local AI image upscaler with a browser-based UI. Runs entirely on your machine — no cloud, no subscriptions. Powered by Real-ESRGAN and Real-HAT transformer models, served through a FastAPI backend with a dark glass single-page UI. The browser opens automatically on startup.

> **Screenshots coming soon**

---

## Table of Contents

- [Features](#features)
- [Models](#models)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [File Overview](#file-overview)
- [Troubleshooting](#troubleshooting)

---

## Features

- **Single image mode** — drag-and-drop upload with an instant before/after side-by-side preview; switches to an interactive comparison slider on completion
- **Batch mode** — folder-to-folder processing with a live image preview of the current file and a tile-grid progress display
- **Real-time progress** — Server-Sent Events (SSE) push per-tile progress bars and an animated tile grid to the browser without polling
- **Comparison slider** — drag a handle to compare original and upscaled output in-place
- **Configurable tiling** — choose tile size (256 / 512 / 1024 / Auto) to match your available VRAM
- **Device auto-detection** — uses CUDA fp16/bf16 on GPU, fp32 on CPU; active device shown in the header
- **Model caching** — models remain in memory after first load; no reload overhead between images
- **Auto model download** — weights download automatically on first use from GitHub Releases / HuggingFace

---

## Models

| Model | Architecture | Scale | Quality | Speed | VRAM (512 tile) | Weight size |
|---|---|---|---|---|---|---|
| **Real-HAT 4x** | Hybrid Attention Transformer (CVPR 2023) | 4x | Best | Slowest (~5–10× vs ESRGAN) | ~8 GB | ~160 MB |
| **Real-ESRGAN 4x** | RRDBNet 23 blocks | 4x | Very good | Fast | ~4 GB | ~67 MB |
| **Real-ESRGAN 4x Lite** | RRDBNet 6 blocks (anime-optimised) | 4x | Good (anime) | Fastest | ~2 GB | ~17 MB |
| **Real-ESRGAN 2x** | RRDBNet 23 blocks | 2x | Very good | Fast | ~4 GB | ~67 MB |

**Choosing a model:**

- Use **Real-HAT 4x** when quality matters most and you have time and VRAM to spare. It is a transformer-based model from a CVPR 2023 paper and produces noticeably sharper, more faithful textures than convolutional alternatives.
- Use **Real-ESRGAN 4x** as the everyday default — excellent quality at a fraction of the compute cost.
- Use **Real-ESRGAN 4x Lite** for anime/illustration content or when speed is critical.
- Use **Real-ESRGAN 2x** when the source image is already fairly large and a 4x upscale would be excessive.

### Precision handling

| Model family | CUDA — Ampere+ (RTX 3000/4000, sm_80+) | CUDA — older | CPU |
|---|---|---|---|
| Real-ESRGAN | fp16 | fp16 | fp32 |
| Real-HAT | **bfloat16** | fp16 | fp32 |

Real-HAT runs in bfloat16 on Ampere+ GPUs instead of fp16. HAT's attention mechanism (softmax) overflows in fp16, producing NaN values that result in a black output image. bfloat16 retains fp32's exponent range while halving memory use, preventing the overflow. Older CUDA GPUs that do not support bfloat16 fall back to fp16, which is generally stable for those architectures.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | Any CUDA-capable NVIDIA GPU | 8 GB VRAM (for HAT at 512-tile) |
| VRAM | 4 GB (ESRGAN, small tiles) | 6–8 GB |
| RAM | 8 GB | 16 GB |
| Disk | ~300 MB (one model) | ~1 GB (all models) |
| CPU | Any modern x86-64 | — (CPU mode is supported but slow) |

CPU mode works but expect minutes per image rather than seconds. ESRGAN Lite is the most practical choice in CPU-only environments.

---

## Installation

Python **3.10 or 3.11** is strongly recommended. basicsr has build issues on Python 3.13+.

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/upscaler.git
cd upscaler
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install PyTorch

**GPU (CUDA 12.1):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**CPU only:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

For other CUDA versions or platforms, see [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

### Step 4 — Install BasicSR from source

The PyPI release of basicsr has a known build failure on Python 3.10+. Install directly from the upstream repository:

```bash
pip install git+https://github.com/XPixelGroup/BasicSR.git --no-build-isolation
```

### Step 5 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

### Step 6 — Start the server

```bash
python app.py
```

The server starts on `http://localhost:7860` and opens the browser automatically. Model weights download on first use.

---

## Usage

### Single image

1. Open `http://localhost:7860` (auto-opens on startup).
2. Drag an image onto the upload area or click to browse.
3. Select a model and tile size.
4. Click **Upscale**. A per-tile progress bar and animated tile grid update in real time via SSE.
5. When complete, a side-by-side before/after view appears, then switches to an interactive comparison slider.
6. Click **Download** to save the result.

### Batch processing

1. Switch to **Batch** mode.
2. Enter an input folder path and an output folder path.
3. Select model and tile size.
4. Click **Start Batch**. A live preview of the current file and a tile-grid progress display update as each image is processed.

### Supported formats

| Format | Extensions |
|---|---|
| JPEG | `.jpg`, `.jpeg` |
| PNG | `.png` |
| WebP | `.webp` |

---

## Architecture

### Tiling

Large images are split into overlapping tiles before inference. Each tile is processed independently and the results are blended at the seams. Tile size controls the peak VRAM usage: smaller tiles use less memory but add more blending seams; larger tiles produce seamless output at higher VRAM cost. The **Auto** option selects the largest tile size that fits in available VRAM.

```
Input image
    │
    ▼
Split into N×M tiles (with overlap)
    │
    ▼
Model inference per tile  ──── SSE progress events ──▶ Browser
    │
    ▼
Stitch tiles (blend overlap regions)
    │
    ▼
Output image
```

### Real-time progress via SSE

Progress updates are pushed from server to browser using the [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) protocol. The browser opens a persistent HTTP connection to `/progress/{task_id}` or `/batch-progress/{task_id}`; the server writes newline-delimited `data:` frames as each tile completes. No WebSocket handshake or polling is required.

Each SSE frame carries:

- `progress` — overall completion percentage (0–100)
- `current_tile` / `total_tiles` — tile index for the progress bar
- Batch mode additionally sends tile-grid state so the UI can animate each cell

### Model caching

After a model loads for the first time it stays resident in GPU/CPU memory for the lifetime of the server process. Switching between images of the same model incurs no reload overhead. Switching models evicts the previous model to free VRAM before loading the new one.

---

## API Reference

All endpoints are served on `http://localhost:7860`. Responses are JSON unless otherwise noted.

### `GET /`

Returns the web UI (HTML page).

---

### `POST /upscale`

Submit a single image for upscaling.

**Form fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | Yes | Image file (JPG, PNG, WebP) |
| `model` | string | Yes | Model key (see table below) |
| `tile` | integer | No | Tile size in pixels: `256`, `512`, `1024`, or `0` for auto |

**Model keys:**

| Key | Model |
|---|---|
| `realesr-general-x4v3` | Real-ESRGAN 4x |
| `realesr-animevideov3` | Real-ESRGAN 4x Lite |
| `RealESRGAN_x2plus` | Real-ESRGAN 2x |
| `hat` | Real-HAT 4x |

**Response:**

```json
{ "task_id": "abc123" }
```

---

### `GET /progress/{task_id}`

SSE stream for a single-image upscale task. Connect with `EventSource` in the browser or with any SSE-capable HTTP client.

**Event data (JSON):**

```json
{
  "progress": 42,
  "current_tile": 5,
  "total_tiles": 12,
  "status": "processing"
}
```

When `status` is `"done"` the stream closes. On error, `status` is `"error"` and a `message` field is included.

---

### `GET /result/{task_id}`

Download the upscaled image as a PNG file.

**Response:** `image/png` binary

---

### `POST /batch`

Submit a folder of images for batch processing.

**Form fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `in_folder` | string | Yes | Absolute path to the input folder |
| `out_folder` | string | Yes | Absolute path to the output folder (created if missing) |
| `model` | string | Yes | Model key (see `/upscale` above) |
| `tile` | integer | No | Tile size (same options as `/upscale`) |

**Response:**

```json
{ "task_id": "xyz789" }
```

---

### `GET /batch-progress/{task_id}`

SSE stream for a batch task.

**Event data (JSON):**

```json
{
  "progress": 60,
  "current_file": "photo_003.jpg",
  "files_done": 3,
  "files_total": 5,
  "current_tile": 8,
  "total_tiles": 20,
  "tile_grid": [[true, true, false], [false, false, false]],
  "status": "processing"
}
```

`tile_grid` is a 2-D boolean array where `true` marks completed tiles for the animated tile-grid UI.

---

## File Overview

| File | Purpose |
|---|---|
| `app.py` | FastAPI server, model loading and caching, upscale logic, SSE endpoints, and the embedded HTML/CSS/JS frontend (~1300 lines) |
| `hat_arch.py` | Full Real-HAT architecture implementation: HAT, RHAG, WindowAttention, OCAB, and supporting modules |
| `requirements.txt` | Python dependencies |
| `upscale.py` | Legacy CLI script — not used by the web UI; kept for reference |

---

## Troubleshooting

### CUDA out of memory

Reduce the tile size. Start with 512; if it still fails, try 256.

```
Tile size: 1024 → 512 → 256
```

### Black output image with Real-HAT

This was caused by fp16 softmax overflow (NaN) in HAT's attention layers. It is fixed internally: the server automatically uses bfloat16 on Ampere+ GPUs. If you see this on an older GPU that does not support bfloat16, the fallback to fp16 may still trigger it on extreme inputs — try reducing tile size.

### `ModuleNotFoundError: No module named 'einops'`

```bash
pip install einops
```

### basicsr fails to build / install

Python 3.13 breaks several of basicsr's build steps. Downgrade to Python 3.10 or 3.11:

```bash
# Check current version
python --version

# Create a new venv with the correct interpreter
py -3.11 -m venv venv
```

Then repeat the installation steps from [Step 2](#step-2--create-a-virtual-environment).

### Real-HAT is much slower than Real-ESRGAN

This is expected. Real-HAT is a Hybrid Attention Transformer; transformer self-attention scales quadratically with sequence length. At 4x upscaling HAT is approximately 5–10× slower than Real-ESRGAN but produces noticeably better texture fidelity and sharpness, especially on complex patterns and fine detail.

### `ModuleNotFoundError: No module named 'basicsr'`

Make sure you installed BasicSR from source (Step 4) with `--no-build-isolation`, not from PyPI. The PyPI package (`basicsr`) does not install correctly on Python 3.10+.

### Server does not open the browser automatically

The server calls `webbrowser.open()` after binding. If your environment suppresses this (headless server, WSL without a display), navigate to `http://localhost:7860` manually.

---

## Dependencies

Full list in `requirements.txt`. Key packages:

| Package | Version | Purpose |
|---|---|---|
| `torch` | >=2.0.0 | Neural network inference |
| `torchvision` | >=0.15.0 | Image transforms |
| `basicsr` | git HEAD | SR model base (install from source) |
| `realesrgan` | >=0.3.0 | Real-ESRGAN model and pipeline |
| `facexlib` | >=0.3.0 | Face detection dependency |
| `gfpgan` | >=1.3.8 | Face restoration dependency |
| `opencv-python` | >=4.5.0 | Image I/O and processing |
| `Pillow` | >=9.0.0 | Image format support |
| `fastapi` | >=0.100.0 | Web framework |
| `uvicorn[standard]` | >=0.22.0 | ASGI server |
| `python-multipart` | >=0.0.6 | File upload parsing |
| `einops` | >=0.6.0 | Tensor rearrangement (required by HAT) |
| `requests` | >=2.28.0 | Model weight downloads |
| `tqdm` | >=4.60.0 | Progress bars |

---

## License

MIT
