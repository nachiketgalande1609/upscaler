# Image Upscaler

AI-powered CLI tool that upscales images to 4K resolution using **Real-ESRGAN**.
Supports JPG, PNG, and WEBP. Model weights (~64 MB) are downloaded automatically on first run.

## Requirements

- Python 3.9 or newer
- ~3 GB disk space (PyTorch + model weights)
- GPU optional — runs on CPU, but much slower (~30–120 s per image vs ~1–5 s on GPU)

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install PyTorch (and other straightforward deps)
pip install torch torchvision opencv-python Pillow tqdm

# 4. Install basicsr from source
#    (the PyPI release has a build bug on Python 3.13+; installing from
#    the GitHub repo with --no-build-isolation works around it)
pip install git+https://github.com/XPixelGroup/BasicSR.git --no-build-isolation

# 5. Install Real-ESRGAN and remaining dependencies
pip install realesrgan facexlib gfpgan
```

> **GPU users (CUDA):** For significantly faster processing, install the CUDA-enabled
> version of PyTorch first — see [pytorch.org/get-started](https://pytorch.org/get-started/locally/)
> — then continue from step 4 above.

## Usage

```
python upscale.py <input> <output_folder> [options]
```

### Arguments

| Argument | Description |
|---|---|
| `input` | Image file **or** folder of images |
| `output` | Output folder (created automatically if missing) |
| `--scale 2\|4` | Upscale factor — `2` or `4` (default: `4`) |
| `--tile N` | Tile size for very large images (e.g. `512`). Use if you run out of memory. `0` = disabled (default) |
| `--half` | Half-precision (fp16) — faster on CUDA GPU, not recommended for CPU |
| `--model-dir DIR` | Where to cache model weights (default: `models/`) |

### Examples

```bash
# Upscale a single photo to 4x
python upscale.py photo.jpg output/

# Upscale an entire folder of wallpapers
python upscale.py wallpapers/ upscaled/

# 2x upscale (better for images that are already large)
python upscale.py photo.png output/ --scale 2

# Large image — process in tiles to avoid out-of-memory
python upscale.py big_scan.jpg output/ --tile 512

# GPU with half-precision for maximum speed
python upscale.py photos/ output/ --half
```

## Supported Formats

| Format | Input | Output |
|---|---|---|
| JPEG (`.jpg`, `.jpeg`) | Yes | `.jpg` |
| PNG (`.png`) | Yes | `.png` |
| WebP (`.webp`) | Yes | `.png` (lossless) |

## Models

| Scale | Model | Auto-downloaded from |
|---|---|---|
| 4x | `RealESRGAN_x4plus` | GitHub Releases (xinntao/Real-ESRGAN) |
| 2x | `RealESRGAN_x2plus` | GitHub Releases (xinntao/Real-ESRGAN) |

Models are saved in the `models/` folder and reused on subsequent runs.

## Performance Tips

- **CPU:** Expect 30–120 seconds per image depending on resolution.
- **CUDA GPU:** 1–5 seconds per image. Use `--half` for an extra speed boost.
- **Large images (>4K input):** Add `--tile 512` to process in chunks and avoid OOM errors.
- **Batch processing:** Pass a folder as input; a progress bar will track all files.

## Troubleshooting

**`ModuleNotFoundError`** — Make sure you activated the virtual environment before running.

**Out of memory** — Add `--tile 512` (or `--tile 256` for less VRAM).

**Slow on CPU** — This is expected. Consider running on a machine with a CUDA GPU, or use a cloud GPU notebook (Google Colab, Kaggle).

**Corrupted output / black image** — The input may be in a colour space Real-ESRGAN doesn't handle (e.g. CMYK). Convert to RGB/sRGB first with an image editor.
