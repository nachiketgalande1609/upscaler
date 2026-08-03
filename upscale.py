#!/usr/bin/env python3
"""CLI tool to upscale images using Real-ESRGAN."""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

import torch
_cores = os.cpu_count() or 4
torch.set_num_threads(max(1, _cores // 2))

SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.webp'}

MODEL_URLS = {
    4: (
        'RealESRGAN_x4plus',
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    ),
    2: (
        'RealESRGAN_x2plus',
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
    ),
}


def _download_model(model_name: str, url: str, model_dir: Path) -> Path:
    model_path = model_dir / f'{model_name}.pth'
    if model_path.exists():
        return model_path

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {model_name} weights (~64 MB) — this only happens once...')

    def _hook(count, block_size, total_size):
        if total_size > 0:
            pct = min(int(count * block_size * 100 / total_size), 100)
            filled = pct // 2
            print(f'\r  [{"#" * filled}{" " * (50 - filled)}] {pct}%', end='', flush=True)

    urllib.request.urlretrieve(url, model_path, _hook)
    print()
    return model_path


def _build_upsampler(scale: int, model_dir: Path, tile: int, half: bool):
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError as exc:
        print(
            f'Error: {exc}\n'
            'Make sure you have activated the virtual environment and run:\n'
            '  pip install -r requirements.txt',
            file=sys.stderr,
        )
        sys.exit(1)

    model_name, url = MODEL_URLS[scale]
    model_path = _download_model(model_name, url, model_dir)

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=23, num_grow_ch=32, scale=scale,
    )
    return RealESRGANer(
        scale=scale,
        model_path=str(model_path),
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=half,
    )


def _upscale_image(upsampler, src: Path, dst: Path) -> None:
    import cv2
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f'Could not decode image (file may be corrupted): {src.name}')
    output, _ = upsampler.enhance(img, outscale=upsampler.scale)
    cv2.imwrite(str(dst), output)


def _collect_images(path: Path) -> list:
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{path.suffix}'. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )
        return [path]
    if path.is_dir():
        imgs = sorted(
            p for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS
        )
        if not imgs:
            raise ValueError(f'No supported images found in: {path}')
        return imgs
    raise FileNotFoundError(f'Input path not found: {path}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Upscale images to 4K with AI enhancement using Real-ESRGAN.',
        epilog='''
examples:
  python upscale.py photo.jpg output/
  python upscale.py wallpapers/ upscaled/ --scale 4
  python upscale.py big_photo.png out/ --scale 2 --tile 512
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('input', help='Image file or folder containing images')
    parser.add_argument('output', help='Output folder (created if it does not exist)')
    parser.add_argument(
        '--scale', type=int, choices=[2, 4], default=4,
        help='Upscale factor: 2x or 4x (default: 4)',
    )
    parser.add_argument(
        '--tile', type=int, default=0,
        help=(
            'Tile size for processing large images in chunks (default: 0 = no tiling). '
            'Set to 512 or 256 if you run out of memory.'
        ),
    )
    parser.add_argument(
        '--half', action='store_true',
        help='Use half-precision (fp16) — faster on CUDA GPU, may be unstable on CPU',
    )
    parser.add_argument(
        '--model-dir', default='models',
        help='Directory to cache downloaded model weights (default: models/)',
    )
    args = parser.parse_args()

    # Collect input images
    try:
        images = _collect_images(Path(args.input))
    except (FileNotFoundError, ValueError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Initializing Real-ESRGAN {args.scale}x model...')
    try:
        upsampler = _build_upsampler(args.scale, Path(args.model_dir), args.tile, args.half)
    except Exception as exc:
        print(f'Error loading model: {exc}', file=sys.stderr)
        sys.exit(1)

    from tqdm import tqdm  # imported here so import errors surface clearly above

    n = len(images)
    plural = 's' if n != 1 else ''
    print(f'Processing {n} image{plural}...\n')

    errors: list = []
    with tqdm(images, unit='img', disable=(n == 1)) as bar:
        for src in bar:
            # WEBP → PNG to preserve lossless quality on output
            out_ext = '.png' if src.suffix.lower() == '.webp' else src.suffix.lower()
            dst = output_dir / (src.stem + out_ext)
            try:
                _upscale_image(upsampler, src, dst)
                msg = f'-> {dst}'
                if n == 1:
                    print(msg)
                else:
                    tqdm.write(msg)
            except Exception as exc:
                errors.append((src, str(exc)))
                tqdm.write(f'  Error ({src.name}): {exc}', file=sys.stderr)

    ok = n - len(errors)
    print(f'\nDone — {ok}/{n} image{plural} upscaled successfully.')
    if errors:
        print(f'{len(errors)} error{("s" if len(errors) != 1 else "")} encountered:', file=sys.stderr)
        for src, msg in errors:
            print(f'  {src.name}: {msg}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
