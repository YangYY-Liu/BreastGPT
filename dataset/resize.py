"""
Second-stage resize: convert per-dataset preprocessed images into the
canonical input sizes expected by BreastGPT.

Each per-dataset `preprocess.py` produces a `Processed/` folder of raw-resolution
PNG / NIfTI files. This script then resizes them to a fixed visual-token budget
so they can be fed to the model:

  - 2D images (.png / .jpg)        : aspect-preserving rescale into
                                     `image_max_token_num * 32 * 32` pixels
                                     (default 1024 tokens). Use `--maxsize` to
                                     instead clamp the longest edge to a pixel
                                     value (256 for histopathology patches).
  - 3D volumes (.nii / .nii.gz)    : longest in-plane edge -> 384,
                                     z-axis -> 48 slices, saved as int16.

CLI:
    python resize.py one <src> <dst> [--maxsize N]
    python resize.py tree <in_root> <out_root> [--maxsize N]
"""
import math
import os
from shutil import copy2

import fire
import numpy as np
from mpire import WorkerPool
from monai.transforms import Compose, EnsureChannelFirst, LoadImage, Resize, SaveImage
from PIL import Image, PngImagePlugin

# Some pathology PNGs carry oversized tEXt chunks.
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024


def _ensure_parent_dir(p: str) -> None:
    os.makedirs(os.path.dirname(p), exist_ok=True)


def _monai_resize(path: str, outpath: str) -> None:
    """3D volumes (CT / MRI): longest in-plane edge -> 384, depth -> 48."""
    outdir = os.path.dirname(outpath)
    outname = os.path.basename(outpath).split(".")[0]
    os.makedirs(outdir, exist_ok=True)
    transform = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(channel_dim="no_channel"),
        Resize(size_mode="longest", spatial_size=384),
        Resize(spatial_size=(-1, -1, 48)),
        SaveImage(
            output_dir=outdir,
            output_ext=".nii.gz",
            output_postfix="",
            separate_folder=False,
            output_dtype=np.int16,
            output_name_formatter=lambda metadict, saver: {"subject": outname},
        ),
    ])
    transform(path)


def _image_resize(
    path: str,
    outpath: str,
    image_max_token_num: int = 1024,
    maxsize: int | None = None,
) -> None:
    """2D images: BUS / Mammography (token-budget) or WSI patches (`--maxsize 256`)."""
    pixel_budget = image_max_token_num * 32 * 32
    with Image.open(path) as im:
        w, h = im.size
        if maxsize is None:
            area = w * h
            if area > pixel_budget:
                scale = math.sqrt(pixel_budget / float(area))
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                im = im.resize((new_w, new_h), resample=Image.BICUBIC)
        else:
            longest = max(w, h)
            if longest > maxsize:
                scale = maxsize / float(longest)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                im = im.resize((new_w, new_h), resample=Image.BICUBIC)
        im.save(outpath)


def _resize_one(src: str, dst: str, maxsize: int | None) -> None:
    if not os.path.exists(src):
        print(f"MISSING {src}")
        return
    if os.path.exists(dst):
        return
    _ensure_parent_dir(dst)
    try:
        s = src.lower()
        if s.endswith((".nii.gz", ".nii")):
            _monai_resize(src, dst)
        elif s.endswith((".png", ".jpg", ".jpeg")):
            _image_resize(src, dst, maxsize=maxsize)
        else:
            copy2(src, dst)
    except Exception as e:
        print(f"ERROR {src}: {type(e).__name__}: {e}")


def one(src: str, dst: str, maxsize: int | None = None) -> None:
    """Resize a single file."""
    _resize_one(src, dst, maxsize)


def tree(in_root: str, out_root: str, maxsize: int | None = None, num_workers: int = 64) -> None:
    """Mirror `in_root` under `out_root`, resizing each supported file."""
    tasks = []
    for cur, _, files in os.walk(in_root):
        for f in files:
            src = os.path.join(cur, f)
            rel = os.path.relpath(src, in_root)
            tasks.append((src, os.path.join(out_root, rel), maxsize))
    print(f"Resizing {len(tasks)} files with {num_workers} workers ...")
    with WorkerPool(num_workers) as pool:
        pool.map(_resize_one, tasks, progress_bar=True)


if __name__ == "__main__":
    fire.Fire({"one": one, "tree": tree})
