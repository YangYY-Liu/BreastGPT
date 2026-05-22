"""
Histopathology (WSI) image processing for BreastGPT.

This script orchestrates two image-side tasks per WSI source
(TCGA-BRCA, BCNB, HISTAI-breast):

    1. Tile each WSI into 512x512 patches at 20x magnification with 0 overlap
       using the `Trident <https://github.com/mahmoodlab/trident>`_ pipeline
       (see ``run_batch_of_slides.py`` for the actual driver).
    2. Optionally save a representative subset of patches as PNGs for QA /
       resize-script consumption (``extract_patch_from_trident``).

The label / VQA pieces of the original pipeline have been dropped from the
public release.
"""
import os
import h5py
import openslide
import numpy as np
from PIL import Image
from shutil import copy2, move
from pathlib import Path
from tqdm import tqdm
from functools import wraps
from mpire import WorkerPool

Image.MAX_IMAGE_PIXELS = None


def multi_process(num_process, test=False):
    def decorator(func):
        @wraps(func)
        def wrapper(params):
            if test:
                return func(*params[0])
            with WorkerPool(num_process) as pool:
                return pool.map(func, params, progress_bar=True)
        return wrapper
    return decorator


# =============================================================================
# Filesystem helpers (collect / flatten WSIs)
# =============================================================================
@multi_process(32)
def multi_copy(src, dst):
    try:
        if not os.path.exists(src):
            return False, f"Missing: {src}"
        if os.path.exists(dst):
            return True, src
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        copy2(src, dst)
        return True, src
    except Exception as e:
        return False, f"{src} -> {e}"


@multi_process(32)
def multi_move(src, dst):
    if not os.path.exists(src):
        return False, f"Missing: {src}"
    if os.path.exists(dst):
        return True, src
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    move(src, dst)
    return True, src


def search_wsi(dirpath, ext=("tiff", "tif", "svs", "ndpi")):
    """Recursively list WSIs under ``dirpath``."""
    dirpath = Path(dirpath)
    files = []
    for e in ext:
        files.extend(dirpath.rglob(f"*.{e}"))
        files.extend(dirpath.rglob(f"*.{e.upper()}"))
    return sorted(set(files))


def flat_wsi(dirpath, keep_parent=True, copy=True):
    """Flatten nested WSI folders into one ``<dirpath>_FLAT`` directory."""
    new_files, pids, tasks = [], [], []
    dirpath = Path(dirpath)
    new_dir = dirpath.parent / f"{dirpath.name}_FLAT"
    files = search_wsi(dirpath)
    for f in tqdm(files):
        tar = new_dir / (f"{f.parent.name}_{f.name}" if keep_parent else f.name)
        new_files.append(str(tar))
        pids.append(f.parent.name)
        tasks.append((str(f), str(tar)))
    if copy:
        multi_copy(tasks)
    import pandas as pd
    return pd.DataFrame({'wsi': new_files, 'pid': pids}, dtype=str)


# =============================================================================
# Trident wrapper: per-source tile-and-feature pipeline
# =============================================================================
def _run_trident(image_dir, job_dir, mpp_hint=0.505):
    """Tile every WSI in ``image_dir`` and extract CONCH v1.5 features into ``job_dir``."""
    from run_batch_of_slides import run_batch
    run_batch(
        job_dir=job_dir,
        wsi_dir=image_dir,
        task='all',
        gpu=0,
        skip_errors=False,
        max_workers=None,
        batch_size=64,
        wsi_cache=None,
        cache_batch_size=32,
        wsi_ext=None,
        custom_mpp_keys={'mpp': mpp_hint},
        custom_list_of_wsis=None,
        reader_type=None,
        search_nested=False,
        segmenter='hest',
        seg_conf_thresh=0.5,
        remove_holes=False,
        remove_artifacts=False,
        remove_penmarks=False,
        seg_batch_size=None,
        mag=20,
        patch_size=512,
        overlap=0,
        min_tissue_proportion=0.,
        coords_dir=None,
        patch_encoder='conch_v15',
        patch_encoder_ckpt_path=None,
        slide_encoder=None,
        feat_batch_size=None,
    )


def process_bcnb(image_dir='./BCNB/WSI', job_dir='./BCNB/Processed'):
    _run_trident(image_dir=image_dir, job_dir=job_dir, mpp_hint=0.505)


def process_histai(image_dir='./HISTAI-breast/WSI', job_dir='./HISTAI-breast/Processed'):
    _run_trident(image_dir=image_dir, job_dir=job_dir, mpp_hint=0.505)


def process_TCGA(image_dir='./TCGA-BRCA/WSI', job_dir='./TCGA-BRCA/Processed'):
    _run_trident(image_dir=image_dir, job_dir=job_dir, mpp_hint=0.25)


# =============================================================================
# Optional: dump sample patches as PNGs from Trident coords
# =============================================================================
def open_wsi(filepath):
    try:
        return openslide.OpenSlide(filepath), 'openslide'
    except openslide.lowlevel.OpenSlideUnsupportedFormatError:
        return Image.open(filepath).convert('RGB'), 'pil'


@multi_process(32)
def save_patch(savedir, filepath, coordpath, sample_num=64):
    """Sample ``sample_num`` random patches from a WSI and dump them as PNGs."""
    filename = os.path.basename(filepath).split('.')[0]
    savedir = os.path.join(savedir, filename)
    os.makedirs(savedir, exist_ok=True)
    with h5py.File(coordpath, 'r') as f:
        coords = f['coords'][:]
        if sample_num < len(coords):
            indices = np.random.choice(len(coords), sample_num, replace=False)
            coords = coords[indices]
        patch_size = f['coords'].attrs.get('patch_size', 512)
        patch_level = f['coords'].attrs.get('patch_level', 0)
    wsi, mode = open_wsi(filepath)
    for i, (x, y) in enumerate(coords):
        savepath = os.path.join(savedir, f'{i}.png')
        if mode == 'openslide':
            patch = wsi.read_region((x, y), patch_level, (patch_size, patch_size)).convert('RGB')
        else:
            patch = wsi.crop((x, y, x + patch_size, y + patch_size))
        patch.save(savepath)


def extract_patch_from_trident(basedir='.', sample_num=30):
    """Dump ``sample_num`` patches per slide for each of BCNB / HISTAI-breast / TCGA-BRCA.

    Expects ``basedir`` to contain ``<dataset>/Processed/20x_512px_0px_overlap/patches/`` (the
    h5 coordinate files produced by Trident) and ``<dataset>/WSI[_FLAT]/`` (the WSIs).
    """
    for datasetdir in ['BCNB', 'HISTAI-breast', 'TCGA-BRCA']:
        datapath = os.path.join(basedir, datasetdir, 'Processed/20x_512px_0px_overlap/patches')
        savedir = os.path.join(basedir, datasetdir, f'Processed/20x_512px_0px_overlap/images_{sample_num}')
        if not os.path.isdir(datapath):
            continue
        task = []
        for pathes in os.listdir(datapath):
            endfix = {'BCNB': '.jpg', 'HISTAI-breast': '.tiff', 'TCGA-BRCA': '.svs'}[datasetdir]
            coordpath = os.path.join(datapath, pathes)
            wsi_flat = os.path.join(basedir, datasetdir, 'WSI_FLAT')
            wsi_root = wsi_flat if os.path.exists(wsi_flat) else os.path.join(basedir, datasetdir, 'WSI')
            filepath = os.path.join(wsi_root, pathes.replace('_patches.h5', endfix))
            if not os.path.exists(filepath):
                print(f"missing: {filepath}")
                continue
            task.append((savedir, filepath, coordpath, sample_num))
        save_patch(task)


if __name__ == "__main__":
    import fire
    # Available sub-commands:
    #   python preprocess.py process_bcnb --image_dir ... --job_dir ...
    #   python preprocess.py process_histai --image_dir ... --job_dir ...
    #   python preprocess.py process_TCGA --image_dir ... --job_dir ...
    #   python preprocess.py extract_patch_from_trident --basedir . --sample_num 30
    fire.Fire()
