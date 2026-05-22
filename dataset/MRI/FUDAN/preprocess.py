"""
Image-side preprocessing for the FUDAN clinical breast MRI cohort.

Pipeline:
    1. ``cp_data``       - collect raw volumes per patient into ``images/<pid>/``.
    2. ``monai_preprocess`` - resample / re-orient to a canonical RAS grid with
                              MONAI (int16, .nii.gz).
    3. ``rename_crop``      - per-modality foreground crop, padded to a common
                              size so all modalities share an aligned ROI.
    4. ``rename_registration`` - rigid registration of every modality to T1w
                                 with ANTs.

The label-building / VQA-generation pieces are not part of the public release.
The MRI volumes themselves require a Data Use Agreement (see the README).
"""
import os
import shutil
import json
import numpy as np
import torch
import pandas as pd
import nibabel as nib
import fire
import ants
from functools import wraps
from shutil import copy2
from typing import Optional, Literal
from ast import literal_eval
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Orientation,
    SaveImage, Spacing, Transform,
)
from mpire import WorkerPool


# =============================================================================
# Multiprocessing helper
# =============================================================================
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
# 1. Raw collection (per-patient layout)
# =============================================================================
def copy_one(src, dst):
    try:
        if not os.path.exists(src):
            return False, f"Missing: {src}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return True, src
    except Exception as e:
        return False, f"{src} -> {e}"


def cp_data(meta_csv='fdzl_meta.csv', out_dir='images', max_workers=32):
    """Collect the raw modality NIfTIs referenced by ``meta_csv`` into ``out_dir/<pid>/``.

    The meta CSV is expected to have columns:
        index (=pid), root, T1, T1w, T1dyn, T2_f, T2_w, DWI, ADC, eADC
    where each modality cell is either a relative path or a literal-eval list of paths.
    """
    meta = pd.read_csv(meta_csv).replace({pd.NA: None})
    tasks = []
    for _, row in tqdm(meta.iterrows(), total=len(meta)):
        root = row['root']
        pid = str(row['index'])
        for d in ['T1', 'T1w', 'T1dyn', 'T2_f', 'T2_w', 'DWI', 'ADC', 'eADC']:
            val = row.get(d)
            if val is None:
                continue
            try:
                srcs = literal_eval(val)
            except Exception:
                srcs = val
            if not isinstance(srcs, list):
                srcs = [srcs]
            for s in srcs:
                src = os.path.join(root, s)
                fname = s.replace(f'fdzl_{pid}-', '')
                dst = os.path.join(out_dir, pid, fname)
                if not os.path.exists(src):
                    raise FileNotFoundError(src)
                tasks.append((src, dst))

    ok, failed = 0, []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(copy_one, src, dst) for src, dst in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            success, msg = fut.result()
            if success:
                ok += 1
            else:
                failed.append(msg)
    print(f"Copied: {ok}    Failed: {len(failed)}")
    if failed:
        for f in failed[:10]:
            print("  ", f)


# =============================================================================
# 2. MONAI reformat (orient + resample + cast)
# =============================================================================
class ForceAffine(Transform):
    """Reset the affine to a clean diagonal so downstream code sees a canonical RAS frame."""

    def __init__(self, space=None):
        self.affine = None
        if space is not None:
            self.affine = torch.eye(3)
            for i, s in enumerate(space):
                self.affine[i, i] = s

    def __call__(self, img):
        if not hasattr(img, "meta") or not hasattr(img, "affine"):
            raise KeyError
        if self.affine is not None:
            img.meta["affine"][:3, :3] = self.affine
        img.meta["affine"][:, 3] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        return img


@multi_process(64)
def _monai_preprocess(path, outpath, spacing=(0.725, 0.725, 2)):
    outdir = os.path.dirname(outpath)
    outname = os.path.basename(outpath).split('.')[0]
    os.makedirs(outdir, exist_ok=True)
    transform = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(channel_dim='no_channel'),
        Orientation(axcodes='RAS'),
        Spacing(pixdim=spacing),
        ForceAffine(),
        SaveImage(
            output_dir=outdir,
            output_ext='.nii.gz',
            output_postfix='',
            separate_folder=False,
            output_dtype=np.int16,
            output_name_formatter=lambda metadict, saver: {"subject": outname},
        ),
    ])
    transform(path)


def monai_preprocess(input_dir='images', output_dir='images_format'):
    """Reformat every modality NIfTI under ``input_dir`` into a canonical RAS grid."""
    tasks = []
    for pid in os.listdir(input_dir):
        pid_dir = os.path.join(input_dir, pid)
        if not os.path.isdir(pid_dir):
            continue
        for img in os.listdir(pid_dir):
            if img.endswith('.json'):
                continue
            src = os.path.join(pid_dir, img)
            dst = os.path.join(output_dir, pid, img)
            tasks.append([src, dst])
    _monai_preprocess(tasks)


# =============================================================================
# 3. Per-modality foreground crop with shared ROI size
# =============================================================================
@multi_process(64)
def _copy_auxiliary_files(src_data: dict, dst_dir: str):
    """Rename / collect auxiliary modality files into one directory per patient."""
    new_data = {}
    for m, src in src_data.items():
        pid = src.split('/')[-2]
        dst = os.path.join(dst_dir, pid, f'{m}.nii.gz')
        os.makedirs(os.path.join(dst_dir, pid), exist_ok=True)
        new_data[m] = dst
        if os.path.exists(src) and not os.path.exists(dst):
            copy2(src, dst)
    return new_data


def copy_and_rename(meta_jsonl='qas_key_Qcat_close-ended_cleaned.jsonl',
                    src_root='images_format',
                    dst_root='images_rename'):
    """Rename ``<pid>/<file>.nii.gz`` into ``<pid>/<modality>.nii.gz``."""
    meta = pd.read_json(meta_jsonl, lines=True).replace({pd.NA: None, np.nan: None})
    meta = meta[['pid', 'images', 'modalities']].drop_duplicates(subset='pid')
    meta['images'] = meta['images'].map(
        lambda xs: [x.replace('/images/', f'/{src_root}/') for x in xs]
    )
    data = meta.apply(lambda x: dict(zip(x['modalities'], x['images'])), axis=1)
    tasks = [(d, dst_root) for d in data.to_list()]
    _copy_auxiliary_files(tasks)


def get_foreground_bbox(data: np.ndarray, threshold: float = 20) -> Optional[tuple]:
    """Foreground bounding box (min/max corners) for a single 3D array."""
    data_norm = ((data - data.min()) / (data.max() - data.min() + 1e-8) * 255).astype(np.uint8)
    mask = data_norm > threshold
    if not mask.any():
        return None
    coords = np.array(np.where(mask))
    min_coords = coords.min(axis=1)
    max_coords = coords.max(axis=1) + 1
    return tuple(min_coords), tuple(max_coords)


def merge_bboxes(bboxes: list, center_axes: tuple = (0, 2)) -> tuple:
    """Union of a list of foreground bboxes."""
    bboxes = [b for b in bboxes if b is not None]
    if not bboxes:
        raise ValueError("No valid bboxes to merge")
    mins = np.array([b[0] for b in bboxes])
    maxs = np.array([b[1] for b in bboxes])
    merged_min, merged_max = [], []
    for axis in range(3):
        merged_min.append(mins[:, axis].min())
        merged_max.append(maxs[:, axis].max())
    return tuple(merged_min), tuple(merged_max)


def pad_bbox_to_max_size(
    bbox: tuple,
    max_size: np.ndarray,
    data_shape: tuple,
    center_axes: tuple = (0, 2),
) -> tuple:
    """Pad a single bbox out to ``max_size`` on each axis (centered for ``center_axes``)."""
    min_coords, max_coords = np.array(bbox[0]), np.array(bbox[1])
    current_size = max_coords - min_coords
    new_min = min_coords.copy()
    new_max = max_coords.copy()
    for axis in range(3):
        pad_total = max_size[axis] - current_size[axis]
        if pad_total <= 0:
            continue
        if axis in center_axes:
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
        else:
            pad_before = 0
            pad_after = pad_total
        new_min[axis] = max(0, min_coords[axis] - pad_before)
        new_max[axis] = min(data_shape[axis], max_coords[axis] + pad_after)
        actual_size = new_max[axis] - new_min[axis]
        if actual_size < max_size[axis]:
            deficit = max_size[axis] - actual_size
            if new_min[axis] == 0:
                new_max[axis] = min(data_shape[axis], new_max[axis] + deficit)
            else:
                new_min[axis] = max(0, new_min[axis] - deficit)
    return tuple(new_min), tuple(new_max)


def crop_with_bbox(data: np.ndarray, bbox: tuple) -> np.ndarray:
    (z1, y1, x1), (z2, y2, x2) = bbox
    return data[z1:z2, y1:y2, x1:x2]


@multi_process(64)
def _crop_on_max(src_data: dict, dst_dir: str, threshold: float = 50, center_axes: tuple = (0, 2)):
    """For each patient: compute per-modality foreground bbox, pad all bboxes to
    the largest size across modalities, then crop each modality independently.
    Ensures all modalities of a patient share the same spatial extent.
    """
    sample_path = next(iter(src_data.values()))
    pid = sample_path.split('/')[-2]
    output_dir = os.path.join(dst_dir, pid)
    os.makedirs(output_dir, exist_ok=True)
    try:
        images, bboxes, affines = {}, {}, {}
        for modality, src_path in src_data.items():
            if not os.path.exists(src_path):
                continue
            nii = nib.load(src_path)
            data = nii.get_fdata()
            images[modality] = data
            affines[modality] = nii.affine
            bbox = get_foreground_bbox(data, threshold)
            if bbox is not None:
                bboxes[modality] = bbox
        if not bboxes:
            raise ValueError("No valid bboxes found")

        sizes = [np.array(b[1]) - np.array(b[0]) for b in bboxes.values()]
        max_size = np.max(sizes, axis=0)
        for modality, data in images.items():
            if modality not in bboxes:
                continue
            dst_path = os.path.join(output_dir, f'{modality}.nii.gz')
            expanded_bbox = pad_bbox_to_max_size(
                bbox=bboxes[modality],
                max_size=max_size,
                data_shape=data.shape,
                center_axes=center_axes,
            )
            cropped = crop_with_bbox(data, expanded_bbox)
            nib.save(nib.Nifti1Image(cropped.astype(np.int16), affines[modality]), dst_path)
        return {'pid': pid, 'status': 'DONE', 'max_size': tuple(max_size)}
    except Exception as e:
        print(f"[ERROR] {pid}: {e}")
        return {'pid': pid, 'status': str(e)}


def rename_crop(meta_jsonl='qas_key_Qcat_close-ended_cleaned.jsonl',
                src_root='images_rename',
                dst_root='images_crop'):
    """Foreground-crop every modality so all modalities of a patient share an ROI."""
    meta = pd.read_json(meta_jsonl, lines=True).replace({pd.NA: None, np.nan: None})
    meta = meta[['pid', 'images', 'modalities']].drop_duplicates(subset='pid')
    meta['images'] = meta['images'].map(
        lambda xs: [x.replace('/images/', f'/{src_root}/') for x in xs]
    )
    meta = meta[meta['modalities'].map(lambda x: 'T1w' in x)]
    data = meta.apply(lambda x: dict(zip(x['modalities'], x['images'])), axis=1)
    tasks = [(d, dst_root) for d in data.to_list()]
    _crop_on_max(tasks)


# =============================================================================
# 4. ANTs rigid registration (every modality -> T1w)
# =============================================================================
def _ants_register(
    moving,
    fixed,
    output_path: str = None,
    transform_type: str = 'Affine',
    existing_reg: Optional[dict] = None,
    interpolator: Literal['linear', 'nearestNeighbor', 'bSpline'] = 'linear',
    output_dtype: np.dtype = np.int16,
) -> Optional[dict]:
    """Register ``moving`` to ``fixed`` with ANTs and save the warped output."""
    if isinstance(moving, str):
        if not os.path.exists(moving):
            raise FileNotFoundError(f"Moving image not found: {moving}")
        moving = ants.image_read(moving)
    if isinstance(fixed, str):
        if not os.path.exists(fixed):
            raise FileNotFoundError(f"Fixed image not found: {fixed}")
        fixed = ants.image_read(fixed)

    if existing_reg is None:
        existing_reg = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform=transform_type,
        )
    warped = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=existing_reg['fwdtransforms'],
        interpolator=interpolator,
    )
    warped = ants.from_numpy(
        data=warped.numpy().astype(output_dtype),
        origin=warped.origin,
        spacing=warped.spacing,
        direction=warped.direction,
    )
    warped.to_filename(output_path)
    return existing_reg


@multi_process(32)
def ants_register(patient_data: dict, save_dir: str) -> dict:
    """Register every modality of a patient to that patient's T1w volume."""
    patient_dir = os.path.dirname(list(patient_data.values())[0])
    pid = os.path.basename(patient_dir)
    output_dir = os.path.join(save_dir, pid)
    os.makedirs(output_dir, exist_ok=True)
    fixed_path = os.path.join(patient_dir, 'T1w.nii.gz')
    try:
        for m in patient_data.keys():
            _ants_register(
                moving=os.path.join(patient_dir, f'{m}.nii.gz'),
                fixed=fixed_path,
                output_path=os.path.join(output_dir, f'{m}.nii.gz'),
            )
        return {'path': output_dir, 'status': 'DONE'}
    except Exception as e:
        print(f"[ERROR] {patient_dir}: {e}")
        return {'path': output_dir, 'status': str(e)}


def rename_registration(meta_jsonl='qas_key_Qcat_close-ended_cleaned.jsonl',
                        src_root='images_rename',
                        dst_root='images_reg'):
    """Rigid-register every modality of every patient to its T1w."""
    meta = pd.read_json(meta_jsonl, lines=True).replace({pd.NA: None, np.nan: None})
    meta = meta[['pid', 'images', 'modalities']].drop_duplicates(subset='pid')
    meta['images'] = meta['images'].map(
        lambda xs: [x.replace('/images/', f'/{src_root}/') for x in xs]
    )
    meta = meta[meta['modalities'].map(lambda x: 'T1w' in x)]
    data = meta.apply(lambda x: dict(zip(x['modalities'], x['images'])), axis=1)
    tasks = [(d, dst_root) for d in data.to_list()]
    errors = pd.DataFrame(ants_register(tasks))
    errors = errors[errors['status'] != 'DONE']
    errors.to_csv('reg_errors.csv', index=False)


if __name__ == "__main__":
    # Expose every step as a CLI sub-command:
    #   python preprocess.py cp_data
    #   python preprocess.py monai_preprocess
    #   python preprocess.py copy_and_rename
    #   python preprocess.py rename_crop
    #   python preprocess.py rename_registration
    fire.Fire()
