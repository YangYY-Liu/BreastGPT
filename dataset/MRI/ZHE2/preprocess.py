"""
Image-side preprocessing for the ZHE2 (clinical) breast MRI cohort.

Pipeline:
    1. Reformat every modality with MONAI (RAS, fixed spacing, int16, .nii.gz).

The label-building / VQA-generation steps are not part of the public release;
the MRI volumes themselves are governed by a Data Use Agreement and are not
freely available (see the README).
"""
import os
import fire
import numpy as np
import torch
from functools import wraps
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Orientation,
    SaveImage, Spacing, Transform,
)
from mpire import WorkerPool


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
    if os.path.exists(outpath):
        return
    outdir = os.path.dirname(outpath)
    outname = os.path.basename(outpath).split('.')[0]
    os.makedirs(outdir, exist_ok=True)
    transform = Compose([
        LoadImage(reader='ITKReader', image_only=True),
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
    """Format every modality nii.gz under ``input_dir/<pid>/`` into ``output_dir/<pid>/``."""
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


if __name__ == "__main__":
    fire.Fire()
