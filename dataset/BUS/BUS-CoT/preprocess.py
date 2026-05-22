"""
BUS-CoT image-side processing.

The published BUS-CoT release already ships with per-lesion folders, each containing:
  <iid>@raw.png        : original ultrasound frame
  <iid>@mask*.png      : one or more lesion masks (one PNG per lesion / per augmentation)
  <iid>@cropped*.png   : cropped lesion patches (optional)

There is essentially no DICOM->PNG conversion to do; this script's only
image-processing job is to derive a bounding box from each mask PNG and
mirror the raw frames into a ``Processed/`` tree.
"""
import os
import shutil
import cv2
import fire


def get_single_bbox(mask_path, normalize=False):
    """Return ``[x1, y1, x2, y2]`` for the largest contour of a mask PNG."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")
    _, binary = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    max_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_cnt)
    x1, y1, x2, y2 = x, y, x + w, y + h
    if normalize:
        h_img, w_img = mask.shape[:2]
        return [x1 / w_img, y1 / h_img, x2 / w_img, y2 / h_img]
    return [x1, y1, x2, y2]


def mirror_images(input_dir='./BUS-Expert', output_dir='./Processed'):
    """Copy ``<iid>@raw.png`` and ``<iid>@mask*.png`` from the raw release into ``Processed/``.

    The on-disk layout produced by this function is::

        Processed/<iid>/<iid>@raw.png
        Processed/<iid>/<iid>@mask*.png

    which is what downstream BreastGPT training expects.
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)
    for iid in os.listdir(input_dir):
        src_dir = os.path.join(input_dir, iid)
        if not os.path.isdir(src_dir):
            continue
        dst_dir = os.path.join(output_dir, iid)
        os.makedirs(dst_dir, exist_ok=True)
        for fname in os.listdir(src_dir):
            if not fname.lower().endswith('.png'):
                continue
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, fname)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)


def compute_bboxes(processed_dir='./Processed', output_json='bboxes.json'):
    """Walk ``Processed/`` and write a JSON of ``{mask_path: [x1, y1, x2, y2]}``."""
    import json
    out = {}
    for iid in os.listdir(processed_dir):
        d = os.path.join(processed_dir, iid)
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if '@mask' in fname and fname.lower().endswith('.png'):
                mpath = os.path.join(d, fname)
                bbox = get_single_bbox(mpath)
                if bbox is not None:
                    out[mpath] = bbox
    with open(output_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(out)} bboxes to {output_json}")


if __name__ == "__main__":
    # Exposes:
    #   python preprocess.py mirror_images --input_dir ./BUS-Expert --output_dir ./Processed
    #   python preprocess.py compute_bboxes --processed_dir ./Processed --output_json bboxes.json
    fire.Fire()
