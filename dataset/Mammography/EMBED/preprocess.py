import os
import csv
import pandas as pd
import numpy as np
import cv2
import pydicom
import tifffile as tiff
import mpire as mpi
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut
from ast import literal_eval


def np_CountUpContinuingOnes(b_arr):
    left = np.arange(len(b_arr))
    left[b_arr > 0] = 0
    left = np.maximum.accumulate(left)

    rev_arr = b_arr[::-1]
    right = np.arange(len(rev_arr))
    right[rev_arr > 0] = 0
    right = np.maximum.accumulate(right)
    right = len(rev_arr) - 1 - right[::-1]

    return right - left - 1


def ExtractBreast(img, mask=None):
    img_copy = img.copy()
    img = np.where(img <= 10, 0, img)
    height, _ = img.shape

    # ---- column crop ----
    y_a = height // 2 + int(height * 0.4)
    y_b = height // 2 - int(height * 0.4)
    b_arr = img[y_b:y_a].std(axis=0) != 0
    continuing_ones = np_CountUpContinuingOnes(b_arr)
    col_ind = np.where(continuing_ones == continuing_ones.max())[0]

    # ---- row crop ----
    img_col = img[:, col_ind]
    _, width = img_col.shape
    x_a = width // 2 + int(width * 0.4)
    x_b = width // 2 - int(width * 0.4)
    b_arr = img_col[:, x_b:x_a].std(axis=1) != 0
    continuing_ones = np_CountUpContinuingOnes(b_arr)
    row_ind = np.where(continuing_ones == continuing_ones.max())[0]

    x_min = int(col_ind.min())
    x_max = int(col_ind.max()) + 1
    y_min = int(row_ind.min())
    y_max = int(row_ind.max()) + 1

    cropped = img_copy[y_min:y_max, x_min:x_max]
    if mask is not None:
        if isinstance(mask, list):
            mask = [m[y_min:y_max, x_min:x_max] for m in mask]
        else:
            mask = mask[y_min:y_max, x_min:x_max]
    return cropped, mask


def mask_to_bbox_xyxy(mask):
    if mask is None:
        return None
    ys, xs = np.where(mask != 0)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    return [int(x1), int(y1), int(x2), int(y2)]


def bbox_yxyx_to_mask(image, bbox, value=255):
    """EMBED ROI_coords are stored as (ymin, xmin, ymax, xmax). Convert to a binary mask."""
    if bbox is None:
        return None
    if isinstance(bbox, list) and len(bbox) > 0 and isinstance(bbox[0], list):
        return [bbox_yxyx_to_mask(image, b, value) for b in bbox]
    mask = np.zeros_like(image, dtype=np.uint8)
    if len(bbox) != 4:
        return None
    y_min, x_min, y_max, x_max = map(int, bbox)
    h, w = image.shape[:2]

    x_min = max(0, min(w, x_min))
    x_max = max(0, min(w, x_max))
    y_min = max(0, min(h, y_min))
    y_max = max(0, min(h, y_max))
    if y_max > y_min and x_max > x_min:
        mask[y_min:y_max, x_min:x_max] = value
    return mask


def read_image(source):
    if source is None:
        return source
    if source.endswith('.dcm'):
        dicom = pydicom.dcmread(source)
        image = dicom.pixel_array
        image = apply_voi_lut(image, dicom, prefer_lut=False)
        image = (image - image.min()) / (image.max() - image.min())
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            image = 1 - image
        image = (image * 255).astype(np.uint8)
    elif source.endswith('.tif'):
        image = np.array(tiff.imread(source))
    else:
        image = np.array(Image.open(source))
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if image.ndim != 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def process_single_row(row_dict, output_base_dir):
    patient_id = row_dict['patient_id']
    exam_id = row_dict['exam_id']
    unique_id = row_dict['unique_id']
    image_path = row_dict['image_path']
    bbox = row_dict.get('bbox')
    result_row = {
        'patient_id': patient_id,
        'exam_id': exam_id,
        'unique_id': unique_id,
        'image_path': image_path,
    }

    img_output_path = os.path.join(output_base_dir, 'image', f'{patient_id}_{exam_id}', f'{unique_id}.png')
    os.makedirs(os.path.join(output_base_dir, 'image', f'{patient_id}_{exam_id}'), exist_ok=True)

    image = read_image(image_path)
    if image is None:
        return None
    mask = bbox_yxyx_to_mask(image, bbox)
    image, mask = ExtractBreast(image, mask)
    Image.fromarray(image).save(img_output_path)

    cropped_paths = []
    new_bboxes = []
    if mask is not None:
        mi = 0
        for m in mask:
            if m is None or m.sum() == 0:
                continue
            new_box = mask_to_bbox_xyxy(m)
            crop_img = image[int(new_box[1]):int(new_box[3]), int(new_box[0]):int(new_box[2])]
            if crop_img.max() < 10:
                continue
            cropped_output_path = os.path.join(
                output_base_dir, 'finding', f'{patient_id}_{exam_id}', f'{unique_id}_{mi}.png'
            )
            os.makedirs(os.path.dirname(cropped_output_path), exist_ok=True)
            Image.fromarray(crop_img).save(cropped_output_path)
            cropped_paths.append(cropped_output_path)
            new_bboxes.append(new_box)
            mi += 1
    result_row['image_path'] = img_output_path
    result_row['cropped_path'] = cropped_paths if cropped_paths else None
    result_row['bbox'] = new_bboxes if new_bboxes else None
    return result_row


def process_img_mp(metadata_df, output_base_dir="Processed", n_jobs=16):
    output_log = 'processed.csv'
    if 'unique_id' not in metadata_df.columns:
        metadata_df['unique_id'] = metadata_df.groupby(['patient_id', 'exam_id']).cumcount()

    metadata_df = metadata_df.replace({np.nan: None})
    metadata_df['bbox'] = metadata_df['bbox'].map(
        lambda x: None if x is None else literal_eval(x) if isinstance(x, str) else x
    )

    completed_keys = set()
    file_mode = 'w'
    write_header = True
    if os.path.exists(output_log):
        try:
            existing = pd.read_csv(output_log)
            if not existing.empty:
                for _, row in existing.iterrows():
                    completed_keys.add(
                        (str(row['patient_id']), str(row['exam_id']), str(row['unique_id']))
                    )
                file_mode = 'a'
                write_header = False
        except Exception as e:
            print(f"Could not read {output_log}, rewriting: {e}")

    tasks_to_process = []
    for task in metadata_df.to_dict('records'):
        key = (str(task['patient_id']), str(task['exam_id']), str(task['unique_id']))
        if key not in completed_keys:
            tasks_to_process.append((task, output_base_dir))

    if not tasks_to_process:
        print("All tasks done.")
        return

    with open(output_log, file_mode, newline='', encoding='utf-8') as f:
        writer = None
        with mpi.WorkerPool(n_jobs=n_jobs) as pool:
            for result in pool.imap(process_single_row, tasks_to_process, progress_bar=True):
                if result is None:
                    continue
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(result.keys()))
                    if write_header:
                        writer.writeheader()
                writer.writerow(result)


def read_info():
    """Build a minimal image manifest from EMBED's open-data tables.

    Expected files (downloaded from the EMBED registry):
        tables/EMBED_OpenData_metadata.csv
    """
    meta_rename = {
        'empi_anon': 'patient_id',
        'acc_anon': 'exam_id',
        'anon_dicom_path': 'image_path',
        'ROI_coords': 'bbox',
    }
    metadata_csv = pd.read_csv('tables/EMBED_OpenData_metadata.csv').rename(columns=meta_rename)
    cols = [c for c in ['patient_id', 'exam_id', 'image_path', 'bbox'] if c in metadata_csv.columns]
    df = metadata_csv[cols].copy()
    df = df.replace({np.nan: None})
    if 'bbox' in df.columns:
        df['bbox'] = df['bbox'].map(
            lambda x: None if x is None or (isinstance(x, str) and len(literal_eval(x)) == 0) else x
        )
    df['unique_id'] = df.groupby(['patient_id', 'exam_id']).cumcount()
    return df


if __name__ == '__main__':
    metadata_df = read_info()
    process_img_mp(metadata_df)
