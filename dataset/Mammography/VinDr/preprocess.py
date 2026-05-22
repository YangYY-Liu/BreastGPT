import os
import csv
import random
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
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
            mask = [m if m is None else m[y_min:y_max, x_min:x_max] for m in mask]
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


def bbox_xyxy_to_mask(image, bbox, value=255):
    if bbox is None:
        return None
    if isinstance(bbox, list) and len(bbox) > 0 and isinstance(bbox[0], list):
        return [bbox_xyxy_to_mask(image, b, value) for b in bbox]

    mask = np.zeros_like(image, dtype=np.uint8)
    if len(bbox) != 4:
        return mask

    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = value
    return mask


def read_image(source):
    if source is None:
        return source
    if source.endswith('.dicom') or source.endswith('.dcm'):
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


def process_single_patient(patient_rows_list, output_base_dir):
    """Process all images for one patient. Saves:
        - Processed/breast/<image_id>.png       : breast-cropped full image
        - Processed/finding/<image_id>_<i>.png  : per-lesion crop (if bbox)
        - Processed/normal/<image_id>_<i>.png   : a random tissue patch with no lesion overlap
    """
    # group by image
    image_groups = {}
    for row in patient_rows_list:
        uid = str(row['image_id'])
        image_groups.setdefault(uid, []).append(row)

    patient_results = []
    for uid, rows in image_groups.items():
        image_path = rows[0]['image_path']
        relative_img_dir = 'breast'
        full_img_dir = os.path.join(output_base_dir, relative_img_dir)
        img_output_path = os.path.join(full_img_dir, f'{uid}.png')
        os.makedirs(full_img_dir, exist_ok=True)

        if not os.path.exists(image_path):
            continue

        image = read_image(image_path)
        if image is None:
            continue

        all_bboxes = [r.get('bbox') for r in rows]
        combined_masks = [bbox_xyxy_to_mask(image, b) for b in all_bboxes]
        processed_image, processed_masks_list = ExtractBreast(image, combined_masks)
        if processed_image is None or processed_image.size == 0:
            continue

        try:
            Image.fromarray(processed_image).save(img_output_path)
        except Exception as e:
            print(f"Error saving image {img_output_path}: {e}")
            continue

        full_find_dir = os.path.join(output_base_dir, 'finding')
        full_normal_dir = os.path.join(output_base_dir, 'normal')
        os.makedirs(full_find_dir, exist_ok=True)
        os.makedirs(full_normal_dir, exist_ok=True)

        if processed_masks_list is None:
            processed_masks_list = [None] * len(rows)
            total_exclusion_mask = None
        else:
            total_exclusion_mask = np.zeros(processed_image.shape, dtype=np.uint8)
            for m in processed_masks_list:
                if m is not None:
                    total_exclusion_mask = cv2.bitwise_or(total_exclusion_mask, m)

        for i, (row, m) in enumerate(zip(rows, processed_masks_list)):
            result_row = {
                'patient_id': row.get('patient_id'),
                'exam_id': row.get('exam_id'),
                'image_id': uid,
                'image_path': img_output_path,
            }
            if m is None or m.sum() == 0:
                patient_results.append(result_row)
                continue
            new_box = mask_to_bbox_xyxy(m)
            if new_box is None:
                continue
            crop_img = processed_image[int(new_box[1]):int(new_box[3]), int(new_box[0]):int(new_box[2])]
            if crop_img.size == 0 or crop_img.max() < 10:
                continue
            cropped_path = os.path.join(full_find_dir, f'{uid}_{i}.png')
            Image.fromarray(crop_img).save(cropped_path)

            # ---- mine a "normal" patch with no lesion overlap ----
            normal_path = os.path.join(full_normal_dir, f'{uid}_{i}.png')
            img_h, img_w = processed_image.shape
            crop_h, crop_w = crop_img.shape
            success = False
            for t in range(100):
                rx = random.randint(0, max(0, img_w - crop_w))
                ry = random.randint(0, max(0, img_h - crop_h))
                roi_mask = total_exclusion_mask[ry:ry + crop_h, rx:rx + crop_w]
                if roi_mask.sum() < 10:
                    normal_crop = processed_image[ry:ry + crop_h, rx:rx + crop_w]
                    if np.mean(normal_crop) > 25 or t > 50:
                        Image.fromarray(normal_crop).save(normal_path)
                        success = True
                        break
            if not success:
                normal_path = None

            result_row['cropped_path'] = cropped_path
            result_row['normal_path'] = normal_path
            result_row['bbox'] = new_box
            patient_results.append(result_row)
    return patient_results


def process_img_mp(metadata_df, output_base_dir="Processed", n_jobs=8):
    output_log = 'processed.csv'
    metadata_df = metadata_df.replace({np.nan: None, pd.NA: None})

    def safe_eval(x):
        if x is None or isinstance(x, list):
            return x
        try:
            return literal_eval(x)
        except Exception:
            return None

    metadata_df['bbox'] = metadata_df['bbox'].map(safe_eval)

    completed_keys = set()
    file_mode = 'w'
    write_header = True
    if os.path.exists(output_log):
        try:
            existing = pd.read_csv(output_log)
            if not existing.empty:
                for _, row in existing.iterrows():
                    completed_keys.add((str(row['patient_id']), str(row['exam_id'])))
                file_mode = 'a'
                write_header = False
        except Exception as e:
            print(f"Could not read {output_log}, rewriting: {e}")

    grouped = metadata_df.groupby(by=['patient_id', 'exam_id'])
    tasks_to_process = []
    for (pid, eid), group in grouped:
        if (str(pid), str(eid)) not in completed_keys:
            tasks_to_process.append((group.to_dict('records'), output_base_dir))

    if not tasks_to_process:
        print("All tasks done.")
        return

    with open(output_log, file_mode, newline='', encoding='utf-8') as f:
        writer = None
        with mpi.WorkerPool(n_jobs=n_jobs) as pool:
            for patient_results in pool.imap(process_single_patient, tasks_to_process, progress_bar=True):
                if not patient_results:
                    continue
                if writer is None:
                    writer = csv.DictWriter(
                        f, fieldnames=list(patient_results[0].keys()), extrasaction='ignore'
                    )
                    if write_header:
                        writer.writeheader()
                for res in patient_results:
                    try:
                        writer.writerow(res)
                    except ValueError:
                        pass


def read_info():
    """Build a minimal image manifest from VinDr-Mammo's DICOM tree + finding annotations.

    Expected files (from PhysioNet release):
      metadata.csv, breast-level_annotations.csv, finding_annotations.csv
      images/<study_id>/<image_id>.dicom
    """
    finding_csv = pd.read_csv('finding_annotations.csv').rename(columns={
        'study_id': 'patient_id', 'series_id': 'exam_id'
    })

    def gen_bbox(row):
        bbox = row[['xmin', 'ymin', 'xmax', 'ymax']].to_list()
        if bbox[0] is None or pd.isna(bbox[0]):
            return None
        return list(map(int, bbox))

    finding_csv['bbox'] = finding_csv.apply(gen_bbox, axis=1)
    finding_csv = finding_csv.drop(columns=['xmin', 'ymin', 'xmax', 'ymax'], errors='ignore')
    finding_csv['image_path'] = (
        'images/' + finding_csv['patient_id'].astype(str) + '/' + finding_csv['image_id'].astype(str) + '.dicom'
    )
    return finding_csv[['patient_id', 'exam_id', 'image_id', 'image_path', 'bbox']]


if __name__ == '__main__':
    metadata_df = read_info()
    process_img_mp(metadata_df)
