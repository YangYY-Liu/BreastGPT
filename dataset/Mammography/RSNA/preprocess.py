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

    # ---- row crop (on column-cropped image) ----
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
        mask = mask[y_min:y_max, x_min:x_max]
    return cropped, mask


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
        image = image[10:-10, 10:-10]
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
    exam_id = row_dict['exam_id']
    image_id = row_dict['image_id']
    image_path = row_dict['image_path']

    img_output_path = os.path.join(output_base_dir, f'{exam_id}_{image_id}.png')
    os.makedirs(output_base_dir, exist_ok=True)
    if not os.path.exists(image_path):
        return None
    image = read_image(image_path)
    if image is None:
        return None
    image, _ = ExtractBreast(image)
    Image.fromarray(image).save(img_output_path)
    return {
        'patient_id': row_dict.get('patient_id'),
        'exam_id': exam_id,
        'image_id': image_id,
        'image_path': img_output_path,
    }


def process_img_mp(metadata_df, output_base_dir="Processed", n_jobs=16):
    """Multiprocess the breast crop pipeline. Skips files already written."""
    output_log = 'processed.csv'
    metadata_df = metadata_df.replace({np.nan: None, pd.NA: None})

    completed_keys = set()
    file_mode = 'w'
    write_header = True
    if os.path.exists(output_log):
        try:
            existing = pd.read_csv(output_log)
            if not existing.empty:
                for _, row in existing.iterrows():
                    completed_keys.add(
                        (str(row['patient_id']), str(row['exam_id']), str(row['image_id']))
                    )
                file_mode = 'a'
                write_header = False
        except Exception as e:
            print(f"Could not read {output_log}, rewriting: {e}")

    all_tasks = metadata_df.to_dict('records')
    tasks_to_process = []
    for task in all_tasks:
        key = (str(task['patient_id']), str(task['exam_id']), str(task['image_id']))
        if key not in completed_keys:
            tasks_to_process.append((task, output_base_dir))

    if not tasks_to_process:
        print("All tasks done.")
        return

    print(f"Total: {len(all_tasks)} | Done: {len(completed_keys)} | Remaining: {len(tasks_to_process)}")

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
    """Build a minimal image manifest from train.csv (patient_id, exam_id, image_id, image_path)."""
    df = pd.read_csv('train.csv')
    df['image_path'] = 'train_images/' + df['patient_id'].map(str) + '/' + df['image_id'].map(str) + '.dcm'
    df['exam_id'] = df['patient_id'].map(str)
    df['patient_id'] = 'RSNA_' + df['patient_id'].map(str)
    return df[['patient_id', 'exam_id', 'image_id', 'image_path']]


if __name__ == '__main__':
    metadata_df = read_info()
    process_img_mp(metadata_df)
