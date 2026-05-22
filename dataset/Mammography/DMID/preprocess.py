import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import pydicom
import tifffile as tiff
from PIL import Image


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

    # ---- bbox in original image coordinates ----
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


def tif_to_png(dirpath, saved_dirname):
    """Convert raw TIFF / DICOM files inside ``dirpath`` to 8-bit PNGs under ``Original/<saved_dirname>``."""
    output_dir = f'Original/{saved_dirname}'
    os.makedirs(output_dir, exist_ok=True)
    for img in os.listdir(dirpath):
        source = os.path.join(dirpath, img)
        target = os.path.join(output_dir, img.split('.')[0] + '.png')
        if source.endswith('.dcm'):
            image = pydicom.dcmread(source).pixel_array
        else:
            image = np.array(tiff.imread(source))
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        Image.fromarray(image).save(target)
        print(f'{source} -> {target}')


def read_image(source):
    if source is None:
        return source
    if source.endswith('.dcm'):
        image = pydicom.dcmread(source).pixel_array
    elif source.endswith('.tif'):
        image = np.array(tiff.imread(source))
    else:
        image = np.array(Image.open(source))
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if image.ndim != 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def read_mask(mask_path, x_center=None, y_center=None, radius=None, k=1):
    """Read a mask PNG; optionally pick the connected component nearest to a centre point."""
    mask = read_image(mask_path)
    if mask is None:
        return None

    mask = mask.astype(float)
    mask = ((mask - mask.min()) / (mask.max() - mask.min() + 1e-8)).round().astype(np.uint8) * 255
    bin_mask = (mask > 128).astype(np.uint8)

    if k == 1:
        return bin_mask * 255

    num_labels, labels_im = cv2.connectedComponents(bin_mask)
    if x_center is None or y_center is None:
        return bin_mask * 255

    x, y, r = map(int, [x_center, y_center, radius])
    circle_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.circle(circle_mask, (x, y), r, 1, -1)
    if 0 <= y < labels_im.shape[0] and 0 <= x < labels_im.shape[1]:
        target_label = labels_im[y, x]
        if target_label > 0:
            output_mask = (labels_im == target_label).astype(np.uint8) * 255
            if num_labels - 1 == 1:
                output_mask = (circle_mask * output_mask).astype(np.uint8)
            return output_mask

        max_overlap = 0
        best_label = 0
        for label in range(1, num_labels):
            component = (labels_im == label).astype(np.uint8)
            overlap = (component & circle_mask).sum()
            if overlap > max_overlap:
                max_overlap = overlap
                best_label = label
        if best_label > 0:
            bin_mask = (labels_im == best_label).astype(np.uint8)
        output_mask = (circle_mask * bin_mask).astype(np.uint8) * 255
        if output_mask.sum() > 0:
            return output_mask
        return circle_mask.astype(np.uint8) * 255


def process_img(metadata_df, output_base_dir="Processed"):
    os.makedirs(os.path.join(output_base_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, 'finding'), exist_ok=True)

    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
        row = row.where(pd.notna(row), None)
        image_id = row['image_id']
        mask_id = row.get('mask_id', 0)

        image_path = row['image_path']
        mask_path = row['mask_path'] if row.get('mask_path') and os.path.exists(row['mask_path']) else None

        img_output_path = os.path.join(output_base_dir, 'image', image_id + '.png')
        mask_output_path = os.path.join(output_base_dir, 'mask', f'{image_id}_{mask_id}.png')
        cropped_output_path = os.path.join(output_base_dir, 'finding', f'{image_id}_{mask_id}.png')

        image = read_image(image_path)
        mask = read_mask(mask_path, row.get("x_center"), row.get("y_center"), row.get('radius'), row.get('mask_num', 1))
        image, mask = ExtractBreast(image, mask)
        Image.fromarray(image).save(img_output_path)
        if mask is not None:
            Image.fromarray(mask).save(mask_output_path)
            bbox = mask_to_bbox_xyxy(mask)
            if bbox is not None:
                Image.fromarray(image).crop(bbox).save(cropped_output_path)
        print(f"Saved {image_path} -> {img_output_path}")
    print("Processing complete.")


def read_info():
    """
    Expected columns in Metadata.csv:
      image_id, image_path, mask_path, x_center, y_center, radius, mask_id, mask_num
    """
    df = pd.read_csv('Metadata.csv')
    df['image_id'] = df['image_id'].map(str.strip)
    df['image_path'] = df['image_id'].map(lambda x: f'Original/image/{x}.png')
    df['mask_path'] = df['image_id'].map(lambda x: f'Original/mask/{x}.png')
    df['mask_id'] = df.groupby('image_path').cumcount()
    df['mask_num'] = df.groupby('image_path')['image_path'].transform('count')
    return df


if __name__ == '__main__':
    # Step 1 (optional): convert raw TIFF / DICOM dumps to PNGs
    # tif_to_png('DICOM Images', 'image')
    # tif_to_png('ROI Masks', 'mask')

    # Step 2: extract breast region + masks + finding crops
    metadata_df = read_info()
    process_img(metadata_df)
