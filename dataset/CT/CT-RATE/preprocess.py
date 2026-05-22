"""
CT-RATE filtering / copy step.

CT-RATE is a large public chest-CT dataset distributed on HuggingFace. For
BreastGPT we only use the subset that:

    1. Is a non-contrast scan (``scan_type == 'Plain'``); and
    2. Has axial coverage that includes breast tissue
       (``meta_info.inner_path.path`` denotes the per-volume location).

The MONAI-based resize step (longest in-plane edge to 384, depth to 48 slices,
int16) is performed downstream by ``scripts/swift_train_full_node.sh``; here we
only filter the manifest and copy the selected ``.nii.gz`` volumes into a
``Processed/`` tree that mirrors the source layout.
"""
import os
import shutil
import argparse
import pandas as pd
from tqdm import tqdm


def filter_breast_subset(manifest_path: str) -> pd.DataFrame:
    """Return only non-contrast chest CTs that include breast tissue."""
    df = pd.read_csv(manifest_path)

    # Plain (non-contrast) only
    if 'scan_type' in df.columns:
        df = df[df['scan_type'].astype(str).str.lower() == 'plain']

    # Only scans whose volume path mentions a breast-containing field of view.
    # CT-RATE encodes the per-volume relative path under "meta_info.inner_path.path"
    # (NIfTI sub-folders). Match the canonical "breast" / "chest" tag here.
    path_col = None
    for c in df.columns:
        if c.endswith('inner_path.path') or c.endswith('inner_path/path'):
            path_col = c
            break
    if path_col is not None:
        df = df[df[path_col].astype(str).str.contains('breast|chest', case=False, regex=True, na=False)]

    return df.reset_index(drop=True)


def copy_volumes(manifest: pd.DataFrame, source_root: str, output_root: str) -> None:
    """Copy each selected NIfTI into ``output_root`` keeping its relative path."""
    os.makedirs(output_root, exist_ok=True)
    # Identify the column that stores the relative NIfTI path.
    path_col = None
    for c in manifest.columns:
        if c.endswith('inner_path.path') or c.endswith('inner_path/path'):
            path_col = c
            break
    if path_col is None:
        # fall back to a column literally named "VolumeName" (used by CT-RATE)
        path_col = 'VolumeName' if 'VolumeName' in manifest.columns else manifest.columns[0]

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc='copy'):
        rel = str(row[path_col]).strip()
        src = os.path.join(source_root, rel)
        dst = os.path.join(output_root, rel)
        if not os.path.exists(src):
            print(f"missing: {src}")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='train_info.csv',
                        help='CT-RATE manifest CSV (e.g. train_info.csv)')
    parser.add_argument('--source_root', default='.',
                        help='Directory containing the downloaded NIfTI tree')
    parser.add_argument('--output_root', default='./Processed',
                        help='Where filtered volumes are written')
    parser.add_argument('--write_manifest', default='filtered.csv',
                        help='Write the filtered subset to this CSV (set to "" to skip)')
    args = parser.parse_args()

    subset = filter_breast_subset(args.manifest)
    print(f"Selected {len(subset)} volumes from {args.manifest}.")
    if args.write_manifest:
        subset.to_csv(args.write_manifest, index=False)
        print(f"Wrote filtered manifest to {args.write_manifest}.")
    copy_volumes(subset, args.source_root, args.output_root)


if __name__ == '__main__':
    main()
