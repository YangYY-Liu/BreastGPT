"""
Example usage:
```
python run_single_slide.py --slide_path output/wsis/394140.svs --job_dir output/ --mag 20 --patch_size 256
```

"""
import argparse
import os

from trident import load_wsi
from trident.segmentation_models import segmentation_model_factory
from trident.patch_encoder_models import encoder_factory
from trident.patch_encoder_models import encoder_registry as patch_encoder_registry


def process_slide(
    slide_path: str,
    job_dir: str,
    gpu: int = 0,
    patch_encoder: str = 'conch_v15',
    mag: int = 20,
    patch_size: int = 256,
    segmenter: str = 'hest',
    seg_conf_thresh: float = 0.5,
    remove_holes: bool = False,
    remove_artifacts: bool = False,
    remove_penmarks: bool = False,
    custom_mpp_keys: list = None,
    overlap: int = 0,
    batch_size: int = 32,
):
    """
    Process a single WSI by performing segmentation, patch extraction, and feature extraction sequentially.
    
    Args:
        slide_path: Path to the WSI file to process
        job_dir: Directory to store outputs
        gpu: GPU index to use for processing tasks
        patch_encoder: Patch encoder to use
        mag: Magnification at which patches/features are extracted (5, 10, 20, or 40)
        patch_size: Patch size at which coords/features are extracted
        segmenter: Type of tissue vs background segmenter ('hest' or 'grandqc')
        seg_conf_thresh: Confidence threshold for segmentation binarization
        remove_holes: Whether to remove holes
        remove_artifacts: Whether to run additional model to remove artifacts
        remove_penmarks: Whether to run additional model to remove penmarks
        custom_mpp_keys: Custom keys for MPP resolution storage
        overlap: Absolute overlap for patching in pixels
        batch_size: Batch size for feature extraction
    """
    # Initialize the WSI
    print(f"Processing slide: {slide_path}")
    slide = load_wsi(slide_path=slide_path, lazy_init=False, custom_mpp_keys=custom_mpp_keys)

    # Step 1: Tissue Segmentation
    print("Running tissue segmentation...")
    segmentation_model = segmentation_model_factory(
        model_name=segmenter,
        confidence_thresh=seg_conf_thresh,
    )
    if remove_artifacts or remove_penmarks:
        artifact_remover_model = segmentation_model_factory(
            'grandqc_artifact',
            remove_penmarks_only=remove_penmarks and not remove_artifacts
        )
    else:
        artifact_remover_model = None

    slide.segment_tissue(
        segmentation_model=segmentation_model,
        target_mag=segmentation_model.target_mag,
        job_dir=job_dir,
        device=f"cuda:{gpu}",
        holes_are_tissue=not remove_holes
    )
    if artifact_remover_model is not None:
        slide.segment_tissue(
            segmentation_model=artifact_remover_model,
            target_mag=artifact_remover_model.target_mag,
            holes_are_tissue=False,
            job_dir=job_dir
        )
    print(f"Tissue segmentation completed. Results saved to {os.path.join(job_dir, 'contours_geojson')} and {os.path.join(job_dir, 'contours')}")

    # Step 2: Tissue Coordinate Extraction (Patching)
    print("Extracting tissue coordinates...")
    save_coords = os.path.join(job_dir, f'{mag}x_{patch_size}px_{overlap}px_overlap')

    coords_path = slide.extract_tissue_coords(
        target_mag=mag,
        patch_size=patch_size,
        save_coords=save_coords
    )
    print(f"Tissue coordinates extracted and saved to {coords_path}.")

    # Step 3: Visualize patching
    viz_coords_path = slide.visualize_coords(
        coords_path=coords_path,
        save_patch_viz=os.path.join(save_coords, 'visualization'),
    )
    print(f"Tissue coordinates extracted and saved to {viz_coords_path}.")

    # Step 4: Feature Extraction
    print("Extracting features from patches...")
    encoder = encoder_factory(patch_encoder)
    encoder.eval()
    encoder.to(f"cuda:{gpu}")
    features_path = features_dir = os.path.join(save_coords, "features_{}".format(patch_encoder))
    slide.extract_patch_features(
        patch_encoder=encoder,
        coords_path=os.path.join(save_coords, 'patches', f'{slide.name}_patches.h5'),
        save_features=features_dir,
        device=f"cuda:{gpu}",
        batch_limit=batch_size
    )
    print(f"Feature extraction completed. Results saved to {features_path}")


def parse_arguments():
    """Parse command-line arguments for processing a single WSI."""
    parser = argparse.ArgumentParser(description="Process a WSI from A to Z.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--slide_path", type=str, required=True)
    parser.add_argument("--job_dir", type=str, required=True)
    parser.add_argument('--patch_encoder', type=str, default='conch_v15', 
                        choices=patch_encoder_registry.keys())
    parser.add_argument("--mag", type=int, choices=[5, 10, 20, 40], default=20)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument('--segmenter', type=str, default='hest', choices=['hest', 'grandqc'])
    parser.add_argument('--seg_conf_thresh', type=float, default=0.5)
    parser.add_argument('--remove_holes', action='store_true', default=False)
    parser.add_argument('--remove_artifacts', action='store_true', default=False)
    parser.add_argument('--remove_penmarks', action='store_true', default=False)
    parser.add_argument('--custom_mpp_keys', type=str, nargs='+', default=None)
    parser.add_argument('--overlap', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_arguments()
    process_slide(
        slide_path=args.slide_path,
        job_dir=args.job_dir,
        gpu=args.gpu,
        patch_encoder=args.patch_encoder,
        mag=args.mag,
        patch_size=args.patch_size,
        segmenter=args.segmenter,
        seg_conf_thresh=args.seg_conf_thresh,
        remove_holes=args.remove_holes,
        remove_artifacts=args.remove_artifacts,
        remove_penmarks=args.remove_penmarks,
        custom_mpp_keys=args.custom_mpp_keys,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()