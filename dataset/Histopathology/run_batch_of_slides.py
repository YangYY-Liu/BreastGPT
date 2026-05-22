"""
Example usage:
```
python run_batch_of_slides.py --task all --wsi_dir output/wsis --job_dir output --patch_encoder uni_v1 --mag 20 --patch_size 256
```

"""
import os
import argparse
import torch
from typing import Any, List, Optional, Literal

from trident import Processor 
from trident.patch_encoder_models import encoder_registry as patch_encoder_registry
from trident.slide_encoder_models import encoder_registry as slide_encoder_registry


def build_parser() -> argparse.ArgumentParser:
    """
    Parse command-line arguments for the Trident processing script.
    """
    parser = argparse.ArgumentParser(description='Run Trident')

    # Generic arguments 
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--task', type=str, default='seg', choices=['seg', 'coords', 'feat', 'all'])
    parser.add_argument('--job_dir', type=str, required=True)
    parser.add_argument('--skip_errors', action='store_true', default=False)
    parser.add_argument('--max_workers', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=64)

    # Caching
    parser.add_argument('--wsi_cache', type=str, default=None)
    parser.add_argument('--cache_batch_size', type=int, default=32)

    # Slide-related
    parser.add_argument('--wsi_dir', type=str, required=True)
    parser.add_argument('--wsi_ext', type=str, nargs='+', default=None)
    parser.add_argument('--custom_mpp_keys', type=str, nargs='+', default=None)
    parser.add_argument('--custom_list_of_wsis', type=str, default=None)
    parser.add_argument('--reader_type', type=str, choices=['openslide', 'image', 'cucim', 'sdpc'], default=None)
    parser.add_argument("--search_nested", action="store_true")

    # Segmentation
    parser.add_argument('--segmenter', type=str, default='hest', choices=['hest', 'grandqc'])
    parser.add_argument('--seg_conf_thresh', type=float, default=0.5)
    parser.add_argument('--remove_holes', action='store_true', default=False)
    parser.add_argument('--remove_artifacts', action='store_true', default=False)
    parser.add_argument('--remove_penmarks', action='store_true', default=False)
    parser.add_argument('--seg_batch_size', type=int, default=None)

    # Patching
    parser.add_argument('--mag', type=int, choices=[5, 10, 20, 40, 80], default=20)
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--overlap', type=int, default=0)
    parser.add_argument('--min_tissue_proportion', type=float, default=0.)
    parser.add_argument('--coords_dir', type=str, default=None)

    # Feature extraction
    parser.add_argument('--patch_encoder', type=str, default='conch_v15', choices=patch_encoder_registry.keys())
    parser.add_argument('--patch_encoder_ckpt_path', type=str, default=None)
    parser.add_argument('--slide_encoder', type=str, default=None, choices=slide_encoder_registry.keys())
    parser.add_argument('--feat_batch_size', type=int, default=None)
    return parser


def parse_arguments() -> argparse.Namespace:
    return build_parser().parse_args()


def generate_help_text() -> str:
    return build_parser().format_help()


def initialize_processor(
    job_dir: str,
    wsi_dir: str,
    wsi_ext: Optional[List[str]] = None,
    wsi_cache: Optional[str] = None,
    skip_errors: bool = False,
    custom_mpp_keys: Optional[List[str]] = None,
    custom_list_of_wsis: Optional[str] = None,
    max_workers: Optional[int] = None,
    reader_type: Optional[str] = None,
    search_nested: bool = False,
) -> Processor:
    """
    Initialize the Trident Processor.

    Args:
        job_dir: Directory to store outputs
        wsi_dir: Directory containing WSI files
        wsi_ext: List of allowed file extensions for WSI files
        wsi_cache: Path to local cache for faster WSI access
        skip_errors: Skip errored slides and continue processing
        custom_mpp_keys: Custom keys for MPP resolution storage
        custom_list_of_wsis: Custom list of WSIs in a csv file
        max_workers: Maximum number of workers (0 for main process)
        reader_type: Force specific WSI reader ('openslide', 'image', 'cucim', 'sdpc')
        search_nested: Recursively search for WSIs in subdirectories
    """
    return Processor(
        job_dir=job_dir,
        wsi_source=wsi_dir,
        wsi_ext=wsi_ext,
        wsi_cache=wsi_cache,
        skip_errors=skip_errors,
        custom_mpp_keys=custom_mpp_keys,
        custom_list_of_wsis=custom_list_of_wsis,
        max_workers=max_workers,
        reader_type=reader_type,
        search_nested=search_nested,
    )


def run_task(
    processor: Processor,
    task: Literal['seg', 'coords', 'feat'],
    gpu: int = 0,
    # Segmentation params
    segmenter: str = 'hest',
    seg_conf_thresh: float = 0.5,
    remove_holes: bool = False,
    remove_artifacts: bool = False,
    remove_penmarks: bool = False,
    seg_batch_size: Optional[int] = None,
    batch_size: int = 64,
    # Patching params
    mag: int = 20,
    patch_size: int = 512,
    overlap: int = 0,
    min_tissue_proportion: float = 0.,
    coords_dir: Optional[str] = None,
    # Feature extraction params
    patch_encoder: str = 'conch_v15',
    patch_encoder_ckpt_path: Optional[str] = None,
    slide_encoder: Optional[str] = None,
    feat_batch_size: Optional[int] = None,
) -> None:
    """
    Execute the specified task using the Trident Processor.

    Args:
        processor: Initialized Trident Processor instance
        task: Task to run ('seg', 'coords', 'feat')
        gpu: GPU index
        segmenter: Tissue vs background segmenter ('hest' or 'grandqc')
        seg_conf_thresh: Confidence threshold for segmentation
        remove_holes: Remove holes in tissue
        remove_artifacts: Run model to remove artifacts
        remove_penmarks: Run model to remove penmarks
        seg_batch_size: Batch size for segmentation
        batch_size: Default batch size
        mag: Magnification for extraction (5, 10, 20, 40, 80)
        patch_size: Patch size for extraction
        overlap: Absolute overlap in pixels
        min_tissue_proportion: Minimum tissue proportion (0-1)
        coords_dir: Directory to save/restore tissue coordinates
        patch_encoder: Patch encoder to use
        patch_encoder_ckpt_path: Local path to patch encoder checkpoint
        slide_encoder: Slide encoder to use
        feat_batch_size: Batch size for feature extraction
    """
    device = f'cuda:{gpu}'

    if task == 'seg':
        from trident.segmentation_models.load import segmentation_model_factory

        segmentation_model = segmentation_model_factory(
            segmenter,
            confidence_thresh=seg_conf_thresh,
        )
        if remove_artifacts or remove_penmarks:
            artifact_remover_model = segmentation_model_factory(
                'grandqc_artifact',
                remove_penmarks_only=remove_penmarks and not remove_artifacts
            )
        else:
            artifact_remover_model = None

        processor.run_segmentation_job(
            segmentation_model,
            seg_mag=segmentation_model.target_mag,
            holes_are_tissue=not remove_holes,
            artifact_remover_model=artifact_remover_model,
            batch_size=seg_batch_size if seg_batch_size is not None else batch_size,
            device=device,
        )
    elif task == 'coords':
        processor.run_patching_job(
            target_magnification=mag,
            patch_size=patch_size,
            overlap=overlap,
            saveto=coords_dir,
            min_tissue_proportion=min_tissue_proportion
        )
    elif task == 'feat':
        effective_coords_dir = coords_dir or f'{mag}x_{patch_size}px_{overlap}px_overlap'
        effective_batch_size = feat_batch_size if feat_batch_size is not None else batch_size

        if slide_encoder is None:
            from trident.patch_encoder_models.load import encoder_factory
            encoder = encoder_factory(patch_encoder, weights_path=patch_encoder_ckpt_path)
            processor.run_patch_feature_extraction_job(
                coords_dir=effective_coords_dir,
                patch_encoder=encoder,
                device=device,
                saveas='h5',
                batch_limit=effective_batch_size,
            )
        else:
            from trident.slide_encoder_models.load import encoder_factory
            encoder = encoder_factory(slide_encoder)
            processor.run_slide_feature_extraction_job(
                slide_encoder=encoder,
                coords_dir=effective_coords_dir,
                device=device,
                saveas='h5',
                batch_limit=effective_batch_size,
            )
    else:
        raise ValueError(f'Invalid task: {task}')


def run_batch(
    job_dir: str,
    wsi_dir: str,
    task: Literal['seg', 'coords', 'feat', 'all'] = 'seg',
    gpu: int = 0,
    skip_errors: bool = False,
    max_workers: Optional[int] = None,
    batch_size: int = 64,
    # Caching
    wsi_cache: Optional[str] = None,
    cache_batch_size: int = 32,
    # Slide-related
    wsi_ext: Optional[List[str]] = None,
    custom_mpp_keys: Optional[List[str]] = None,
    custom_list_of_wsis: Optional[str] = None,
    reader_type: Optional[str] = None,
    search_nested: bool = False,
    # Segmentation
    segmenter: str = 'hest',
    seg_conf_thresh: float = 0.5,
    remove_holes: bool = False,
    remove_artifacts: bool = False,
    remove_penmarks: bool = False,
    seg_batch_size: Optional[int] = None,
    # Patching
    mag: int = 20,
    patch_size: int = 512,
    overlap: int = 0,
    min_tissue_proportion: float = 0.,
    coords_dir: Optional[str] = None,
    # Feature extraction
    patch_encoder: str = 'conch_v15',
    patch_encoder_ckpt_path: Optional[str] = None,
    slide_encoder: Optional[str] = None,
    feat_batch_size: Optional[int] = None,
) -> None:
    """
    Main entry point for batch processing slides.

    Args:
        job_dir: Directory to store outputs
        wsi_dir: Directory containing WSI files
        task: Task to run ('seg', 'coords', 'feat', 'all')
        gpu: GPU index
        skip_errors: Skip errored slides
        max_workers: Maximum number of workers
        batch_size: Default batch size
        wsi_cache: Path to local cache for faster WSI access
        cache_batch_size: Maximum slides to cache at once
        wsi_ext: Allowed file extensions
        custom_mpp_keys: Custom MPP keys
        custom_list_of_wsis: Path to custom WSI list csv
        reader_type: Force specific WSI reader
        search_nested: Search subdirectories
        segmenter: Segmenter type ('hest' or 'grandqc')
        seg_conf_thresh: Segmentation confidence threshold
        remove_holes: Remove holes
        remove_artifacts: Remove artifacts
        remove_penmarks: Remove penmarks
        seg_batch_size: Segmentation batch size
        mag: Magnification (5, 10, 20, 40, 80)
        patch_size: Patch size
        overlap: Overlap in pixels
        min_tissue_proportion: Minimum tissue proportion
        coords_dir: Coordinates directory
        patch_encoder: Patch encoder name
        patch_encoder_ckpt_path: Patch encoder checkpoint path
        slide_encoder: Slide encoder name
        feat_batch_size: Feature extraction batch size
    """
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'

    # Common kwargs for run_task
    task_kwargs = dict(
        gpu=gpu,
        segmenter=segmenter,
        seg_conf_thresh=seg_conf_thresh,
        remove_holes=remove_holes,
        remove_artifacts=remove_artifacts,
        remove_penmarks=remove_penmarks,
        seg_batch_size=seg_batch_size,
        batch_size=batch_size,
        mag=mag,
        patch_size=patch_size,
        overlap=overlap,
        min_tissue_proportion=min_tissue_proportion,
        coords_dir=coords_dir,
        patch_encoder=patch_encoder,
        patch_encoder_ckpt_path=patch_encoder_ckpt_path,
        slide_encoder=slide_encoder,
        feat_batch_size=feat_batch_size,
    )

    if wsi_cache:
        # === Parallel pipeline with caching ===
        from queue import Queue
        from threading import Thread

        from trident.Concurrency import batch_producer, batch_consumer, cache_batch
        from trident.IO import collect_valid_slides

        queue = Queue(maxsize=1)
        valid_slides = collect_valid_slides(
            wsi_dir=wsi_dir,
            custom_list_path=custom_list_of_wsis,
            wsi_ext=wsi_ext,
            search_nested=search_nested,
            max_workers=max_workers
        )
        print(f"[MAIN] Found {len(valid_slides)} valid slides in {wsi_dir}.")

        warm = valid_slides[:cache_batch_size]
        warmup_dir = os.path.join(wsi_cache, "batch_0")
        print(f"[MAIN] Warmup caching batch: {warmup_dir}")
        cache_batch(warm, warmup_dir)
        queue.put(0)

        def processor_factory(local_wsi_dir: str) -> Processor:
            return initialize_processor(
                job_dir=job_dir,
                wsi_dir=local_wsi_dir,
                wsi_ext=wsi_ext,
                wsi_cache=None,
                skip_errors=skip_errors,
                custom_mpp_keys=custom_mpp_keys,
                custom_list_of_wsis=None,
                max_workers=max_workers,
                reader_type=reader_type,
                search_nested=False,
            )

        def run_task_fn(processor: Processor, task_name: str) -> None:
            run_task(processor, task=task_name, **task_kwargs)

        producer = Thread(target=batch_producer, args=(
            queue, valid_slides, cache_batch_size, cache_batch_size, wsi_cache
        ))
        consumer = Thread(target=batch_consumer, args=(
            queue, task, wsi_cache, processor_factory, run_task_fn
        ))

        print("[MAIN] Starting producer and consumer threads.")
        producer.start()
        consumer.start()
        producer.join()
        consumer.join()
    else:
        # === Sequential mode ===
        processor = initialize_processor(
            job_dir=job_dir,
            wsi_dir=wsi_dir,
            wsi_ext=wsi_ext,
            wsi_cache=wsi_cache,
            skip_errors=skip_errors,
            custom_mpp_keys=custom_mpp_keys,
            custom_list_of_wsis=custom_list_of_wsis,
            max_workers=max_workers,
            reader_type=reader_type,
            search_nested=search_nested,
        )
        tasks = ['seg', 'coords', 'feat'] if task == 'all' else [task]
        for task_name in tasks:
            run_task(processor, task=task_name, **task_kwargs)


def main() -> None:
    """CLI entry point."""
    args = parse_arguments()
    run_batch(
        job_dir=args.job_dir,
        wsi_dir=args.wsi_dir,
        task=args.task,
        gpu=args.gpu,
        skip_errors=args.skip_errors,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        wsi_cache=args.wsi_cache,
        cache_batch_size=args.cache_batch_size,
        wsi_ext=args.wsi_ext,
        custom_mpp_keys=args.custom_mpp_keys,
        custom_list_of_wsis=args.custom_list_of_wsis,
        reader_type=args.reader_type,
        search_nested=args.search_nested,
        segmenter=args.segmenter,
        seg_conf_thresh=args.seg_conf_thresh,
        remove_holes=args.remove_holes,
        remove_artifacts=args.remove_artifacts,
        remove_penmarks=args.remove_penmarks,
        seg_batch_size=args.seg_batch_size,
        mag=args.mag,
        patch_size=args.patch_size,
        overlap=args.overlap,
        min_tissue_proportion=args.min_tissue_proportion,
        coords_dir=args.coords_dir,
        patch_encoder=args.patch_encoder,
        patch_encoder_ckpt_path=args.patch_encoder_ckpt_path,
        slide_encoder=args.slide_encoder,
        feat_batch_size=args.feat_batch_size,
    )


if __name__ == "__main__":
    main()