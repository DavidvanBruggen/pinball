import argparse
import io
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)


def _save_image_record(image_value, out_path: Path) -> None:
    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(image_value, "save"):
        image = image_value.convert("RGB")
        image.save(out_path, format="JPEG", quality=95)
        return
    if isinstance(image_value, dict):
        raw_bytes = image_value.get("bytes", None)
        if raw_bytes is not None:
            image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            image.save(out_path, format="JPEG", quality=95)
            return
        raw_path = image_value.get("path", None)
        if raw_path:
            image = Image.open(raw_path).convert("RGB")
            image.save(out_path, format="JPEG", quality=95)
            return
    raise ValueError(f"Unsupported image record type: {type(image_value)}")


def _split_exists(dataset_id: str, split_name: str) -> bool:
    """Check if a split name exists in the dataset."""
    from datasets import load_dataset
    try:
        load_dataset(dataset_id, split=split_name, streaming=True)
        return True
    except ValueError:
        return False


def _resolve_split_name(dataset_id: str, requested: str, split_category: str) -> str:
    """Resolve the actual split name, trying common variations.

    Args:
        dataset_id: HuggingFace dataset identifier
        requested: The split name requested by caller
        split_category: One of 'train', 'validation'

    Returns:
        The actual split name that exists in the dataset

    Raises:
        ValueError: If no valid split is found, with available splits listed
    """
    if _split_exists(dataset_id, requested):
        return requested

    variations = {
        'train': ['train', 'training'],
        'validation': ['validation', 'valid', 'val'],
    }

    candidates = variations.get(split_category, [requested])
    for candidate in candidates:
        if candidate != requested and _split_exists(dataset_id, candidate):
            logger.warning("Mapped %s split '%s' -> '%s'", split_category, requested, candidate)
            return candidate

    available = []
    all_candidates = ['train', 'validation', 'valid', 'val', 'test', 'testing', 'eval']
    for split in all_candidates:
        if _split_exists(dataset_id, split):
            available.append(split)

    raise ValueError(
        f"Could not resolve {split_category} split name. "
        f"Requested: '{requested}'. Available splits in '{dataset_id}': {available}"
    )


def _write_split(
    dataset_id: str,
    split_name: str,
    split_dir: Path,
    image_field: str,
    label_field: str,
    streaming: bool,
    max_samples: int,
) -> int:
    from datasets import load_dataset

    split_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(dataset_id, split=split_name, streaming=bool(streaming))

    written = 0
    progress_total = None if int(max_samples) <= 0 else int(max_samples)
    pbar = tqdm(desc=f"Preparing {split_name}", total=progress_total, dynamic_ncols=True)
    for idx, sample in enumerate(ds):
        if int(max_samples) > 0 and idx >= int(max_samples):
            break
        image_value = sample.get(image_field, None)
        label_value = sample.get(label_field, None)
        if image_value is None or label_value is None:
            continue
        class_dir = split_dir / f"{int(label_value):04d}"
        out_path = class_dir / f"{idx:08d}.jpg"
        if out_path.exists():
            written += 1
            pbar.update(1)
            continue
        try:
            _save_image_record(image_value, out_path)
            written += 1
            pbar.update(1)
        except Exception as exc:
            logger.warning("Skipping sample %d in %s due to error: %s", idx, split_name, str(exc))
            continue
    pbar.close()
    return int(written)


def prepare_hf_imagenet_imagefolder(
    output_root: str,
    dataset_id: str = "ILSVRC/imagenet-1k",
    train_split: str = "train",
    val_split: str = "validation",
    image_field: str = "image",
    label_field: str = "label",
    streaming: bool = True,
    max_train_samples: int = 0,
    max_val_samples: int = 0,
    overwrite: bool = False,
) -> Dict[str, int]:
    root = Path(output_root)
    train_dir = root / "train"
    val_dir = root / "val"
    meta_path = root / "prepare_meta.json"

    root.mkdir(parents=True, exist_ok=True)

    if bool(overwrite):
        import shutil

        if train_dir.exists():
            shutil.rmtree(train_dir)
        if val_dir.exists():
            shutil.rmtree(val_dir)

    resolved_train = _resolve_split_name(dataset_id, train_split, 'train')
    resolved_val = _resolve_split_name(dataset_id, val_split, 'validation')

    train_written = _write_split(
        dataset_id=dataset_id,
        split_name=resolved_train,
        split_dir=train_dir,
        image_field=image_field,
        label_field=label_field,
        streaming=streaming,
        max_samples=max_train_samples,
    )
    val_written = _write_split(
        dataset_id=dataset_id,
        split_name=resolved_val,
        split_dir=val_dir,
        image_field=image_field,
        label_field=label_field,
        streaming=streaming,
        max_samples=max_val_samples,
    )

    meta = {
        "dataset_id": str(dataset_id),
        "train_split": str(resolved_train),
        "val_split": str(resolved_val),
        "image_field": str(image_field),
        "label_field": str(label_field),
        "streaming": bool(streaming),
        "max_train_samples": int(max_train_samples),
        "max_val_samples": int(max_val_samples),
        "train_written": int(train_written),
        "val_written": int(val_written),
        "prepared_at_unix": int(time.time()),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    logger.info(
        "Prepared ImageFolder at %s (train=%d, val=%d)",
        str(root),
        int(train_written),
        int(val_written),
    )
    return {"train_written": int(train_written), "val_written": int(val_written)}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare HF image dataset into ImageFolder layout")
    parser.add_argument("--output_root", type=str, required=True, help="Output root directory")
    parser.add_argument("--dataset_id", type=str, default="ILSVRC/imagenet-1k", help="HF dataset id")
    parser.add_argument("--train_split", type=str, default="train", help="Train split name")
    parser.add_argument("--val_split", type=str, default="validation", help="Validation split name")
    parser.add_argument("--image_field", type=str, default="image", help="Image field name")
    parser.add_argument("--label_field", type=str, default="label", help="Label field name")
    parser.add_argument("--streaming", action="store_true", default=True, help="Use HF streaming mode")
    parser.add_argument("--no_streaming", action="store_false", dest="streaming", help="Disable streaming mode")
    parser.add_argument("--max_train_samples", type=int, default=0, help="Optional train cap (0=all)")
    parser.add_argument("--max_val_samples", type=int, default=0, help="Optional val cap (0=all)")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing train/val directories")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args()
    prepare_hf_imagenet_imagefolder(
        output_root=args.output_root,
        dataset_id=args.dataset_id,
        train_split=args.train_split,
        val_split=args.val_split,
        image_field=args.image_field,
        label_field=args.label_field,
        streaming=args.streaming,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
