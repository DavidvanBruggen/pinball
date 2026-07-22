import json
import logging
import os
import random
import hashlib
import gc
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Callable

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset

try:
    from tqdm.auto import tqdm
except Exception:  # tqdm is a hard dep of the trainer; degrade to a no-op if ever absent
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

logger = logging.getLogger(__name__)


def _split_has_images(split_dir: Path) -> bool:
    if not split_dir.is_dir():
        return False
    for class_dir in split_dir.iterdir():
        if not class_dir.is_dir():
            continue
        for item in class_dir.iterdir():
            if item.is_file():
                return True
    return False


def _dataset_root_ready(root: Path) -> bool:
    train_dir = root / "train"
    val_dir = root / "val"
    if _split_has_images(train_dir) and _split_has_images(val_dir):
        return True
    if _split_has_images(root):
        return True
    return False


def _build_image_transforms(image_size: int, frozen_aug: bool = False):
    try:
        from torchvision import transforms
    except Exception as exc:
        raise ImportError("torchvision is required for image loading") from exc

    img_size = int(image_size)
    if frozen_aug:
        train_tf = transforms.Compose(
            [
                transforms.Resize(img_size, antialias=True),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
            ]
        )
    else:
        train_tf = transforms.Compose(
            [
                transforms.Resize(img_size, antialias=True),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
    val_tf = transforms.Compose(
        [
            transforms.Resize(img_size, antialias=True),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]
    )
    return train_tf, val_tf


def _dataset_num_classes(dataset) -> int:
    if isinstance(dataset, Subset):
        return _dataset_num_classes(dataset.dataset)
    return len(getattr(dataset, "classes", []))


class IndexedImageDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        indices: Optional[List[int]] = None,
    ):
        self.base_dataset = base_dataset
        self.indices = indices if indices is not None else list(range(len(base_dataset)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        image, label = self.base_dataset[real_idx]
        return image, label, idx, real_idx


class SampleIndexDataset(Dataset):
    def __init__(self, indices: List[int]):
        self.indices = [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return int(self.indices[idx])


class ImageCacheBackend:
    def __init__(
        self,
        cache_dir: str,
        split: str,
        mode: str,
        shard_size: int = 10000,
        dtype: str = "fp16",
        image_size: int = 256,
        frozen_aug: bool = True,
        overwrite: bool = False,
        verify: bool = True,
        fail_fast: bool = False,
        image_token_mode: str = "latent",
        image_objective: str = "diffusion",
        image_maskgit_variant: str = "continuous",
        image_latent_model_name: str = "stabilityai/sd-vae-ft-mse",
        image_latent_scaling_factor: float = 0.18215,
        image_latent_channels: int = 4,
        image_latent_downsample: int = 8,
        image_maskgit_vq_model_name: str = "",
        image_maskgit_vq_subfolder: Optional[str] = None,
        grid_shape: Optional[Tuple[int, int]] = None,
        image_patch_size: int = 16,
        num_classes: int = 1000,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split = str(split)
        self.mode = str(mode).lower()
        self.shard_size = max(1, int(shard_size))
        self.dtype = str(dtype).lower()
        self.image_size = int(image_size)
        self.frozen_aug = bool(frozen_aug)
        self.overwrite = bool(overwrite)
        self.verify = bool(verify)
        self.fail_fast = bool(fail_fast)
        self.image_token_mode = str(image_token_mode).lower()
        self.image_objective = str(image_objective).lower()
        self.image_maskgit_variant = str(image_maskgit_variant).lower()
        self.image_latent_model_name = str(image_latent_model_name)
        self.image_latent_scaling_factor = float(image_latent_scaling_factor)
        self.image_latent_channels = int(image_latent_channels)
        self.image_latent_downsample = int(image_latent_downsample)
        self.image_maskgit_vq_model_name = str(image_maskgit_vq_model_name or "").strip()
        self.image_maskgit_vq_subfolder = str(image_maskgit_vq_subfolder or "").strip() or None
        self.grid_shape_override = tuple(int(v) for v in grid_shape) if grid_shape is not None else None
        self.image_patch_size = max(1, int(image_patch_size))
        self.num_classes = int(num_classes)

        self._shards: List[Dict[str, Any]] = []
        self._loaded_shards: OrderedDict[int, Dict[str, torch.Tensor]] = OrderedDict()
        self.max_loaded_shards = 4
        self._initialized = False

    def _cache_kind(self) -> Optional[str]:
        if self.mode == "off" or not self.cache_dir:
            return None
        if self.image_objective == "maskgit" and self.image_maskgit_variant == "discrete":
            return "token_ids"
        if self.image_token_mode in {"latent", "raw_rgb_patches"}:
            return "input_features"
        return None

    def _fingerprint(self) -> str:
        parts = [
            f"split={self.split}",
            f"image_size={self.image_size}",
            f"frozen_aug={self.frozen_aug}",
            f"mode={self.mode}",
            f"token_mode={self.image_token_mode}",
            f"objective={self.image_objective}",
            f"variant={self.image_maskgit_variant}",
            f"latent_model={self.image_latent_model_name}",
            f"latent_scale={self.image_latent_scaling_factor}",
            f"latent_channels={self.image_latent_channels}",
            f"latent_downsample={self.image_latent_downsample}",
            f"vq_model={self.image_maskgit_vq_model_name}",
            f"vq_subfolder={self.image_maskgit_vq_subfolder}",
            f"grid_shape={self.grid_shape_override}",
            f"patch_size={self.image_patch_size}",
            f"num_classes={self.num_classes}",
            f"dtype={self.dtype}",
        ]
        fp_str = "|".join(parts)
        return hashlib.sha1(fp_str.encode("utf-8")).hexdigest()[:16]

    def _shard_path(self, shard_idx: int) -> Path:
        return self.cache_dir / f"{self.split}_{self._fingerprint()}_shard{shard_idx:04d}"

    def _metadata_path(self) -> Path:
        return self.cache_dir / f"{self.split}_{self._fingerprint()}_metadata.json"

    def _grid_shape(self) -> Tuple[int, int]:
        if self.grid_shape_override is not None:
            return int(self.grid_shape_override[0]), int(self.grid_shape_override[1])
        if self.image_objective == "maskgit" and self.image_maskgit_variant == "discrete":
            # Discrete caches should always be provided an explicit grid shape.
            return max(1, self.image_size // self.image_latent_downsample), max(1, self.image_size // self.image_latent_downsample)
        if self.image_token_mode == "raw_rgb_patches":
            return max(1, self.image_size // self.image_patch_size), max(1, self.image_size // self.image_patch_size)
        return max(1, self.image_size // self.image_latent_downsample), max(1, self.image_size // self.image_latent_downsample)

    def _token_dim(self) -> int:
        kind = self._cache_kind()
        if kind == "token_ids":
            return 1
        if self.image_token_mode == "raw_rgb_patches":
            return int(3 * self.image_patch_size * self.image_patch_size)
        return int(self.image_latent_channels)

    def _load_shard(self, shard_idx: int) -> Dict[str, Any]:
        path = self._shard_path(shard_idx)
        out: Dict[str, Any] = {}
        tokens_path = path / "tokens.pt"
        if tokens_path.exists():
            out["token_data"] = torch.load(tokens_path, map_location="cpu")
        labels_path = path / "labels.pt"
        if labels_path.exists():
            out["label_data"] = torch.load(labels_path, map_location="cpu")
        indices_path = path / "indices.pt"
        if indices_path.exists():
            out["index_data"] = torch.load(indices_path, map_location="cpu")
        return out

    def _load_shard_cached(self, shard_idx: int) -> Dict[str, Any]:
        shard_idx = int(shard_idx)
        cached = self._loaded_shards.get(shard_idx)
        if cached is not None:
            self._loaded_shards.move_to_end(shard_idx)
            return cached
        data = self._load_shard(int(shard_idx))
        self._loaded_shards[shard_idx] = data
        self._loaded_shards.move_to_end(shard_idx)
        while len(self._loaded_shards) > int(self.max_loaded_shards):
            self._loaded_shards.popitem(last=False)
        return data

    def _save_shard(
        self,
        shard_idx: int,
        token_data: torch.Tensor,
        label_data: torch.Tensor,
        index_data: torch.Tensor,
    ) -> Dict[str, Any]:
        path = self._shard_path(shard_idx)
        path.mkdir(parents=True, exist_ok=True)

        kind = self._cache_kind()
        if kind == "token_ids":
            tokens_out = token_data.to(dtype=torch.int32)
        elif token_data.dtype.is_floating_point:
            tokens_out = token_data.to(dtype=torch.float16 if self.dtype == "fp16" else torch.float32)
        else:
            tokens_out = token_data

        torch.save(tokens_out.contiguous(), path / "tokens.pt")
        torch.save(label_data.to(dtype=torch.long).contiguous(), path / "labels.pt")
        torch.save(index_data.to(dtype=torch.long).contiguous(), path / "indices.pt")

        shard_meta = {
            "shard_idx": int(shard_idx),
            "num_samples": int(token_data.size(0)),
            "dtype": self.dtype,
            "grid_shape": list(self._grid_shape()),
            "token_dim": int(self._token_dim()),
            "cache_kind": kind,
        }
        with open(path / "shard_meta.json", "w", encoding="utf-8") as f:
            json.dump(shard_meta, f)
        return shard_meta

    def _build_shard_index(self) -> List[Dict[str, Any]]:
        if not self.cache_dir or not self.cache_dir.exists():
            return []
        fp = self._fingerprint()
        prefix = f"{self.split}_{fp}_shard"
        shards: List[Dict[str, Any]] = []
        for p in self.cache_dir.iterdir():
            if not (p.is_dir() and p.name.startswith(prefix)):
                continue
            meta_path = p / "shard_meta.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    shard_meta = json.load(f)
                shards.append(shard_meta)
                continue
            tokens_path = p / "tokens.pt"
            if tokens_path.exists():
                data = torch.load(tokens_path, map_location="cpu")
                shards.append(
                    {
                        "shard_idx": int(p.name.split("shard")[-1]),
                        "num_samples": int(data.size(0)),
                        "dtype": self.dtype,
                        "grid_shape": list(self._grid_shape()),
                        "token_dim": int(self._token_dim()),
                        "cache_kind": self._cache_kind(),
                    }
                )
        shards.sort(key=lambda x: int(x["shard_idx"]))
        return shards

    def init(self) -> bool:
        if self.mode == "off" or not self.cache_dir:
            return False
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._shards = self._build_shard_index()
        self._loaded_shards = OrderedDict()
        self._initialized = True
        return bool(self._shards)

    def build_cache(
        self,
        dataset,
        *,
        device: torch.device,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        image_maskgit_vq_tokenizer: Optional[object] = None,
    ) -> Optional[Dict[str, Any]]:
        kind = self._cache_kind()
        if kind is None or self.mode == "off":
            return None
        if self.cache_dir is None:
            raise ValueError("cache_dir is required when image cache is enabled")

        if self.overwrite:
            prefix = f"{self.split}_{self._fingerprint()}_shard"
            if self.cache_dir.exists():
                for p in self.cache_dir.iterdir():
                    if p.is_dir() and p.name.startswith(prefix):
                        shutil.rmtree(p, ignore_errors=True)
                meta_path = self._metadata_path()
                if meta_path.exists():
                    meta_path.unlink()

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        grid_shape = self._grid_shape()
        sample_shape = (grid_shape[0] * grid_shape[1],) if kind == "token_ids" else (
            grid_shape[0] * grid_shape[1],
            self._token_dim(),
        )
        manifest = {
            "version": 1,
            "split": self.split,
            "cache_kind": kind,
            "fingerprint": self._fingerprint(),
            "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
            "sample_shape": [int(v) for v in sample_shape],
            "dtype": self.dtype,
            "shard_size": int(self.shard_size),
            "num_classes": int(self.num_classes),
            "cache_mode": self.mode,
        }

        loader = DataLoader(
            dataset,
            batch_size=max(1, int(batch_size)),
            shuffle=False,
            drop_last=False,
            num_workers=max(0, int(num_workers)),
            pin_memory=bool(pin_memory and device.type == "cuda"),
        )

        tokenizer = None
        vae = None
        if kind == "token_ids":
            tokenizer = image_maskgit_vq_tokenizer
            if tokenizer is None:
                try:
                    from pinball.model.image_maskgit_vq import ImageMaskGITVQTokenizer
                except Exception as exc:
                    raise ImportError("image_maskgit_vq.py is required for discrete image cache building") from exc
                tokenizer = ImageMaskGITVQTokenizer.from_pretrained(
                    self.image_maskgit_vq_model_name,
                    device=device,
                    subfolder=self.image_maskgit_vq_subfolder,
                )
            if hasattr(tokenizer, "to"):
                tokenizer = tokenizer.to(device)
        elif self.image_token_mode == "latent":
            try:
                from diffusers import AutoencoderKL
            except Exception as exc:
                raise ImportError("diffusers is required for latent image cache building") from exc
            vae = AutoencoderKL.from_pretrained(self.image_latent_model_name)
            vae.requires_grad_(False)
            vae.eval()
            vae.to(device)

        logger.info(
            "Building image cache split=%s kind=%s samples=%d shard_size=%d device=%s",
            self.split,
            kind,
            len(dataset),
            int(self.shard_size),
            str(device),
        )

        pending_tokens: List[torch.Tensor] = []
        pending_labels: List[torch.Tensor] = []
        pending_indices: List[torch.Tensor] = []
        shard_idx = 0
        samples_written = 0
        preview_real = None
        preview_encoded = None

        def _flush_pending(final: bool = False):
            nonlocal pending_tokens, pending_labels, pending_indices, shard_idx, samples_written
            if not pending_tokens:
                return
            tokens = torch.cat(pending_tokens, dim=0)
            labels = torch.cat(pending_labels, dim=0)
            indices = torch.cat(pending_indices, dim=0)
            start = 0
            while int(tokens.size(0)) - start >= self.shard_size:
                end = start + self.shard_size
                self._save_shard(shard_idx, tokens[start:end], labels[start:end], indices[start:end])
                shard_idx += 1
                samples_written += int(end - start)
                start = end
            if final and start < int(tokens.size(0)):
                end = int(tokens.size(0))
                self._save_shard(shard_idx, tokens[start:end], labels[start:end], indices[start:end])
                shard_idx += 1
                samples_written += int(end - start)
                start = end
            if start < int(tokens.size(0)):
                pending_tokens = [tokens[start:].contiguous()]
                pending_labels = [labels[start:].contiguous()]
                pending_indices = [indices[start:].contiguous()]
            else:
                pending_tokens = []
                pending_labels = []
                pending_indices = []

        torch.manual_seed(int(samples_written + 17))
        random.seed(int(samples_written + 17))
        np.random.seed((int(samples_written + 17)) % (2**32 - 1))

        pbar = tqdm(
            loader,
            total=len(loader),
            desc=f"Caching {self.split} ({kind})",
            unit="batch",
            dynamic_ncols=True,
        )
        with torch.no_grad():
            for batch in pbar:
                images, labels, sample_indices, _real_indices = batch
                images = images.to(device=device, non_blocking=True)
                labels = labels.to(device="cpu", dtype=torch.long)
                sample_indices = sample_indices.to(device="cpu", dtype=torch.long)

                if kind == "token_ids":
                    token_ids, _ = tokenizer.encode(images)
                    tokens = token_ids.to(device="cpu", dtype=torch.long).contiguous()
                elif self.image_token_mode == "latent":
                    vae_in = images.clamp(0.0, 1.0) * 2.0 - 1.0
                    latent_dist = vae.encode(vae_in).latent_dist
                    latents = latent_dist.sample() * float(self.image_latent_scaling_factor)
                    tokens = latents.permute(0, 2, 3, 1).reshape(images.size(0), -1, int(self.image_latent_channels)).contiguous()
                    tokens = tokens.to(device="cpu")
                elif self.image_token_mode == "raw_rgb_patches":
                    tokens = F.unfold(images, kernel_size=int(self.image_patch_size), stride=int(self.image_patch_size)).transpose(1, 2).contiguous()
                    tokens = tokens.to(device="cpu")
                else:
                    raise ValueError(
                        f"Unsupported cache build config: objective={self.image_objective} variant={self.image_maskgit_variant} token_mode={self.image_token_mode}"
                    )

                if self.verify and tokens.dtype.is_floating_point and not bool(torch.isfinite(tokens).all()):
                    msg = f"Image cache build produced non-finite values for split={self.split}"
                    if self.fail_fast:
                        raise RuntimeError(msg)
                    logger.warning(msg)

                if preview_real is None:
                    preview_real = images[: min(4, images.size(0))].detach().cpu()
                    preview_encoded = tokens[: min(4, tokens.size(0))].detach().cpu()

                pending_tokens.append(tokens)
                pending_labels.append(labels)
                pending_indices.append(sample_indices)

                total_pending = sum(int(t.size(0)) for t in pending_tokens)
                if total_pending >= self.shard_size:
                    _flush_pending()

                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(shards=shard_idx, written=samples_written, refresh=False)

            _flush_pending(final=True)

        if preview_real is not None and preview_encoded is not None and self.verify:
            try:
                gh, gw = grid_shape
                take = int(preview_real.size(0))
                if kind == "token_ids":
                    recon = tokenizer.decode(preview_encoded[:take], grid_shape)
                elif self.image_token_mode == "latent":
                    lat = preview_encoded[:take].reshape(take, gh, gw, int(self.image_latent_channels)).permute(0, 3, 1, 2).contiguous().to(device)
                    dec = vae.decode(lat / float(self.image_latent_scaling_factor)).sample
                    recon = (dec / 2.0 + 0.5).clamp(0.0, 1.0)
                else:
                    patches = preview_encoded[:take].transpose(1, 2).contiguous()
                    recon = F.fold(
                        patches,
                        output_size=(gh * self.image_patch_size, gw * self.image_patch_size),
                        kernel_size=int(self.image_patch_size),
                        stride=int(self.image_patch_size),
                    ).clamp(0.0, 1.0)
                recon = recon.detach().cpu()
                mse = float((preview_real[:take] - recon[:take]).pow(2).mean().item())
                mae = float((preview_real[:take] - recon[:take]).abs().mean().item())
                logger.info(
                    "Image cache roundtrip check split=%s kind=%s mse=%.6f mae=%.6f",
                    self.split,
                    kind,
                    mse,
                    mae,
                )
                if mse > 0.1 and self.fail_fast:
                    raise RuntimeError(f"Image cache roundtrip MSE too high for split={self.split}: {mse:.4f}")
            except Exception as exc:
                if self.fail_fast:
                    raise
                logger.warning("Image cache verification skipped/failed for split=%s: %s", self.split, exc)

        manifest["num_samples"] = int(samples_written)
        manifest["num_shards"] = len(self._build_shard_index())
        with open(self._metadata_path(), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

        if vae is not None:
            del vae
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()

        self._initialized = False
        self._shards = []
        self._loaded_shards = OrderedDict()
        return manifest

    def write_shard_tokens(
        self,
        shard_idx: int,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        indices: torch.Tensor,
    ) -> Dict[str, Any]:
        return self._save_shard(shard_idx, tokens, labels, indices)

    def read_batch(
        self,
        sample_indices: torch.Tensor,
    ) -> Optional[Dict[str, Any]]:
        if not self._initialized or self.mode == "off":
            return None
        if sample_indices is None or sample_indices.numel() == 0:
            return None

        token_rows: List[torch.Tensor] = []
        label_rows: List[torch.Tensor] = []
        index_rows: List[torch.Tensor] = []
        for idx in sample_indices.tolist():
            shard_idx = int(idx) // self.shard_size
            offset = int(idx) % self.shard_size
            shard = self._load_shard_cached(shard_idx)
            if not shard or "token_data" not in shard:
                logger.warning("Cache miss for index %d (shard %d)", int(idx), int(shard_idx))
                return None
            token_rows.append(shard["token_data"][offset : offset + 1])
            label_rows.append(shard["label_data"][offset : offset + 1])
            index_rows.append(shard["index_data"][offset : offset + 1])

        return {
            "tokens": torch.cat(token_rows, dim=0),
            "labels": torch.cat(label_rows, dim=0).long(),
            "indices": torch.cat(index_rows, dim=0).long(),
        }

    def has_cache(self) -> bool:
        return len(self._shards) > 0

    def cache_info(self) -> Dict[str, Any]:
        total = sum(int(s.get("num_samples", 0)) for s in self._shards)
        return {
            "mode": self.mode,
            "split": self.split,
            "fingerprint": self._fingerprint(),
            "total_samples": total,
            "num_shards": len(self._shards),
            "shard_size": self.shard_size,
            "grid_shape": list(self._grid_shape()),
            "token_dim": self._token_dim(),
            "dtype": self.dtype,
            "cache_kind": self._cache_kind(),
        }


def create_image_dataloaders(
    dataset_root: str,
    batch_size: int = 8,
    image_size: int = 256,
    val_split: float = 0.1,
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
    auto_prepare: bool = False,
    hf_dataset_id: str = "ILSVRC/imagenet-1k",
    prepare_streaming: bool = True,
    prepare_max_train_samples: int = 0,
    prepare_max_val_samples: int = 0,
    prepare_overwrite: bool = False,
    cache_mode: str = "off",
    cache_dir: str = "",
    cache_shard_size: int = 10000,
    cache_build_batch_size: int = 64,
    cache_dtype: str = "fp16",
    cache_frozen_aug: bool = True,
    cache_overwrite: bool = False,
    cache_verify: bool = True,
    cache_fail_fast: bool = False,
    cache_image_token_mode: str = "latent",
    cache_image_objective: str = "diffusion",
    cache_image_maskgit_variant: str = "continuous",
    cache_image_latent_model_name: str = "stabilityai/sd-vae-ft-mse",
    cache_image_latent_scaling_factor: float = 0.18215,
    cache_image_latent_channels: int = 4,
    cache_image_latent_downsample: int = 8,
    cache_image_patch_size: int = 16,
    cache_grid_shape: Optional[Tuple[int, int]] = None,
    cache_device: Optional[str] = None,
    cache_image_maskgit_vq_model_name: str = "",
    cache_image_maskgit_vq_subfolder: Optional[str] = None,
    cache_image_maskgit_vq_tokenizer: Optional[object] = None,
    cache_num_classes: int = 1000,
) -> Tuple[Callable, Callable]:
    try:
        from torchvision import datasets
    except Exception as exc:
        raise ImportError("torchvision is required for image loading") from exc

    root = Path(dataset_root)
    if (not root.exists() or not _dataset_root_ready(root)) and bool(auto_prepare):
        logger.info(
            "Image dataset not ready at %s. Auto-preparing from HF dataset %s",
            str(root),
            str(hf_dataset_id),
        )
        from .prepare_hf_imagenet import prepare_hf_imagenet_imagefolder

        prepare_hf_imagenet_imagefolder(
            output_root=str(root),
            dataset_id=str(hf_dataset_id),
            streaming=bool(prepare_streaming),
            max_train_samples=int(prepare_max_train_samples),
            max_val_samples=int(prepare_max_val_samples),
            overwrite=bool(prepare_overwrite),
        )

    if not root.exists():
        raise FileNotFoundError(f"Image dataset root does not exist: {root}")
    if not _dataset_root_ready(root):
        raise FileNotFoundError(
            f"Image dataset root is missing usable data layout at {root}. "
            "Expected train/ and val/ ImageFolder-style directories (or a populated single ImageFolder root)."
        )

    cache_mode = str(cache_mode).lower()
    if cache_mode not in {"off", "read", "write", "readwrite"}:
        cache_mode = "off"
    if cache_mode != "off" and not cache_dir:
        cache_dir = str(root / ".image_cache")

    if cache_device is None:
        cache_device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        cache_device_obj = torch.device(cache_device)

    frozen_aug = bool(cache_frozen_aug and cache_mode != "off")
    train_tf, val_tf = _build_image_transforms(image_size=image_size, frozen_aug=frozen_aug)
    train_dir = root / "train"
    val_dir = root / "val"

    if train_dir.is_dir():
        train_base = datasets.ImageFolder(str(train_dir), transform=train_tf)
        if val_dir.is_dir():
            val_base = datasets.ImageFolder(str(val_dir), transform=val_tf)
        else:
            logger.warning("No val/ directory found; splitting train/ with val_split=%.4f", float(val_split))
            full_train = datasets.ImageFolder(str(train_dir), transform=train_tf)
            full_val = datasets.ImageFolder(str(train_dir), transform=val_tf)
            n = len(full_train)
            n_val = max(1, int(round(float(val_split) * n)))
            n_train = max(1, n - n_val)
            order = list(range(n))
            rng = random.Random(int(seed))
            rng.shuffle(order)
            train_idx = order[:n_train]
            val_idx = order[n_train:]
            train_base = Subset(full_train, train_idx)
            val_base = Subset(full_val, val_idx)
    else:
        logger.warning("No train/ directory found; treating dataset_root as full ImageFolder")
        full_train = datasets.ImageFolder(str(root), transform=train_tf)
        full_val = datasets.ImageFolder(str(root), transform=val_tf)
        n = len(full_train)
        n_val = max(1, int(round(float(val_split) * n)))
        n_train = max(1, n - n_val)
        order = list(range(n))
        rng = random.Random(int(seed))
        rng.shuffle(order)
        train_idx = order[:n_train]
        val_idx = order[n_train:]
        train_base = Subset(full_train, train_idx)
        val_base = Subset(full_val, val_idx)

    train_indexed = IndexedImageDataset(train_base)
    val_indexed = IndexedImageDataset(val_base)
    num_classes = _dataset_num_classes(train_base)

    cache_grid = tuple(cache_grid_shape) if cache_grid_shape is not None else None
    if (
        cache_grid is None
        and str(cache_image_objective).lower() == "maskgit"
        and str(cache_image_maskgit_variant).lower() == "discrete"
        and cache_image_maskgit_vq_tokenizer is not None
        and hasattr(cache_image_maskgit_vq_tokenizer, "infer_grid_shape")
    ):
        try:
            cache_grid = tuple(cache_image_maskgit_vq_tokenizer.infer_grid_shape(int(image_size)))
        except Exception:
            cache_grid = None
    train_cache = ImageCacheBackend(
        cache_dir=cache_dir,
        split="train",
        mode=cache_mode,
        shard_size=cache_shard_size,
        dtype=cache_dtype,
        image_size=image_size,
        frozen_aug=frozen_aug,
        overwrite=cache_overwrite,
        verify=cache_verify,
        fail_fast=cache_fail_fast,
        image_token_mode=cache_image_token_mode,
        image_objective=cache_image_objective,
        image_maskgit_variant=cache_image_maskgit_variant,
        image_latent_model_name=cache_image_latent_model_name,
        image_latent_scaling_factor=cache_image_latent_scaling_factor,
        image_latent_channels=cache_image_latent_channels,
        image_latent_downsample=cache_image_latent_downsample,
        image_maskgit_vq_model_name=cache_image_maskgit_vq_model_name,
        image_maskgit_vq_subfolder=cache_image_maskgit_vq_subfolder,
        grid_shape=cache_grid,
        image_patch_size=cache_image_patch_size,
        num_classes=num_classes,
    )
    val_cache = ImageCacheBackend(
        cache_dir=cache_dir,
        split="val",
        mode=cache_mode,
        shard_size=cache_shard_size,
        dtype=cache_dtype,
        image_size=image_size,
        frozen_aug=frozen_aug,
        overwrite=cache_overwrite,
        verify=cache_verify,
        fail_fast=cache_fail_fast,
        image_token_mode=cache_image_token_mode,
        image_objective=cache_image_objective,
        image_maskgit_variant=cache_image_maskgit_variant,
        image_latent_model_name=cache_image_latent_model_name,
        image_latent_scaling_factor=cache_image_latent_scaling_factor,
        image_latent_channels=cache_image_latent_channels,
        image_latent_downsample=cache_image_latent_downsample,
        image_maskgit_vq_model_name=cache_image_maskgit_vq_model_name,
        image_maskgit_vq_subfolder=cache_image_maskgit_vq_subfolder,
        grid_shape=cache_grid,
        image_patch_size=cache_image_patch_size,
        num_classes=num_classes,
    )

    train_cache.init()
    val_cache.init()

    cache_supported = train_cache._cache_kind() is not None
    if cache_mode != "off" and not cache_supported:
        logger.warning(
            "Image cache requested for token_mode=%s objective=%s variant=%s, but this configuration is not cache-backed. Falling back to on-the-fly encoding.",
            str(cache_image_token_mode),
            str(cache_image_objective),
            str(cache_image_maskgit_variant),
        )
    use_cached_train = cache_supported and cache_mode != "off"
    use_cached_val = cache_supported and cache_mode != "off"

    if cache_mode in {"write", "readwrite"} and cache_supported:
        if cache_mode == "write" or not train_cache.has_cache() or cache_overwrite:
            logger.info("Materializing image cache for train split at %s", str(cache_dir))
            train_cache.build_cache(
                train_indexed,
                device=cache_device_obj,
                batch_size=int(cache_build_batch_size),
                num_workers=max(0, int(num_workers)),
                pin_memory=bool(pin_memory),
                image_maskgit_vq_tokenizer=cache_image_maskgit_vq_tokenizer,
            )
            train_cache.init()

        if cache_mode == "write" or not val_cache.has_cache() or cache_overwrite:
            logger.info("Materializing image cache for val split at %s", str(cache_dir))
            val_cache.build_cache(
                val_indexed,
                device=cache_device_obj,
                batch_size=int(cache_build_batch_size),
                num_workers=max(0, int(num_workers)),
                pin_memory=bool(pin_memory),
                image_maskgit_vq_tokenizer=cache_image_maskgit_vq_tokenizer,
            )
            val_cache.init()

    if cache_mode == "read" and cache_supported and (not train_cache.has_cache() or not val_cache.has_cache()):
        raise FileNotFoundError(f"Requested image cache read mode, but no cache exists at {cache_dir}")

    train_loader_dataset = SampleIndexDataset(list(range(len(train_indexed)))) if use_cached_train and train_cache.has_cache() else train_indexed
    val_loader_dataset = SampleIndexDataset(list(range(len(val_indexed)))) if use_cached_val and val_cache.has_cache() else val_indexed
    val_eval_loader_dataset = val_indexed

    train_loader = DataLoader(
        train_loader_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=max(0, int(num_workers)),
        pin_memory=bool(pin_memory),
    )
    val_loader = DataLoader(
        val_loader_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=bool(pin_memory),
    )
    val_eval_loader = DataLoader(
        val_eval_loader_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=bool(pin_memory),
    )
    val_eval_shuffled_loader = DataLoader(
        val_eval_loader_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=bool(pin_memory),
    )

    logger.info(
        "Image data ready: train=%d val=%d classes=%d image_size=%d cache_mode=%s frozen_aug=%s",
        len(train_indexed),
        len(val_indexed),
        num_classes,
        int(image_size),
        str(cache_mode),
        bool(frozen_aug),
    )

    if train_cache.has_cache():
        logger.info("Train cache: %s", train_cache.cache_info())
    if val_cache.has_cache():
        logger.info("Val cache: %s", val_cache.cache_info())

    state = {
        "train_iter": iter(train_loader),
        "val_iter": iter(val_loader),
        "val_eval_iter": iter(val_eval_loader),
        "val_eval_shuffled_iter": iter(val_eval_shuffled_loader),
        "train_seen": 0,
        "val_seen": 0,
        "val_eval_seen": 0,
    }

    def _next_batch(which: str, device: torch.device):
        if which == "train":
            key = "train_iter"
            loader = train_loader
            cache = train_cache
            cached = cache.has_cache() and cache.mode != "off"
        elif which == "val":
            key = "val_iter"
            loader = val_loader
            cache = val_cache
            cached = cache.has_cache() and cache.mode != "off"
        elif which == "val_eval":
            key = "val_eval_iter"
            loader = val_eval_loader
            cache = None
            cached = False
        elif which == "val_eval_shuffled":
            key = "val_eval_shuffled_iter"
            loader = val_eval_shuffled_loader
            cache = None
            cached = False
        else:
            raise ValueError(f"Unknown batch split: {which}")

        try:
            batch = next(state[key])
        except StopIteration:
            state[key] = iter(loader)
            batch = next(state[key])

        if cached and isinstance(loader.dataset, SampleIndexDataset):
            if not torch.is_tensor(batch):
                batch = torch.as_tensor(batch, dtype=torch.long)
            sample_indices = batch.to(device="cpu", dtype=torch.long)
            cached_batch = cache.read_batch(sample_indices)
            if cached_batch is None:
                raise RuntimeError(f"Failed to read cached image batch for split={which}")

            kind = cache._cache_kind()
            data_key = "token_ids" if kind == "token_ids" else "input_features"
            data_tensor = cached_batch["tokens"].to(device=device)
            if kind == "token_ids":
                data_tensor = data_tensor.to(dtype=torch.long)
            else:
                data_tensor = data_tensor.to(dtype=torch.float32)

            if which == "train":
                state["train_seen"] += int(data_tensor.size(0))
            elif which == "val":
                state["val_seen"] += int(data_tensor.size(0))
            else:
                state["val_eval_seen"] += int(data_tensor.size(0))

            return {
                data_key: data_tensor,
                "class_labels": cached_batch["labels"].to(device=device, dtype=torch.long),
                "attention_mask": None,
                "grid_shape": list(cache._grid_shape()) if cache is not None else None,
                "images_seen": state["train_seen"] if which == "train" else (state["val_seen"] if which == "val" else state["val_eval_seen"]),
                "sample_indices": sample_indices.to(device=device, dtype=torch.long),
            }

        images, labels, sample_indices, _real_indices = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
        sample_indices = sample_indices.to(device=device, dtype=torch.long, non_blocking=True)

        if which == "train":
            state["train_seen"] += int(images.size(0))
        elif which == "val":
            state["val_seen"] += int(images.size(0))
        else:
            state["val_eval_seen"] += int(images.size(0))

        return {
            "pixel_values": images,
            "class_labels": labels,
            "attention_mask": None,
            "images_seen": state["train_seen"] if which == "train" else (state["val_seen"] if which == "val" else state["val_eval_seen"]),
            "sample_indices": sample_indices,
        }

    get_train_batch = lambda device: _next_batch("train", device)
    get_val_batch = lambda device: _next_batch("val", device)
    get_val_eval_batch = lambda device: _next_batch("val_eval", device)
    get_val_eval_shuffled_batch = lambda device: _next_batch("val_eval_shuffled", device)

    def samples_processed(split: str = "train"):
        return int(state["train_seen"] if split == "train" else state["val_seen"])

    get_train_batch.samples_processed = samples_processed
    get_val_batch.samples_processed = samples_processed
    get_val_eval_batch.samples_processed = samples_processed
    get_val_eval_shuffled_batch.samples_processed = samples_processed
    get_train_batch.num_classes = num_classes
    get_val_batch.num_classes = num_classes
    get_val_eval_batch.num_classes = num_classes
    get_val_eval_shuffled_batch.num_classes = num_classes

    get_train_batch.train_cache = train_cache
    get_val_batch.val_cache = val_cache
    get_val_eval_batch.val_cache = None
    get_val_eval_shuffled_batch.val_cache = None
    get_val_batch.raw_eval_batch = get_val_eval_batch
    get_val_batch.raw_eval_shuffled_batch = get_val_eval_shuffled_batch
    get_train_batch.cache_mode = str(cache_mode)
    get_val_batch.cache_mode = str(cache_mode)
    get_val_eval_batch.cache_mode = "off"
    get_val_eval_shuffled_batch.cache_mode = "off"

    return get_train_batch, get_val_batch
