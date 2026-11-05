from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import scipy.io
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as skimage_ssim
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.DRUNet_image import build_drunet
from src.TDnCNN_image import DnCNN_2D, align_lpips_scaling_layer, prepare_for_lpips


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-12
MODELS = ("tdncnn", "drunet")
VARIANTS = ("full", "topk_p10", "topk_p20", "topk_p30")
VARIANT_LABELS = {
    "full": "Full",
    "topk_p10": "Top-K 10%",
    "topk_p20": "Top-K 20%",
    "topk_p30": "Top-K 30%",
}
TRAIN_FRACTIONS = {"full": 1.0, "topk_p10": 0.10, "topk_p20": 0.20, "topk_p30": 0.30}
METRICS = ("PSNR", "SSIM", "LPIPS", "MSE")
LOWER_IS_BETTER = {"LPIPS", "MSE"}


@dataclass(frozen=True)
class SiddExpected:
    scenes: int
    blocks_per_scene: int
    block_size: int
    channels: int
    strict: bool = True


@dataclass(frozen=True)
class MatArraySpec:
    path: Path
    key: str
    raw_shape: tuple[int, ...]
    raw_dtype: str
    axis_roles: dict[str, int]
    mat_backend: str
    array: np.ndarray | None = None

    @property
    def canonical_shape(self) -> tuple[int, int, int, int, int]:
        return (
            int(self.raw_shape[self.axis_roles["scene"]]),
            int(self.raw_shape[self.axis_roles["block"]]),
            int(self.raw_shape[self.axis_roles["channel"]]),
            int(self.raw_shape[self.axis_roles["height"]]),
            int(self.raw_shape[self.axis_roles["width"]]),
        )


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    return payload


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(payload), f=path)


def stable_json_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def git_status_short() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True)
    except Exception as error:
        return f"git status unavailable: {error}"


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def assert_output_root_safe(source_root: Path, output_root: Path) -> None:
    source = source_root.resolve()
    output = output_root.resolve()
    if output == source:
        raise ValueError(f"SIDD output root must not equal source ImageNet experiment root: {output}")
    try:
        output.relative_to(source)
    except ValueError:
        return
    raise ValueError(f"SIDD output root must not be inside source ImageNet experiment root: output={output}, source={source}")


def expected_from_cfg(cfg: dict[str, Any]) -> SiddExpected:
    sidd = cfg["sidd"]
    return SiddExpected(
        scenes=int(sidd["expected_scenes"]),
        blocks_per_scene=int(sidd["expected_blocks_per_scene"]),
        block_size=int(sidd["expected_block_size"]),
        channels=int(sidd["channels"]),
        strict=bool(sidd.get("strict_expected_workload", True)),
    )


def list_hdf5_datasets(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with h5py.File(path, "r") as handle:
            def visitor(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset):
                    rows.append(
                        {
                            "key": name,
                            "shape": [int(v) for v in obj.shape],
                            "dtype": str(obj.dtype),
                            "ndim": int(obj.ndim),
                        }
                    )

            handle.visititems(visitor)
    except OSError:
        return []
    return rows


def list_scipy_mat_arrays(path: Path) -> list[dict[str, Any]]:
    try:
        payload = scipy.io.loadmat(path)
    except Exception:
        return []
    rows = []
    for key, value in payload.items():
        if key.startswith("__") or not isinstance(value, np.ndarray):
            continue
        rows.append({"key": key, "shape": [int(v) for v in value.shape], "dtype": str(value.dtype), "ndim": int(value.ndim)})
    return rows


def _detect_axis_roles(shape: tuple[int, ...], expected: SiddExpected) -> dict[str, int]:
    if len(shape) != 5:
        raise ValueError(f"SIDD MAT dataset must be 5D, got shape={shape}")
    dims = list(map(int, shape))

    def unique_axis(value: int, role: str, used: set[int]) -> int:
        matches = [idx for idx, dim in enumerate(dims) if dim == value and idx not in used]
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely detect {role} axis with size {value} in raw shape {shape}")
        used.add(matches[0])
        return matches[0]

    used: set[int] = set()
    channel_axis = unique_axis(expected.channels, "channel", used)
    scene_axis = unique_axis(expected.scenes, "scene", used)
    block_axis = unique_axis(expected.blocks_per_scene, "block", used)
    spatial_axes = [idx for idx, dim in enumerate(dims) if idx not in used and dim == expected.block_size]
    if len(spatial_axes) != 2:
        raise ValueError(f"Cannot detect two spatial axes of size {expected.block_size} in raw shape {shape}")
    return {
        "scene": scene_axis,
        "block": block_axis,
        "channel": channel_axis,
        "height": spatial_axes[0],
        "width": spatial_axes[1],
    }


def _load_scipy_mat_array_spec(path: Path, expected: SiddExpected) -> MatArraySpec:
    payload = scipy.io.loadmat(path)
    candidates = []
    for key, value in payload.items():
        if key.startswith("__") or not isinstance(value, np.ndarray) or value.ndim != 5:
            continue
        candidates.append((key, value))
    if not candidates:
        raise ValueError(f"No numeric 5D SIDD array found in scipy MAT file: {path}")
    candidates.sort(key=lambda item: int(np.prod(item[1].shape)), reverse=True)
    key, array = candidates[0]
    shape = tuple(int(v) for v in array.shape)
    official_layout = (
        expected.scenes,
        expected.blocks_per_scene,
        expected.block_size,
        expected.block_size,
        expected.channels,
    )
    if shape == official_layout:
        axis_roles = {"scene": 0, "block": 1, "height": 2, "width": 3, "channel": 4}
    else:
        axis_roles = _detect_axis_roles(shape, expected)
    spec = MatArraySpec(path=path, key=str(key), raw_shape=shape, raw_dtype=str(array.dtype), axis_roles=axis_roles, mat_backend="scipy", array=array)
    _validate_spec_shape(spec, expected)
    return spec


def _load_h5py_mat_array_spec(path: Path, expected: SiddExpected) -> MatArraySpec:
    datasets = list_hdf5_datasets(path)
    candidates = [
        row
        for row in datasets
        if int(row["ndim"]) == 5 and not str(row["key"]).startswith("#refs#")
    ]
    if not candidates:
        raise ValueError(f"No numeric 5D SIDD dataset found in {path}. HDF5 keys: {datasets}")
    candidates.sort(key=lambda row: int(np.prod(row["shape"])), reverse=True)
    selected = candidates[0]
    shape = tuple(int(v) for v in selected["shape"])
    axis_roles = _detect_axis_roles(shape, expected)
    spec = MatArraySpec(path=path, key=str(selected["key"]), raw_shape=shape, raw_dtype=str(selected["dtype"]), axis_roles=axis_roles, mat_backend="h5py")
    _validate_spec_shape(spec, expected)
    return spec


def _validate_spec_shape(spec: MatArraySpec, expected: SiddExpected) -> None:
    if expected.strict and spec.canonical_shape != (
        expected.scenes,
        expected.blocks_per_scene,
        expected.channels,
        expected.block_size,
        expected.block_size,
    ):
        raise ValueError(f"Canonical shape mismatch for {spec.path}: {spec.canonical_shape}")


def load_mat_array_spec(path: Path, expected: SiddExpected) -> MatArraySpec:
    try:
        return _load_scipy_mat_array_spec(path, expected)
    except Exception as scipy_error:
        try:
            return _load_h5py_mat_array_spec(path, expected)
        except Exception as h5py_error:
            raise ValueError(f"Unable to load SIDD MAT {path} with scipy or h5py. scipy={scipy_error}; h5py={h5py_error}") from h5py_error


def _to_float01(array: np.ndarray) -> np.ndarray:
    if np.issubdtype(array.dtype, np.integer):
        if array.dtype == np.uint8:
            out = array.astype(np.float32) / 255.0
        elif array.dtype == np.uint16:
            out = array.astype(np.float32) / 65535.0
        else:
            info = np.iinfo(array.dtype)
            out = array.astype(np.float32) / float(info.max)
    else:
        out = array.astype(np.float32)
        finite_max = float(np.nanmax(out)) if out.size else 0.0
        if finite_max > 1.0 and finite_max <= 255.0:
            out = out / 255.0
    return out


class SiddValidationDataset(Dataset):
    def __init__(self, noisy_spec: MatArraySpec, gt_spec: MatArraySpec, *, max_blocks: int | None = None) -> None:
        if noisy_spec.canonical_shape != gt_spec.canonical_shape:
            raise ValueError(f"Noisy/GT canonical shapes differ: {noisy_spec.canonical_shape} vs {gt_spec.canonical_shape}")
        self.noisy_spec = noisy_spec
        self.gt_spec = gt_spec
        self.scenes, self.blocks_per_scene, self.channels, self.height, self.width = noisy_spec.canonical_shape
        self.total_blocks = int(self.scenes * self.blocks_per_scene)
        self.limit = self.total_blocks if max_blocks is None else min(int(max_blocks), self.total_blocks)
        self._noisy_handle: h5py.File | None = None
        self._gt_handle: h5py.File | None = None

    def __len__(self) -> int:
        return int(self.limit)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_noisy_handle"] = None
        state["_gt_handle"] = None
        return state

    def close(self) -> None:
        for handle in [self._noisy_handle, self._gt_handle]:
            if handle is not None:
                handle.close()
        self._noisy_handle = None
        self._gt_handle = None

    def _dataset(self, spec: MatArraySpec, which: str) -> h5py.Dataset | np.ndarray:
        if spec.mat_backend == "scipy":
            if spec.array is None:
                raise RuntimeError(f"Missing in-memory scipy array for {spec.path}")
            return spec.array
        if which == "noisy":
            if self._noisy_handle is None:
                self._noisy_handle = h5py.File(spec.path, "r")
            return self._noisy_handle[spec.key]
        if self._gt_handle is None:
            self._gt_handle = h5py.File(spec.path, "r")
        return self._gt_handle[spec.key]

    def _read_block(self, spec: MatArraySpec, which: str, scene_index: int, block_index: int) -> torch.Tensor:
        index: list[Any] = [slice(None)] * len(spec.raw_shape)
        index[spec.axis_roles["scene"]] = int(scene_index)
        index[spec.axis_roles["block"]] = int(block_index)
        raw = np.asarray(self._dataset(spec, which)[tuple(index)])
        remaining_axes = [axis for axis in range(len(spec.raw_shape)) if axis not in {spec.axis_roles["scene"], spec.axis_roles["block"]}]
        remaining_roles = []
        for axis in remaining_axes:
            for role, role_axis in spec.axis_roles.items():
                if role_axis == axis:
                    remaining_roles.append(role)
                    break
        perm = [remaining_roles.index(role) for role in ["channel", "height", "width"]]
        chw = np.transpose(raw, perm)
        chw = _to_float01(chw)
        return torch.from_numpy(np.ascontiguousarray(chw)).to(torch.float32).clamp(0.0, 1.0)

    @staticmethod
    def ids_from_global(global_block_index: int, blocks_per_scene: int) -> dict[str, int]:
        scene_index = int(global_block_index) // int(blocks_per_scene)
        block_index = int(global_block_index) % int(blocks_per_scene)
        return {
            "scene_index": scene_index,
            "scene_id": scene_index + 1,
            "block_index": block_index,
            "block_id": block_index + 1,
            "global_block_index": int(global_block_index),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        ids = self.ids_from_global(int(index), self.blocks_per_scene)
        noisy = self._read_block(self.noisy_spec, "noisy", ids["scene_index"], ids["block_index"])
        gt = self._read_block(self.gt_spec, "gt", ids["scene_index"], ids["block_index"])
        return {"noisy": noisy, "gt": gt, **ids}


def collate_sidd(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "noisy": torch.stack([item["noisy"] for item in batch], dim=0),
        "gt": torch.stack([item["gt"] for item in batch], dim=0),
        "scene_index": torch.tensor([item["scene_index"] for item in batch], dtype=torch.long),
        "scene_id": torch.tensor([item["scene_id"] for item in batch], dtype=torch.long),
        "block_index": torch.tensor([item["block_index"] for item in batch], dtype=torch.long),
        "block_id": torch.tensor([item["block_id"] for item in batch], dtype=torch.long),
        "global_block_index": torch.tensor([item["global_block_index"] for item in batch], dtype=torch.long),
    }


def psnr_from_mse(mse: float) -> float:
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / float(mse)))


def ssim_item(pred_chw: np.ndarray, gt_chw: np.ndarray) -> float:
    min_hw = int(min(pred_chw.shape[-2:]))
    win_size = min(7, min_hw if min_hw % 2 == 1 else min_hw - 1)
    kwargs: dict[str, Any] = {"data_range": 1.0, "channel_axis": 0}
    if win_size >= 3:
        kwargs["win_size"] = win_size
    return float(skimage_ssim(gt_chw, pred_chw, **kwargs))


def prepare_lpips_rgb_01(x: torch.Tensor) -> torch.Tensor:
    return prepare_for_lpips(x)


def create_lpips_model(device: torch.device) -> nn.Module | None:
    try:
        import lpips

        model = lpips.LPIPS(net="alex").to(device)
        model.eval()
        align_lpips_scaling_layer(model, device)
        return model
    except Exception as error:  # noqa: BLE001
        print(f"[sidd] LPIPS disabled: {error}", flush=True)
        return None


def ensure_lpips_available(ctx: "SiddContext", *, allow_missing: bool) -> nn.Module | None:
    if not bool(ctx.cfg["runtime"].get("lpips_enabled", True)):
        return None
    model = create_lpips_model(ctx.device)
    if model is None and not allow_missing:
        raise RuntimeError("LPIPS is enabled but the LPIPS model/weights are unavailable; full scientific preflight fails.")
    if model is None:
        print("[sidd:warning] LPIPS is enabled but unavailable; continuing only because this is smoke/debug.", flush=True)
    return model


def compute_metrics_batch(pred: torch.Tensor, gt: torch.Tensor, lpips_model: nn.Module | None) -> list[dict[str, float]]:
    pred = pred.detach().cpu().to(torch.float32).clamp(0.0, 1.0)
    gt = gt.detach().cpu().to(torch.float32).clamp(0.0, 1.0)
    mse_values = ((pred - gt) ** 2).flatten(1).mean(dim=1).numpy()
    pred_np = pred.numpy()
    gt_np = gt.numpy()
    lpips_values = np.full((pred.shape[0],), np.nan, dtype=np.float64)
    if lpips_model is not None:
        device = next(lpips_model.parameters()).device
        with torch.inference_mode():
            values = lpips_model(prepare_lpips_rgb_01(pred).to(device), prepare_lpips_rgb_01(gt).to(device))
        lpips_values = values.detach().cpu().view(-1).numpy().astype(np.float64)
    rows = []
    for idx, mse in enumerate(mse_values):
        rows.append(
            {
                "MSE": float(mse),
                "PSNR": psnr_from_mse(float(mse)),
                "SSIM": ssim_item(pred_np[idx], gt_np[idx]),
                "LPIPS": float(lpips_values[idx]),
            }
        )
    return rows


def dataset_manifest(dataset: SiddValidationDataset) -> pd.DataFrame:
    return pd.DataFrame(
        [SiddValidationDataset.ids_from_global(idx, dataset.blocks_per_scene) for idx in range(dataset.total_blocks)]
    )


def _dataset_min_max(spec: MatArraySpec) -> dict[str, float]:
    if spec.mat_backend == "scipy":
        if spec.array is None:
            raise RuntimeError(f"Missing scipy array for {spec.path}")
        return {"min": float(np.min(spec.array)), "max": float(np.max(spec.array))}
    with h5py.File(spec.path, "r") as handle:
        data = handle[spec.key]
        return {"min": float(np.min(data)), "max": float(np.max(data))}


def build_dataset_audit(noisy_path: Path, gt_path: Path, expected: SiddExpected, output_root: Path) -> dict[str, Any]:
    for path in [noisy_path, gt_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing SIDD MAT file: {path}")
    noisy_spec = load_mat_array_spec(noisy_path, expected)
    gt_spec = load_mat_array_spec(gt_path, expected)
    if noisy_spec.canonical_shape != gt_spec.canonical_shape:
        raise ValueError(f"Noisy/GT shape mismatch: {noisy_spec.canonical_shape} vs {gt_spec.canonical_shape}")
    dataset = SiddValidationDataset(noisy_spec, gt_spec)
    sample_indices = sorted(set([0, dataset.total_blocks // 2, dataset.total_blocks - 1]))
    sample_index_set = set(sample_indices)
    sanity_rows = []
    has_difference = False
    final_min = float("inf")
    final_max = float("-inf")
    has_nan_inf = False
    for idx in range(dataset.total_blocks):
        item = dataset[idx]
        noisy = item["noisy"].unsqueeze(0)
        gt = item["gt"].unsqueeze(0)
        final_min = min(final_min, float(noisy.min()), float(gt.min()))
        final_max = max(final_max, float(noisy.max()), float(gt.max()))
        has_nan_inf = has_nan_inf or (not torch.isfinite(noisy).all().item()) or (not torch.isfinite(gt).all().item())
        diff_mse = float(((noisy - gt) ** 2).mean().item())
        has_difference = has_difference or diff_mse > 0.0
        if idx in sample_index_set:
            metrics = compute_metrics_batch(noisy, gt, None)[0]
            sanity_rows.append({**{k: int(v) for k, v in item.items() if isinstance(v, int)}, "noisy_to_gt_psnr": metrics["PSNR"], "mse": diff_mse})
    dataset.close()
    if not has_difference:
        raise ValueError("SIDD audit failed: sampled noisy blocks are identical to GT blocks")
    if has_nan_inf:
        raise ValueError("SIDD audit failed: NaN/Inf detected after canonical conversion")
    audit = {
        "dataset": "SIDD Validation sRGB",
        "color_space": "sRGB",
        "expected_files": {"noisy": noisy_path.name, "gt": gt_path.name},
        "mat_backend": noisy_spec.mat_backend if noisy_spec.mat_backend == gt_spec.mat_backend else "mixed",
        "mat_backends": {"noisy": noisy_spec.mat_backend, "gt": gt_spec.mat_backend},
        "input_sha256": {"noisy": sha256_file(noisy_path), "gt": sha256_file(gt_path)},
        "mat_keys": {"noisy": list_scipy_mat_arrays(noisy_path), "gt": list_scipy_mat_arrays(gt_path)},
        "hdf5_keys": {"noisy": list_hdf5_datasets(noisy_path), "gt": list_hdf5_datasets(gt_path)},
        "noisy": {
            "path": str(noisy_path.resolve()),
            "key": noisy_spec.key,
            "mat_backend": noisy_spec.mat_backend,
            "raw_shape": list(noisy_spec.raw_shape),
            "raw_dtype": noisy_spec.raw_dtype,
            "raw_min_max": _dataset_min_max(noisy_spec),
            "axis_roles": noisy_spec.axis_roles,
            "canonical_shape": list(noisy_spec.canonical_shape),
        },
        "gt": {
            "path": str(gt_path.resolve()),
            "key": gt_spec.key,
            "mat_backend": gt_spec.mat_backend,
            "raw_shape": list(gt_spec.raw_shape),
            "raw_dtype": gt_spec.raw_dtype,
            "raw_min_max": _dataset_min_max(gt_spec),
            "axis_roles": gt_spec.axis_roles,
            "canonical_shape": list(gt_spec.canonical_shape),
        },
        "canonical": {
            "shape": list(noisy_spec.canonical_shape),
            "scenes": int(dataset.scenes),
            "blocks_per_scene": int(dataset.blocks_per_scene),
            "total_blocks": int(dataset.total_blocks),
            "channels": int(dataset.channels),
            "height": int(dataset.height),
            "width": int(dataset.width),
            "float_dtype": "float32",
            "range": [final_min, final_max],
            "nan_or_inf": False,
        },
        "pairing": {
            "shapes_match": True,
            "actual_noisy_gt_difference_sampled": True,
            "indexing": "scene_index and block_index are applied to the detected scene/block axes in both MAT files",
            "dataset_ordering_contract": "global_block_index = scene_index * blocks_per_scene + block_index; scene/block indices are 0-based; IDs are 1-based",
            "sanity_blocks": sanity_rows,
        },
        "preprocessing": {
            "resize": False,
            "center_crop": False,
            "synthetic_noise": False,
            "gamma_correction": False,
            "imagenet_mean_std_normalization": False,
            "resolution_change": False,
            "prediction_clamp_before_metrics": True,
        },
    }
    protocol_dir = output_root / "00_protocol"
    write_json(protocol_dir / "dataset_audit.json", audit)
    lines = [
        "# SIDD Dataset Audit",
        "",
        f"- Dataset: `{audit['dataset']}`",
        f"- Noisy MAT: `{noisy_path}`",
        f"- GT MAT: `{gt_path}`",
        f"- MAT backend noisy/GT: `{noisy_spec.mat_backend}` / `{gt_spec.mat_backend}`",
        f"- Noisy raw shape: `{noisy_spec.raw_shape}` -> canonical `{noisy_spec.canonical_shape}`",
        f"- GT raw shape: `{gt_spec.raw_shape}` -> canonical `{gt_spec.canonical_shape}`",
        f"- Axis roles noisy: `{noisy_spec.axis_roles}`",
        f"- Axis roles GT: `{gt_spec.axis_roles}`",
        f"- Total blocks: `{dataset.total_blocks}`",
        f"- Float range sampled after conversion: `{final_min:.6f}`..`{final_max:.6f}`",
        f"- Dataset ordering: `{audit['pairing']['dataset_ordering_contract']}`",
        "",
        "Sanity noisy->GT PSNR blocks:",
    ]
    for row in sanity_rows:
        lines.append(f"- scene {row['scene_id']} block {row['block_id']}: PSNR `{row['noisy_to_gt_psnr']:.4f}` dB")
    (protocol_dir / "dataset_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    dataset_manifest(dataset).to_csv(protocol_dir / "dataset_manifest.csv", index=False)
    save_visual_audit(dataset, output_root / "00_protocol" / "visual_audit")
    dataset.close()
    return audit


def save_visual_audit(dataset: SiddValidationDataset, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = sorted(set([0, dataset.total_blocks // 2, dataset.total_blocks - 1]))
    for idx in indices:
        item = dataset[idx]
        image = make_panel_image(
            [("GT", item["gt"], None), ("Noisy", item["noisy"], None)],
            title=f"scene {item['scene_id']} block {item['block_id']} global {item['global_block_index']}",
        )
        image.save(output_dir / f"gt_noisy_scene{item['scene_id']:02d}_block{item['block_id']:02d}.png")


def checkpoint_path(source_root: Path, model: str, variant: str) -> Path:
    if model not in MODELS:
        raise ValueError(f"Unsupported model: {model}")
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported variant: {variant}")
    return source_root / "06_downstream" / model / variant / "best.pt"


def load_source_imagenet_config(source_root: Path, fallback_config_path: Path) -> tuple[dict[str, Any], Path, str, str | None]:
    snapshot = source_root / "00_protocol" / "resolved_config.yaml"
    if snapshot.exists():
        return read_yaml(snapshot), snapshot.resolve(), sha256_file(snapshot), None
    if fallback_config_path.exists():
        warning = (
            f"Source run-specific config snapshot is missing at {snapshot}; "
            f"falling back to explicit ImageNet config path {fallback_config_path}."
        )
        print(f"[sidd:warning] {warning}", flush=True)
        return read_yaml(fallback_config_path), fallback_config_path.resolve(), sha256_file(fallback_config_path), warning
    raise FileNotFoundError(f"Missing source ImageNet resolved config snapshot and fallback config: {snapshot}, {fallback_config_path}")


def run_meta_path(checkpoint: Path) -> Path:
    return checkpoint.parent / "run_meta.json"


def build_model(model_name: str, imagenet_cfg: dict[str, Any], device: torch.device) -> nn.Module:
    channels = int(imagenet_cfg.get("dataset", {}).get("channels", 3))
    if model_name == "tdncnn":
        return DnCNN_2D(in_channels=channels).to(device)
    if model_name == "drunet":
        drunet_cfg = dict(imagenet_cfg.get("drunet", {}))
        return build_drunet(
            in_channels=channels,
            features=int(drunet_cfg.get("features", 64)),
            num_layers=int(drunet_cfg.get("num_layers", 5)),
            official=bool(drunet_cfg.get("official", True)),
            nc=drunet_cfg.get("nc", [64, 128, 256, 512]),
            nb=int(drunet_cfg.get("nb", 4)),
            act_mode=str(drunet_cfg.get("act_mode", "R")),
            downsample_mode=str(drunet_cfg.get("downsample_mode", "strideconv")),
            upsample_mode=str(drunet_cfg.get("upsample_mode", "convtranspose")),
        ).to(device)
    raise ValueError(f"Unsupported model: {model_name}")


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    if any(str(key).startswith("module.") for key in state_dict):
        state_dict = {str(key).removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint_strict(model_name: str, checkpoint: Path, imagenet_cfg: dict[str, Any], device: torch.device) -> nn.Module:
    model = build_model(model_name, imagenet_cfg, device)
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(extract_state_dict(payload), strict=True)
    model.eval()
    return model


def checkpoint_provenance(checkpoint: Path, model_name: str, variant: str, imagenet_cfg: dict[str, Any], *, strict_load: bool) -> dict[str, Any]:
    if checkpoint.name != "best.pt":
        raise ValueError(f"Evaluation checkpoint must be best.pt, got {checkpoint}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    meta_path = run_meta_path(checkpoint)
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing run_meta.json for checkpoint: {checkpoint}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    failures = []
    if str(meta.get("model")) != model_name:
        failures.append(f"model={meta.get('model')!r}, expected {model_name!r}")
    if str(meta.get("variant")) != variant:
        failures.append(f"variant={meta.get('variant')!r}, expected {variant!r}")
    if bool(meta.get("early_stopping_enabled", False)) is not True:
        failures.append("early_stopping_enabled is not true")
    if str(meta.get("early_stopping_monitor", "")) != "val_loss":
        failures.append("early_stopping_monitor is not val_loss")
    if str(meta.get("early_stopping_mode", "")) != "min":
        failures.append("early_stopping_mode is not min")
    if failures:
        raise ValueError(f"Invalid checkpoint provenance for {checkpoint}:\n" + "\n".join(failures))
    if strict_load:
        device = torch.device("cpu")
        model = load_checkpoint_strict(model_name, checkpoint, imagenet_cfg, device)
        del model
    return {
        "model": model_name,
        "variant": variant,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_name": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "run_meta": str(meta_path.resolve()),
        "run_meta_sha256": sha256_file(meta_path),
        "best_step": int(meta.get("best_step", meta.get("step", 0))),
        "subset_size": int(meta.get("subset_size", 0)),
        "optimizer_steps": int(meta.get("optimizer_steps", meta.get("stopped_step", 0))),
        "best_val_loss": float(meta.get("best_val_loss", float("nan"))),
        "early_stopping_enabled": bool(meta.get("early_stopping_enabled", False)),
        "early_stopping_monitor": str(meta.get("early_stopping_monitor", "")),
        "early_stopping_mode": str(meta.get("early_stopping_mode", "")),
        "validation_interval_steps": int(meta.get("validation_interval_steps", 0)),
        "training_protocol_hash": str(meta.get("training_protocol_hash", "")),
        "task_noise": meta.get("task_noise"),
        "effective_epochs_at_best": float(meta.get("effective_epochs_at_best", 0.0)),
        "total_seen_samples_at_best": int(meta.get("total_seen_samples_at_best", 0)),
    }


def build_checkpoint_manifest(source_root: Path, variants: Iterable[str], imagenet_cfg: dict[str, Any], output_root: Path, *, strict_load: bool, source_config_path: Path | None = None, source_config_sha256: str | None = None) -> dict[str, Any]:
    missing: list[str] = []
    for model_name in MODELS:
        for variant in variants:
            ckpt = checkpoint_path(source_root, model_name, variant)
            if not ckpt.exists() or not run_meta_path(ckpt).exists():
                missing.append(str(ckpt))
    if missing:
        raise FileNotFoundError("Missing required ImageNet downstream artifacts:\n" + "\n".join(missing))
    rows = [
        checkpoint_provenance(checkpoint_path(source_root, model_name, variant), model_name, variant, imagenet_cfg, strict_load=strict_load)
        for model_name in MODELS
        for variant in variants
    ]
    manifest = {
        "source_imagenet_experiment_root": str(source_root.resolve()),
        "source_imagenet_resolved_config": None if source_config_path is None else str(source_config_path.resolve()),
        "source_imagenet_resolved_config_sha256": source_config_sha256,
        "checkpoints": rows,
    }
    write_json(output_root / "00_protocol" / "checkpoint_manifest.json", manifest)
    pd.DataFrame(rows).to_csv(output_root / "00_protocol" / "checkpoint_manifest.csv", index=False)
    return manifest


def marker_path(output_root: Path, stage: str, model_name: str | None = None, variant: str | None = None, sigma: int | None = None) -> Path:
    stem = stage
    if model_name:
        stem += f"__{model_name}"
    if variant:
        stem += f"__{variant}"
    if sigma is not None:
        stem += f"__sigma{int(sigma)}"
    return output_root / "00_protocol" / "markers" / f"{stem}.success.json"


def marker_payload(ctx: "SiddContext", stage: str, *, checkpoint_sha256: str | None = None, model_name: str | None = None, variant: str | None = None, sigma: int | None = None) -> dict[str, Any]:
    return {
        "stage": stage,
        "model": model_name,
        "variant": variant,
        "sigma": sigma,
        "resolved_config_hash": ctx.config_hash,
        "scientific_identity_hash": ctx.scientific_identity_hash,
        "workload_mode": ctx.workload_mode,
        "max_blocks": ctx.max_blocks,
        "expected_total_blocks": ctx.expected_total_blocks,
        "source_imagenet_resolved_config_sha256": ctx.source_config_sha256,
        "sidd_noisy_sha256": ctx.noisy_sha256,
        "sidd_gt_sha256": ctx.gt_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def marker_valid(ctx: "SiddContext", stage: str, *, checkpoint_sha256: str | None = None, model_name: str | None = None, variant: str | None = None, sigma: int | None = None) -> bool:
    path = marker_path(ctx.output_root, stage, model_name, variant, sigma)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    expected = marker_payload(ctx, stage, checkpoint_sha256=checkpoint_sha256, model_name=model_name, variant=variant, sigma=sigma)
    for key in [
        "stage",
        "model",
        "variant",
        "sigma",
        "resolved_config_hash",
        "scientific_identity_hash",
        "workload_mode",
        "max_blocks",
        "expected_total_blocks",
        "source_imagenet_resolved_config_sha256",
        "sidd_noisy_sha256",
        "sidd_gt_sha256",
        "checkpoint_sha256",
    ]:
        if payload.get(key) != expected.get(key):
            return False
    return True


def write_marker(ctx: "SiddContext", stage: str, *, checkpoint_sha256: str | None = None, model_name: str | None = None, variant: str | None = None, sigma: int | None = None) -> None:
    write_json(marker_path(ctx.output_root, stage, model_name, variant, sigma), marker_payload(ctx, stage, checkpoint_sha256=checkpoint_sha256, model_name=model_name, variant=variant, sigma=sigma))


@dataclass
class SiddContext:
    cfg: dict[str, Any]
    config_path: Path
    output_root: Path
    source_root: Path
    sidd_root: Path
    imagenet_cfg: dict[str, Any]
    imagenet_config_path: Path
    stage: str
    resume: bool = False
    dry_run: bool = False
    max_blocks: int | None = None
    source_config_sha256: str = "UNKNOWN"
    source_config_warning: str | None = None

    @property
    def variants(self) -> list[str]:
        variants = [str(v) for v in self.cfg.get("variants", VARIANTS)]
        invalid = [v for v in variants if v not in VARIANTS]
        if invalid:
            raise ValueError(f"Unsupported SIDD variants: {invalid}")
        return variants

    @property
    def noisy_path(self) -> Path:
        return self.sidd_root / str(self.cfg["sidd"]["noisy_mat_filename"])

    @property
    def gt_path(self) -> Path:
        return self.sidd_root / str(self.cfg["sidd"]["gt_mat_filename"])

    @property
    def noisy_sha256(self) -> str:
        return sha256_file(self.noisy_path) if self.noisy_path.exists() else "MISSING"

    @property
    def gt_sha256(self) -> str:
        return sha256_file(self.gt_path) if self.gt_path.exists() else "MISSING"

    @property
    def config_hash(self) -> str:
        return stable_json_hash(self.cfg)

    @property
    def expected_total_blocks(self) -> int:
        expected = expected_from_cfg(self.cfg)
        return int(expected.scenes * expected.blocks_per_scene)

    @property
    def workload_mode(self) -> str:
        if self.stage == "smoke":
            return "smoke"
        if self.max_blocks is not None:
            return "debug_max_blocks"
        return "full"

    @property
    def is_full_workload(self) -> bool:
        return self.workload_mode == "full"

    @property
    def scientific_identity_hash(self) -> str:
        return stable_json_hash(
            {
                "resolved_config_hash": self.config_hash,
                "workload_mode": self.workload_mode,
                "max_blocks": self.max_blocks,
                "expected_total_blocks": self.expected_total_blocks,
                "source_imagenet_resolved_config_sha256": self.source_config_sha256,
            }
        )

    @property
    def device(self) -> torch.device:
        raw = str(self.cfg.get("runtime", {}).get("device", "cuda"))
        if raw == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(raw)


def safe_runtime_environment() -> dict[str, Any]:
    safe_names = [
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PYTHONHASHSEED",
        "MPLBACKEND",
    ]
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "safe_env": {name: os_value for name in safe_names if (os_value := __import__("os").environ.get(name)) is not None},
    }


def assert_output_protocol_compatible(ctx: SiddContext) -> None:
    provenance = ctx.output_root / "00_protocol" / "run_provenance.json"
    if not provenance.exists():
        return
    try:
        existing = json.loads(provenance.read_text(encoding="utf-8"))
    except Exception as error:
        raise RuntimeError(f"Existing output root has unreadable run_provenance.json: {provenance}") from error
    mismatches = []
    for key, expected_value in {
        "resolved_config_hash": ctx.config_hash,
        "scientific_identity_hash": ctx.scientific_identity_hash,
        "workload_mode": ctx.workload_mode,
        "max_blocks": ctx.max_blocks,
        "expected_total_blocks": ctx.expected_total_blocks,
        "source_imagenet_resolved_config_sha256": ctx.source_config_sha256,
    }.items():
        if existing.get(key) != expected_value:
            mismatches.append(f"{key}: existing={existing.get(key)!r}, current={expected_value!r}")
    if mismatches:
        raise RuntimeError(
            "Existing SIDD output root was created with a different protocol identity. "
            "Use the original config/workload/source snapshot or choose a new --output-root.\n"
            + "\n".join(mismatches)
        )


def write_protocol_files(ctx: SiddContext) -> None:
    assert_output_protocol_compatible(ctx)
    protocol = ctx.output_root / "00_protocol"
    protocol.mkdir(parents=True, exist_ok=True)
    save_yaml(protocol / "resolved_config.yaml", ctx.cfg)
    write_json(
        protocol / "run_provenance.json",
        {
            "command": sys.argv,
            "cwd": str(Path.cwd()),
            "git_commit": git_commit(),
            "git_status_short": git_status_short(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "resolved_config_hash": ctx.config_hash,
            "scientific_identity_hash": ctx.scientific_identity_hash,
            "workload_mode": ctx.workload_mode,
            "max_blocks": ctx.max_blocks,
            "expected_total_blocks": ctx.expected_total_blocks,
            "source_imagenet_resolved_config": str(ctx.imagenet_config_path),
            "source_imagenet_resolved_config_sha256": ctx.source_config_sha256,
            "source_imagenet_config_warning": ctx.source_config_warning,
        },
    )
    write_json(protocol / "environment.json", safe_runtime_environment())
    write_json(
        protocol / "protocol_metadata.json",
        {
            "scientific_goal": "Zero-shot evaluation of already trained ImageNet-100 denoisers on SIDD Validation sRGB.",
            "forbidden_actions": [
                "DDPM training",
                "DDPM inference/scoring",
                "score validation or computation",
                "filtering",
                "TDnCNN/DRUNet training",
                "BN recalibration",
                "finetuning or SIDD-specific adaptation",
                "checkpoint modification",
                "sigma selection using SIDD GT",
                "best Top-K fraction selection by hiding other fractions",
            ],
            "metrics": {
                "range": "[0,1]",
                "MSE": "mean squared error over RGB pixels after prediction clamp to [0,1]",
                "PSNR": "10*log10(1/MSE), inf for identical images",
                "SSIM": "skimage.metrics.structural_similarity with data_range=1.0 and channel_axis=0",
                "LPIPS": "LPIPS AlexNet on RGB tensors mapped from [0,1] to [-1,1], no ImageNet mean/std",
                "primary": "PSNR",
                "lpips_enabled": bool(ctx.cfg["runtime"].get("lpips_enabled", True)),
                "lpips_preparation": "src.TDnCNN_image.prepare_for_lpips, matching existing ImageNet/external pipeline",
            },
            "drunet_sigma_policy": {
                "assumed_sigma_255": int(ctx.cfg["drunet"]["primary_sigma_255"]),
                "assumed_sigma_normalized": float(int(ctx.cfg["drunet"]["primary_sigma_255"]) / 255.0),
                "sigma_role": "primary_pre_registered_condition",
                "sensitivity_sigmas_255": [int(v) for v in ctx.cfg["drunet"]["sensitivity_sigmas_255"]],
                "best_sigma_selection": False,
            },
        },
    )


def preflight(ctx: SiddContext, *, strict_load: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    assert_output_root_safe(ctx.source_root, ctx.output_root)
    write_protocol_files(ctx)
    audit = build_dataset_audit(ctx.noisy_path, ctx.gt_path, expected_from_cfg(ctx.cfg), ctx.output_root)
    lpips_probe = ensure_lpips_available(ctx, allow_missing=not ctx.is_full_workload)
    del lpips_probe
    manifest = build_checkpoint_manifest(
        ctx.source_root,
        ctx.variants,
        ctx.imagenet_cfg,
        ctx.output_root,
        strict_load=strict_load,
        source_config_path=ctx.imagenet_config_path,
        source_config_sha256=ctx.source_config_sha256,
    )
    write_marker(ctx, "preflight")
    return audit, manifest


def open_dataset(ctx: SiddContext) -> SiddValidationDataset:
    expected = expected_from_cfg(ctx.cfg)
    noisy_spec = load_mat_array_spec(ctx.noisy_path, expected)
    gt_spec = load_mat_array_spec(ctx.gt_path, expected)
    return SiddValidationDataset(noisy_spec, gt_spec, max_blocks=ctx.max_blocks)


def log_eval_progress(model_name: str, variant: str, sigma: int | None, batch_idx: int, total_batches: int, processed: int, start: float, device: torch.device) -> None:
    elapsed = time.time() - start
    fields = {
        "model": model_name,
        "variant": variant,
        "sigma": sigma,
        "batch": batch_idx,
        "total_batches": total_batches,
        "blocks_processed": processed,
        "elapsed_sec": round(elapsed, 3),
        "blocks_per_sec": round(processed / max(elapsed, EPS), 4),
    }
    if device.type == "cuda":
        fields["cuda_allocated_mb"] = round(float(torch.cuda.memory_allocated(device) / 1024**2), 2)
        fields["cuda_reserved_mb"] = round(float(torch.cuda.memory_reserved(device) / 1024**2), 2)
    print("[sidd:evaluate] " + json.dumps(fields, sort_keys=True), flush=True)


def row_ids(batch: dict[str, Any], idx: int) -> dict[str, int]:
    return {
        "scene_index": int(batch["scene_index"][idx]),
        "scene_id": int(batch["scene_id"][idx]),
        "block_index": int(batch["block_index"][idx]),
        "block_id": int(batch["block_id"][idx]),
        "global_block_index": int(batch["global_block_index"][idx]),
    }


def baseline_metrics(ctx: SiddContext, dataset: SiddValidationDataset, lpips_model: nn.Module | None) -> pd.DataFrame:
    out = ctx.output_root / "02_evaluation" / "baseline_noisy_input.csv"
    if ctx.resume and out.exists() and marker_valid(ctx, "baseline"):
        return pd.read_csv(out)
    loader = DataLoader(dataset, batch_size=max(1, int(ctx.cfg["runtime"]["tdncnn_batch_size"])), shuffle=False, num_workers=int(ctx.cfg["runtime"].get("num_workers", 0)), collate_fn=collate_sidd)
    rows = []
    for batch in loader:
        metrics = compute_metrics_batch(batch["noisy"], batch["gt"], lpips_model)
        for idx, item_metrics in enumerate(metrics):
            rows.append(
                {
                    **row_ids(batch, idx),
                    "model": "noisy_input",
                    "variant": "baseline",
                    "assumed_sigma_255": np.nan,
                    "assumed_sigma_normalized": np.nan,
                    **item_metrics,
                }
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    write_marker(ctx, "baseline")
    return pd.DataFrame(rows)


def predict_batch(model: nn.Module, model_name: str, noisy: torch.Tensor, sigma_255: int | None, device: torch.device) -> torch.Tensor:
    x = noisy.to(device, non_blocking=True)
    with torch.inference_mode():
        if model_name == "drunet":
            sigma = float(int(sigma_255) / 255.0)
            pred = model(x, sigma=sigma)
        else:
            pred = model(x)
    return pred.detach().cpu().clamp(0.0, 1.0)


def evaluate_combination(ctx: SiddContext, dataset: SiddValidationDataset, checkpoint_manifest: dict[str, Any], model_name: str, variant: str, sigma_255: int | None, lpips_model: nn.Module | None) -> Path:
    manifest_by_key = {(row["model"], row["variant"]): row for row in checkpoint_manifest["checkpoints"]}
    ckpt_row = manifest_by_key[(model_name, variant)]
    out_dir = ctx.output_root / "02_evaluation" / model_name / variant
    if sigma_255 is not None:
        out_dir = out_dir / f"sigma_{int(sigma_255):03d}"
    out = out_dir / "per_block_metrics.csv"
    if ctx.resume and out.exists() and marker_valid(ctx, "evaluate", checkpoint_sha256=ckpt_row["checkpoint_sha256"], model_name=model_name, variant=variant, sigma=sigma_255):
        print(f"[sidd:resume] {model_name} {variant} sigma={sigma_255} -> {out}", flush=True)
        return out
    device = ctx.device
    batch_size_key = "drunet_batch_size" if model_name == "drunet" else "tdncnn_batch_size"
    loader = DataLoader(dataset, batch_size=max(1, int(ctx.cfg["runtime"][batch_size_key])), shuffle=False, num_workers=int(ctx.cfg["runtime"].get("num_workers", 0)), collate_fn=collate_sidd, pin_memory=device.type == "cuda")
    model = load_checkpoint_strict(model_name, Path(ckpt_row["checkpoint"]), ctx.imagenet_cfg, device)
    rows = []
    start = time.time()
    total_batches = int(math.ceil(len(dataset) / max(1, int(ctx.cfg["runtime"][batch_size_key]))))
    processed = 0
    for batch_idx, batch in enumerate(loader, start=1):
        pred = predict_batch(model, model_name, batch["noisy"], sigma_255, device)
        metrics = compute_metrics_batch(pred, batch["gt"], lpips_model)
        for idx, item_metrics in enumerate(metrics):
            rows.append(
                {
                    **row_ids(batch, idx),
                    "model": model_name,
                    "variant": variant,
                    "assumed_sigma_255": np.nan if sigma_255 is None else int(sigma_255),
                    "assumed_sigma_normalized": np.nan if sigma_255 is None else float(int(sigma_255) / 255.0),
                    **item_metrics,
                }
            )
        processed += int(batch["noisy"].shape[0])
        log_eval_progress(model_name, variant, sigma_255, batch_idx, total_batches, processed, start, device)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    write_marker(ctx, "evaluate", checkpoint_sha256=ckpt_row["checkpoint_sha256"], model_name=model_name, variant=variant, sigma=sigma_255)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def combination_output_path(ctx: SiddContext, model_name: str, variant: str, sigma_255: int | None) -> Path:
    out_dir = ctx.output_root / "02_evaluation" / model_name / variant
    if sigma_255 is not None:
        out_dir = out_dir / f"sigma_{int(sigma_255):03d}"
    return out_dir / "per_block_metrics.csv"


def evaluate_loaded_combination(
    ctx: SiddContext,
    dataset: SiddValidationDataset,
    model: nn.Module,
    ckpt_row: dict[str, Any],
    model_name: str,
    variant: str,
    sigma_255: int | None,
    lpips_model: nn.Module | None,
) -> Path:
    out = combination_output_path(ctx, model_name, variant, sigma_255)
    if ctx.resume and out.exists() and marker_valid(ctx, "evaluate", checkpoint_sha256=ckpt_row["checkpoint_sha256"], model_name=model_name, variant=variant, sigma=sigma_255):
        print(f"[sidd:resume] {model_name} {variant} sigma={sigma_255} -> {out}", flush=True)
        return out
    device = ctx.device
    batch_size_key = "drunet_batch_size" if model_name == "drunet" else "tdncnn_batch_size"
    batch_size = max(1, int(ctx.cfg["runtime"][batch_size_key]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=int(ctx.cfg["runtime"].get("num_workers", 0)), collate_fn=collate_sidd, pin_memory=device.type == "cuda")
    rows = []
    start = time.time()
    total_batches = int(math.ceil(len(dataset) / batch_size))
    processed = 0
    for batch_idx, batch in enumerate(loader, start=1):
        pred = predict_batch(model, model_name, batch["noisy"], sigma_255, device)
        metrics = compute_metrics_batch(pred, batch["gt"], lpips_model)
        for idx, item_metrics in enumerate(metrics):
            rows.append(
                {
                    **row_ids(batch, idx),
                    "model": model_name,
                    "variant": variant,
                    "assumed_sigma_255": np.nan if sigma_255 is None else int(sigma_255),
                    "assumed_sigma_normalized": np.nan if sigma_255 is None else float(int(sigma_255) / 255.0),
                    **item_metrics,
                }
            )
        processed += int(batch["noisy"].shape[0])
        log_eval_progress(model_name, variant, sigma_255, batch_idx, total_batches, processed, start, device)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    write_marker(ctx, "evaluate", checkpoint_sha256=ckpt_row["checkpoint_sha256"], model_name=model_name, variant=variant, sigma=sigma_255)
    return out


def evaluate(ctx: SiddContext, *, smoke: bool = False) -> list[Path]:
    if ctx.dry_run:
        print(json.dumps({"dry_run": True, "stage": "evaluate", "output_root": str(ctx.output_root)}, indent=2))
        return []
    audit, checkpoint_manifest = preflight(ctx, strict_load=not smoke)
    dataset = open_dataset(ctx)
    lpips_model = ensure_lpips_available(ctx, allow_missing=smoke or not ctx.is_full_workload)
    baseline_metrics(ctx, dataset, lpips_model)
    outputs = []
    primary_sigma = int(ctx.cfg["drunet"]["primary_sigma_255"])
    drunet_sigmas = [primary_sigma] if smoke else [int(v) for v in ctx.cfg["drunet"]["sensitivity_sigmas_255"]]
    manifest_by_key = {(row["model"], row["variant"]): row for row in checkpoint_manifest["checkpoints"]}
    for model_name in MODELS:
        for variant in ctx.variants:
            if model_name == "tdncnn":
                outputs.append(evaluate_combination(ctx, dataset, checkpoint_manifest, model_name, variant, None, lpips_model))
            else:
                ckpt_row = manifest_by_key[(model_name, variant)]
                model = load_checkpoint_strict(model_name, Path(ckpt_row["checkpoint"]), ctx.imagenet_cfg, ctx.device)
                for sigma in drunet_sigmas:
                    outputs.append(evaluate_loaded_combination(ctx, dataset, model, ckpt_row, model_name, variant, sigma, lpips_model))
                del model
                if ctx.device.type == "cuda":
                    torch.cuda.empty_cache()
    dataset.close()
    write_marker(ctx, "smoke" if smoke else "evaluate_all")
    return outputs


def expected_result_shards(ctx: SiddContext) -> list[tuple[Path, dict[str, Any]]]:
    manifest_path = ctx.output_root / "00_protocol" / "checkpoint_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing checkpoint manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_by_key = {(row["model"], row["variant"]): row for row in manifest["checkpoints"]}
    rows: list[tuple[Path, dict[str, Any]]] = []
    drunet_sigmas = [int(ctx.cfg["drunet"]["primary_sigma_255"])] if ctx.workload_mode == "smoke" else [int(v) for v in ctx.cfg["drunet"]["sensitivity_sigmas_255"]]
    for model_name in MODELS:
        for variant in ctx.variants:
            if model_name == "tdncnn":
                ckpt_row = manifest_by_key[(model_name, variant)]
                rows.append((combination_output_path(ctx, model_name, variant, None), {"model": model_name, "variant": variant, "sigma": None, "checkpoint_sha256": ckpt_row["checkpoint_sha256"]}))
            else:
                ckpt_row = manifest_by_key[(model_name, variant)]
                for sigma in drunet_sigmas:
                    rows.append((combination_output_path(ctx, model_name, variant, sigma), {"model": model_name, "variant": variant, "sigma": sigma, "checkpoint_sha256": ckpt_row["checkpoint_sha256"]}))
    return rows


def load_all_per_block(ctx: SiddContext) -> pd.DataFrame:
    frames = []
    baseline = ctx.output_root / "02_evaluation" / "baseline_noisy_input.csv"
    if not baseline.exists() or not marker_valid(ctx, "baseline"):
        raise RuntimeError(f"Missing or stale baseline result/marker: {baseline}")
    frames.append(pd.read_csv(baseline))
    missing = []
    for path, spec in expected_result_shards(ctx):
        if not path.exists() or not marker_valid(ctx, "evaluate", checkpoint_sha256=spec["checkpoint_sha256"], model_name=spec["model"], variant=spec["variant"], sigma=spec["sigma"]):
            missing.append(str(path))
            continue
        frames.append(pd.read_csv(path))
    if missing:
        raise RuntimeError("Missing or stale expected SIDD result shards:\n" + "\n".join(missing))
    return pd.concat(frames, ignore_index=True)


def _same_sigma(series: pd.Series, expected_sigma: int | None) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if expected_sigma is None:
        return values.isna()
    return values == int(expected_sigma)


def validate_production_completeness(per_block: pd.DataFrame, ctx: SiddContext) -> None:
    expected = expected_from_cfg(ctx.cfg)
    expected_total = int(expected.scenes * expected.blocks_per_scene)
    if expected_total != 1280:
        raise RuntimeError(f"Full SIDD production analysis requires expected total blocks=1280, got {expected_total}")
    required_cols = {"scene_index", "scene_id", "block_index", "block_id", "global_block_index", "model", "variant", "assumed_sigma_255", *METRICS}
    missing_cols = sorted(required_cols - set(per_block.columns))
    if missing_cols:
        raise RuntimeError(f"SIDD per-block table is missing required columns: {missing_cols}")
    expected_indices = set(range(expected_total))
    failures = []

    def check_subset(frame: pd.DataFrame, label: str) -> None:
        if len(frame) != expected_total:
            failures.append(f"{label}: expected {expected_total} rows, got {len(frame)}")
            return
        key_cols = ["model", "variant", "assumed_sigma_255", "global_block_index"]
        duplicates = int(frame.duplicated(key_cols).sum())
        if duplicates:
            failures.append(f"{label}: duplicate rows by {key_cols}: {duplicates}")
        indices = set(pd.to_numeric(frame["global_block_index"], errors="coerce").dropna().astype(int).tolist())
        if indices != expected_indices:
            failures.append(f"{label}: global_block_index is not exactly 0..{expected_total - 1}")
        scenes = set(pd.to_numeric(frame["scene_id"], errors="coerce").dropna().astype(int).tolist())
        blocks = set(pd.to_numeric(frame["block_id"], errors="coerce").dropna().astype(int).tolist())
        if scenes != set(range(1, expected.scenes + 1)) or blocks != set(range(1, expected.blocks_per_scene + 1)):
            failures.append(f"{label}: scene/block coverage is not {expected.scenes}x{expected.blocks_per_scene}")
        for metric in ["PSNR", "SSIM", "MSE"]:
            values = pd.to_numeric(frame[metric], errors="coerce")
            if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
                failures.append(f"{label}: non-finite {metric}")
        if bool(ctx.cfg["runtime"].get("lpips_enabled", True)):
            values = pd.to_numeric(frame["LPIPS"], errors="coerce")
            if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
                failures.append(f"{label}: LPIPS enabled but non-finite LPIPS values found")

    check_subset(per_block[(per_block["model"] == "noisy_input") & (per_block["variant"] == "baseline")], "baseline noisy_input")
    for variant in ctx.variants:
        check_subset(per_block[(per_block["model"] == "tdncnn") & (per_block["variant"] == variant) & _same_sigma(per_block["assumed_sigma_255"], None)], f"tdncnn/{variant}")
    for variant in ctx.variants:
        for sigma in [int(v) for v in ctx.cfg["drunet"]["sensitivity_sigmas_255"]]:
            check_subset(per_block[(per_block["model"] == "drunet") & (per_block["variant"] == variant) & _same_sigma(per_block["assumed_sigma_255"], sigma)], f"drunet/{variant}/sigma_{sigma}")
    observed = set(zip(per_block["model"].astype(str), per_block["variant"].astype(str), pd.to_numeric(per_block["assumed_sigma_255"], errors="coerce").fillna(-1).astype(int)))
    expected_combos = {("noisy_input", "baseline", -1)}
    expected_combos.update(("tdncnn", variant, -1) for variant in ctx.variants)
    expected_combos.update(("drunet", variant, int(sigma)) for variant in ctx.variants for sigma in ctx.cfg["drunet"]["sensitivity_sigmas_255"])
    missing_combos = sorted(expected_combos - observed)
    if missing_combos:
        failures.append(f"missing expected combinations: {missing_combos}")
    if failures:
        raise RuntimeError("SIDD production completeness gate failed:\n" + "\n".join(failures))


def aggregate_block_summary(per_block: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["model", "variant", "assumed_sigma_255", "assumed_sigma_normalized"]
    for keys, group in per_block.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, keys))
        row["N"] = int(len(group))
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
            row[f"{metric}_median"] = float(values.median()) if values.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def scene_metrics(per_block: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["scene_index", "scene_id", "model", "variant", "assumed_sigma_255", "assumed_sigma_normalized"]
    return per_block.groupby(group_cols, dropna=False, sort=False)[list(METRICS)].mean(numeric_only=True).reset_index()


def bootstrap_ci(values: np.ndarray, *, seed: int, iterations: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(iterations), dtype=np.float64)
    for idx in range(int(iterations)):
        sample = rng.choice(values, size=len(values), replace=True)
        means[idx] = float(np.mean(sample))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_vs_full(scene_df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired_rows = []
    summary_rows = []
    seed = int(cfg["statistics"]["bootstrap_seed"])
    iters = int(cfg["statistics"]["bootstrap_iterations"])
    for model_name in MODELS:
        model_rows = scene_df[scene_df["model"] == model_name]
        for sigma in sorted(model_rows["assumed_sigma_255"].dropna().unique().tolist()) if model_name == "drunet" else [np.nan]:
            sigma_rows = model_rows if model_name == "tdncnn" else model_rows[model_rows["assumed_sigma_255"] == sigma]
            full = sigma_rows[sigma_rows["variant"] == "full"].set_index("scene_index")
            if full.empty:
                continue
            for variant in ["topk_p10", "topk_p20", "topk_p30"]:
                topk = sigma_rows[sigma_rows["variant"] == variant].set_index("scene_index")
                joined = topk.join(full, lsuffix="_topk", rsuffix="_full", how="inner")
                metric_deltas: dict[str, np.ndarray] = {}
                for scene_index, row in joined.iterrows():
                    item = {
                        "scene_index": int(scene_index),
                        "scene_id": int(row["scene_id_topk"]),
                        "model": model_name,
                        "topk_variant": variant,
                        "assumed_sigma_255": np.nan if pd.isna(sigma) else int(sigma),
                    }
                    for metric in METRICS:
                        item[f"delta_{metric}"] = float(row[f"{metric}_topk"] - row[f"{metric}_full"])
                    paired_rows.append(item)
                for metric in METRICS:
                    deltas = (joined[f"{metric}_topk"] - joined[f"{metric}_full"]).to_numpy(dtype=np.float64)
                    metric_deltas[metric] = deltas
                    ci_low, ci_high = bootstrap_ci(deltas, seed=seed, iterations=iters)
                    improved = deltas < 0 if metric in LOWER_IS_BETTER else deltas > 0
                    summary_rows.append(
                        {
                            "model": model_name,
                            "topk_variant": variant,
                            "assumed_sigma_255": np.nan if pd.isna(sigma) else int(sigma),
                            "metric": metric,
                            "N_scenes": int(len(deltas)),
                            "mean_delta": float(np.mean(deltas)) if len(deltas) else np.nan,
                            "median_delta": float(np.median(deltas)) if len(deltas) else np.nan,
                            "std_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                            "bootstrap_ci95_low": ci_low,
                            "bootstrap_ci95_high": ci_high,
                            "fraction_scenes_improved": float(np.mean(improved)) if len(deltas) else np.nan,
                            "number_scenes_improved": int(np.sum(improved)),
                            "number_scenes_total": int(len(deltas)),
                            "improvement_direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
                        }
                    )
    return pd.DataFrame(paired_rows), pd.DataFrame(summary_rows)


def primary_sidd_table(agg: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    primary_sigma = int(cfg["drunet"]["primary_sigma_255"])
    for model_name in MODELS:
        model_rows = agg[agg["model"] == model_name]
        if model_name == "drunet":
            model_rows = model_rows[model_rows["assumed_sigma_255"] == primary_sigma]
        else:
            model_rows = model_rows[model_rows["assumed_sigma_255"].isna()]
        full = model_rows[model_rows["variant"] == "full"]
        if full.empty:
            continue
        full_psnr = float(full.iloc[0]["PSNR_mean"])
        for variant in VARIANTS:
            row = model_rows[model_rows["variant"] == variant]
            if row.empty:
                continue
            item = row.iloc[0]
            rows.append(
                {
                    "Model": "TDnCNN" if model_name == "tdncnn" else "DRUNet",
                    "Variant": VARIANT_LABELS[variant],
                    "Train fraction": TRAIN_FRACTIONS[variant],
                    "PSNR": float(item["PSNR_mean"]),
                    "Delta PSNR vs Full": float(item["PSNR_mean"] - full_psnr),
                    "SSIM": float(item["SSIM_mean"]),
                    "LPIPS": float(item["LPIPS_mean"]),
                    "MSE": float(item["MSE_mean"]),
                }
            )
    return pd.DataFrame(rows)


def drunet_sensitivity(agg: pd.DataFrame) -> pd.DataFrame:
    rows = agg[(agg["model"] == "drunet") & (agg["variant"].isin(VARIANTS))].copy()
    return rows[["model", "variant", "assumed_sigma_255", "assumed_sigma_normalized", "N", "PSNR_mean", "SSIM_mean", "LPIPS_mean", "MSE_mean"]]


def analyze(ctx: SiddContext) -> None:
    assert_output_protocol_compatible(ctx)
    out = ctx.output_root / "03_analysis"
    out.mkdir(parents=True, exist_ok=True)
    per_block = load_all_per_block(ctx)
    if ctx.is_full_workload:
        validate_production_completeness(per_block, ctx)
    per_block.to_csv(out / "per_block_metrics.csv", index=False)
    agg = aggregate_block_summary(per_block)
    agg.to_csv(out / "aggregate_block_summary.csv", index=False)
    scenes = scene_metrics(per_block)
    scenes.to_csv(out / "per_scene_metrics.csv", index=False)
    paired, paired_summary = paired_vs_full(scenes, ctx.cfg)
    paired.to_csv(out / "paired_vs_full_per_scene.csv", index=False)
    paired_summary.to_csv(out / "paired_vs_full_summary.csv", index=False)
    drunet_sensitivity(agg).to_csv(out / "drunet_sigma_sensitivity.csv", index=False)
    primary_sidd_table(agg, ctx.cfg).to_csv(out / "primary_sidd_table.csv", index=False)
    write_marker(ctx, "analyze")


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "calibrib.ttf" if bold else "calibri.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def tensor_to_image(x: torch.Tensor, size: int = 256) -> Image.Image:
    arr = x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    img = Image.fromarray((arr * 255.0).round().astype(np.uint8))
    if img.width != size or img.height != size:
        img = img.resize((size, size), Image.Resampling.BILINEAR)
    return img


def make_panel_image(columns: list[tuple[str, torch.Tensor, dict[str, float] | None]], title: str = "") -> Image.Image:
    cell = 256
    label_h = 48
    title_h = 34 if title else 0
    width = cell * len(columns)
    height = title_h + cell + label_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = font(17, True)
    small_font = font(13)
    if title:
        draw.text((8, 8), title, font=header_font, fill=(0, 0, 0))
    for idx, (label, tensor, metrics) in enumerate(columns):
        x0 = idx * cell
        canvas.paste(tensor_to_image(tensor, cell), (x0, title_h))
        text = label
        if metrics is not None:
            text += f"\nPSNR {metrics['PSNR']:.2f} SSIM {metrics['SSIM']:.3f}"
        draw.multiline_text((x0 + 6, title_h + cell + 5), text, font=small_font, fill=(0, 0, 0), spacing=2)
    return canvas


def select_deterministic_examples(per_block: pd.DataFrame) -> list[int]:
    base = per_block[per_block["model"] == "noisy_input"].sort_values("PSNR").reset_index(drop=True)
    if base.empty:
        return []
    positions = [0, len(base) // 2, len(base) - 1]
    return [int(base.iloc[pos]["global_block_index"]) for pos in positions]


def qualitative_panels(ctx: SiddContext) -> None:
    analysis = ctx.output_root / "03_analysis" / "per_block_metrics.csv"
    if not analysis.exists():
        return
    per_block = pd.read_csv(analysis)
    examples = select_deterministic_examples(per_block)
    if not examples:
        return
    dataset = open_dataset(ctx)
    manifest = json.loads((ctx.output_root / "00_protocol" / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    manifest_by_key = {(row["model"], row["variant"]): row for row in manifest["checkpoints"]}
    device = ctx.device
    primary_sigma = int(ctx.cfg["drunet"]["primary_sigma_255"])
    for model_name in MODELS:
        models = {}
        for variant in ctx.variants:
            row = manifest_by_key[(model_name, variant)]
            models[variant] = load_checkpoint_strict(model_name, Path(row["checkpoint"]), ctx.imagenet_cfg, device)
        for global_idx in examples:
            item = dataset[global_idx]
            columns: list[tuple[str, torch.Tensor, dict[str, float] | None]] = [("GT", item["gt"], None), ("Noisy", item["noisy"], compute_metrics_batch(item["noisy"].unsqueeze(0), item["gt"].unsqueeze(0), None)[0])]
            for variant in ctx.variants:
                pred = predict_batch(models[variant], model_name, item["noisy"].unsqueeze(0), primary_sigma if model_name == "drunet" else None, device).squeeze(0)
                metrics = compute_metrics_batch(pred.unsqueeze(0), item["gt"].unsqueeze(0), None)[0]
                columns.append((VARIANT_LABELS[variant], pred, metrics))
            image = make_panel_image(columns, title=f"{model_name} deterministic baseline-PSNR example global {global_idx}")
            out = ctx.output_root / "04_figures" / model_name / f"deterministic_global_{global_idx:04d}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            image.save(out)
        for model in models.values():
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    dataset.close()


def summary(ctx: SiddContext) -> None:
    qualitative_panels(ctx)
    summary_dir = ctx.output_root / "05_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    inventory = []
    for path in sorted(ctx.output_root.rglob("*")):
        if path.is_file():
            inventory.append({"path": str(path.relative_to(ctx.output_root)).replace("\\", "/"), "bytes": int(path.stat().st_size)})
    write_json(summary_dir / "artifact_inventory.json", inventory)
    table_path = ctx.output_root / "03_analysis" / "primary_sidd_table.csv"
    table_text = ""
    if table_path.exists():
        df = pd.read_csv(table_path)
        try:
            table_text = df.to_markdown(index=False)
        except Exception:
            table_text = df.to_csv(index=False)
    lines = [
        "# SIDD Validation sRGB Evaluation Summary",
        "",
        f"- Output root: `{ctx.output_root}`",
        f"- Source ImageNet experiment root: `{ctx.source_root}`",
        f"- SIDD root: `{ctx.sidd_root}`",
        f"- Resolved config hash: `{ctx.config_hash}`",
        f"- Primary DRUNet sigma: `{int(ctx.cfg['drunet']['primary_sigma_255'])}/255`",
        f"- Max blocks: `{ctx.max_blocks if ctx.max_blocks is not None else 'full'}`",
        f"- Scientific status: `{'non-scientific smoke/debug' if ctx.max_blocks is not None else 'full workload'}`",
        "",
        "Primary table:",
        "",
        table_text,
        "",
        "No training, DDPM scoring, filtering, BN recalibration, finetuning, or SIDD adaptation is performed by this runner.",
    ]
    (summary_dir / "final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_marker(ctx, "summary")


def make_synthetic_sidd_fixture(root: Path, *, scenes: int = 2, blocks: int = 4, size: int = 16, channels: int = 3) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    gt = rng.integers(0, 256, size=(channels, size, size, blocks, scenes), dtype=np.uint8)
    noise = rng.normal(0, 8, size=gt.shape)
    noisy = np.clip(gt.astype(np.float32) + noise, 0, 255).round().astype(np.uint8)
    with h5py.File(root / "ValidationNoisyBlocksSrgb.mat", "w") as handle:
        handle.create_dataset("ValidationNoisyBlocksSrgb", data=noisy)
    with h5py.File(root / "ValidationGtBlocksSrgb.mat", "w") as handle:
        handle.create_dataset("ValidationGtBlocksSrgb", data=gt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot SIDD Validation sRGB evaluation for ImageNet-100 denoisers.")
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/imagenet100_sidd.yaml"))
    parser.add_argument("--stage", choices=["preflight", "smoke", "evaluate", "analyze", "summary", "all"], default=None)
    parser.add_argument("--sidd-root", type=Path, default=None)
    parser.add_argument("--imagenet-experiment-root", type=Path, default=None)
    parser.add_argument("--imagenet-config-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None, help="Final SIDD evaluation run root. If omitted, outputs.root/run_id is used.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tdncnn-batch-size", type=int, default=None)
    parser.add_argument("--drunet-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lpips-enabled", choices=["true", "false"], default=None)
    parser.add_argument("--expected-scenes", type=int, default=None)
    parser.add_argument("--expected-blocks-per-scene", type=int, default=None)
    parser.add_argument("--expected-block-size", type=int, default=None)
    parser.add_argument("--expected-channels", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-blocks", type=int, default=None, help="Smoke/debug only. Marks outputs non-scientific when less than full SIDD.")
    parser.add_argument("--allow-noncanonical-workload", action="store_true", help="For synthetic smoke fixtures and tests only.")
    parser.add_argument("--make-synthetic-fixture", type=Path, default=None, help="Create a tiny synthetic HDF5/MAT SIDD-like fixture and exit.")
    return parser.parse_args()


def make_context(args: argparse.Namespace) -> SiddContext:
    if args.make_synthetic_fixture is not None:
        raise RuntimeError("make_context should not be called for --make-synthetic-fixture")
    cfg = read_yaml(args.config)
    if args.sidd_root is not None:
        cfg["sidd"]["root"] = str(args.sidd_root)
    if args.imagenet_experiment_root is not None:
        cfg["source"]["imagenet_experiment_root"] = str(args.imagenet_experiment_root)
    if args.imagenet_config_path is not None:
        cfg["source"]["imagenet_config_path"] = str(args.imagenet_config_path)
    if args.device is not None:
        cfg["runtime"]["device"] = str(args.device)
    if args.tdncnn_batch_size is not None:
        cfg["runtime"]["tdncnn_batch_size"] = int(args.tdncnn_batch_size)
    if args.drunet_batch_size is not None:
        cfg["runtime"]["drunet_batch_size"] = int(args.drunet_batch_size)
    if args.num_workers is not None:
        cfg["runtime"]["num_workers"] = int(args.num_workers)
    if args.lpips_enabled is not None:
        cfg["runtime"]["lpips_enabled"] = args.lpips_enabled == "true"
    if args.expected_scenes is not None:
        cfg["sidd"]["expected_scenes"] = int(args.expected_scenes)
    if args.expected_blocks_per_scene is not None:
        cfg["sidd"]["expected_blocks_per_scene"] = int(args.expected_blocks_per_scene)
    if args.expected_block_size is not None:
        cfg["sidd"]["expected_block_size"] = int(args.expected_block_size)
    if args.expected_channels is not None:
        cfg["sidd"]["channels"] = int(args.expected_channels)
    if args.allow_noncanonical_workload:
        cfg["sidd"]["strict_expected_workload"] = False
    if args.max_blocks is not None:
        cfg["_non_scientific_max_blocks"] = int(args.max_blocks)
    base_output = resolve_path(cfg["outputs"]["root"])
    run_id = args.run_id or cfg.get("outputs", {}).get("run_id") or time.strftime("sidd_eval_%Y%m%d_%H%M%S")
    output_root = args.output_root.resolve() if args.output_root is not None else (base_output / str(run_id)).resolve()
    source_root = resolve_path(cfg["source"]["imagenet_experiment_root"]).resolve()
    sidd_root = resolve_path(cfg["sidd"]["root"]).resolve()
    fallback_imagenet_config_path = resolve_path(cfg["source"]["imagenet_config_path"]).resolve()
    imagenet_cfg, imagenet_config_path, source_config_sha256, source_config_warning = load_source_imagenet_config(source_root, fallback_imagenet_config_path)
    return SiddContext(
        cfg=cfg,
        config_path=args.config.resolve(),
        output_root=output_root,
        source_root=source_root,
        sidd_root=sidd_root,
        imagenet_cfg=imagenet_cfg,
        imagenet_config_path=imagenet_config_path,
        stage=str(args.stage),
        resume=bool(args.resume),
        dry_run=bool(args.dry_run),
        max_blocks=args.max_blocks,
        source_config_sha256=source_config_sha256,
        source_config_warning=source_config_warning,
    )


def main() -> None:
    args = parse_args()
    if args.make_synthetic_fixture is not None:
        make_synthetic_sidd_fixture(args.make_synthetic_fixture)
        print(f"synthetic_fixture={args.make_synthetic_fixture}")
        return
    if args.stage is None:
        raise SystemExit("--stage is required unless --make-synthetic-fixture is used.")
    ctx = make_context(args)
    if args.stage == "preflight":
        preflight(ctx, strict_load=True)
    elif args.stage == "smoke":
        ctx.max_blocks = 1 if ctx.max_blocks is None else ctx.max_blocks
        evaluate(ctx, smoke=True)
        analyze(ctx)
        summary(ctx)
    elif args.stage == "evaluate":
        evaluate(ctx, smoke=False)
    elif args.stage == "analyze":
        analyze(ctx)
    elif args.stage == "summary":
        summary(ctx)
    elif args.stage == "all":
        evaluate(ctx, smoke=False)
        analyze(ctx)
        summary(ctx)
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
