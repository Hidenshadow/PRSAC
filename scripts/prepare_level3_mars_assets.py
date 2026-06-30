"""Prepare Level 3 Mars DTM assets from the MFA/VIKOR research project.

The source project stores three selected Mars HiRISE/DTEED submaps as 500x500
raw elevation arrays.  This script copies those raw arrays into this repo and
exports downsampled, project-compatible ``real_map_layers.npz`` tiles for the
existing real-terrain planning code.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extract_real_dem_tile import (
    derive_planner_layers,
    save_layer_npz,
    save_preview_figures,
    write_readme,
)


DEFAULT_SOURCE_ROOT = Path("<EXTERNAL_MAP_SOURCE_ROOT>")
DEFAULT_RAW_OUTPUT = PROJECT_ROOT / "maps" / "mars_pgda_submaps"
DEFAULT_TILE_OUTPUT = PROJECT_ROOT / "maps" / "real_dem_tiles"
DEFAULT_ROVER_PROFILE = PROJECT_ROOT / "configs" / "rovers" / "mars_rover.json"
DEFAULT_SOURCE_CONFIG_DIR = PROJECT_ROOT / "configs" / "level3_mars_source"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy and convert Mars Level 3 DTM assets.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--raw-output-dir", type=Path, default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--tile-output-root", type=Path, default=DEFAULT_TILE_OUTPUT)
    parser.add_argument("--rover-profile", type=Path, default=DEFAULT_ROVER_PROFILE)
    parser.add_argument("--source-config-output-dir", type=Path, default=DEFAULT_SOURCE_CONFIG_DIR)
    parser.add_argument("--downsample-factor", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def finite_filled(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    if finite.all():
        return array
    if not finite.any():
        raise ValueError("DEM has no finite cells")
    fill_value = float(np.nanmedian(array[finite]))
    out = array.copy()
    out[~finite] = fill_value
    return out


def block_mean_downsample(values: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return finite_filled(values)
    array = finite_filled(values)
    rows = (array.shape[0] // factor) * factor
    cols = (array.shape[1] // factor) * factor
    cropped = array[:rows, :cols]
    return cropped.reshape(rows // factor, factor, cols // factor, factor).mean(axis=(1, 3)).astype(np.float32)


def source_map_path(source_root: Path, raw_path: str) -> Path:
    normalized = str(raw_path).replace("\\", "/")
    return source_root / normalized


def copy_source_files(source_root: Path, raw_output_dir: Path, source_config_output_dir: Path) -> dict[str, Path]:
    source_data_dir = source_root / "data" / "mars_pgda_submaps"
    manifest_path = source_data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Mars manifest not found: {manifest_path}")

    raw_output_dir.mkdir(parents=True, exist_ok=True)
    source_config_output_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    for path in sorted(source_data_dir.glob("*.npy")):
        target = raw_output_dir / path.name
        shutil.copy2(path, target)
        copied[path.name] = target
    manifest_target = raw_output_dir / "manifest.json"
    shutil.copy2(manifest_path, manifest_target)
    copied["manifest.json"] = manifest_target

    for name in (
        "config_batch_mars_three_submaps.yaml",
        "config_batch_mars_three_submaps_relay_constrained.yaml",
        "config_batch_mars_three_submaps_relay_stress.yaml",
    ):
        src = source_root / "configs" / name
        if src.exists():
            dst = source_config_output_dir / name
            shutil.copy2(src, dst)
            copied[name] = dst
    return copied


def rover_metadata(profile: dict[str, Any]) -> tuple[float, float]:
    mobility = profile.get("mobility", {})
    obstacle_slope = float(mobility.get("max_traversable_slope_deg", 30.0))
    p95_filter = max(obstacle_slope - 1.0, 0.0)
    return obstacle_slope, p95_filter


def write_tile(
    map_spec: dict[str, Any],
    source_root: Path,
    raw_output_dir: Path,
    tile_output_root: Path,
    rover_profile: dict[str, Any],
    downsample_factor: int,
    default_resolution: float,
) -> dict[str, Any]:
    name = str(map_spec["name"])
    source_path = source_map_path(source_root, str(map_spec["file_path"]))
    raw_copy_path = raw_output_dir / source_path.name
    dem = np.load(raw_copy_path if raw_copy_path.exists() else source_path)
    downsampled = block_mean_downsample(dem, int(downsample_factor))
    source_resolution = float(map_spec.get("resolution", default_resolution))
    meters_per_pixel = source_resolution * max(int(downsample_factor), 1)
    obstacle_slope_deg, max_slope_p95_deg = rover_metadata(rover_profile)
    layers = derive_planner_layers(
        tile=downsampled,
        meters_per_pixel=meters_per_pixel,
        obstacle_slope_deg=obstacle_slope_deg,
        rover_profile=rover_profile,
    )

    tile_id = f"{name}_level3_tile"
    output_dir = tile_output_root / tile_id
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_meters = float(downsampled.shape[0] * meters_per_pixel)
    finite = np.isfinite(downsampled)
    metadata: dict[str, Any] = {
        "tile_id": tile_id,
        "level": "mars_dtm",
        "scenario": "real_mars_dtm",
        "source_project": str(source_root),
        "source_manifest_name": name,
        "source_name": str(map_spec.get("source_name", "MarsDTEED")),
        "source_dem": str(map_spec.get("source_file_path", "")),
        "source_raw_submap": str(raw_copy_path.relative_to(PROJECT_ROOT)),
        "source_shape": [int(v) for v in dem.shape],
        "source_resolution_m_per_pixel": source_resolution,
        "downsample_factor": int(downsample_factor),
        "row": int(map_spec.get("row_start", 0)),
        "col": int(map_spec.get("col_start", 0)),
        "row_end_exclusive": int(map_spec.get("row_end", dem.shape[0])),
        "col_end_exclusive": int(map_spec.get("col_end", dem.shape[1])),
        "tile_pixels": int(downsampled.shape[0]),
        "tile_meters": tile_meters,
        "requested_meters": tile_meters,
        "meters_per_pixel": meters_per_pixel,
        "stride": 0,
        "seed": 0,
        "rover_profile_path": str(DEFAULT_ROVER_PROFILE.relative_to(PROJECT_ROOT)),
        "rover_name": rover_profile.get("name", "MARS_ROVER"),
        "obstacle_slope_deg": obstacle_slope_deg,
        "max_slope_p95_deg": max_slope_p95_deg,
        "obstacle_cell_ratio": float(np.mean(layers["obstacle_mask"])),
        "finite_ratio": float(finite.mean()),
        "elevation_min_m": float(np.nanmin(downsampled)),
        "elevation_max_m": float(np.nanmax(downsampled)),
        "elevation_mean_m": float(np.nanmean(downsampled)),
        "elevation_std_m": float(np.nanstd(downsampled)),
        "elevation_p05_m": float(np.nanpercentile(downsampled, 5)),
        "elevation_median_m": float(np.nanmedian(downsampled)),
        "elevation_p95_m": float(np.nanpercentile(downsampled, 95)),
        "relief_p95_p05_m": float(np.nanpercentile(downsampled, 95) - np.nanpercentile(downsampled, 5)),
        "slope_mean_deg": float(np.nanmean(layers["slope_degrees"])),
        "slope_p95_deg": float(np.nanpercentile(layers["slope_degrees"], 95)),
        "slope_max_deg": float(np.nanmax(layers["slope_degrees"])),
        "roughness_mean_m": float(np.nanmean(layers["roughness_m"])),
        "roughness_p95_m": float(np.nanpercentile(layers["roughness_m"], 95)),
        "archetype": str(map_spec.get("archetype", "")),
        "planning_free_ratio_source": map_spec.get("planning_free_ratio"),
    }

    np.save(output_dir / "tile_dem.npy", downsampled)
    save_layer_npz(output_dir / "real_map_layers.npz", layers)
    (output_dir / "tile_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "rover_profile_used.json").write_text(
        json.dumps(rover_profile, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    save_preview_figures(output_dir, downsampled, layers, metadata)
    write_readme(output_dir, metadata)
    return {
        "tile_id": tile_id,
        "name": name,
        "archetype": metadata["archetype"],
        "layers": str((output_dir / "real_map_layers.npz").relative_to(PROJECT_ROOT)),
        "metadata": str((output_dir / "tile_metadata.json").relative_to(PROJECT_ROOT)),
        "obstacle_cell_ratio": metadata["obstacle_cell_ratio"],
        "slope_mean_deg": metadata["slope_mean_deg"],
        "slope_p95_deg": metadata["slope_p95_deg"],
        "tile_meters": metadata["tile_meters"],
        "meters_per_pixel": metadata["meters_per_pixel"],
    }


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"source project not found: {source_root}")
    if args.downsample_factor <= 0:
        raise ValueError("--downsample-factor must be positive")

    rover_profile = read_json(args.rover_profile)
    copied = copy_source_files(source_root, args.raw_output_dir, args.source_config_output_dir)
    manifest = read_json(args.raw_output_dir / "manifest.json")
    tiles = [
        write_tile(
            map_spec=dict(map_spec),
            source_root=source_root,
            raw_output_dir=args.raw_output_dir,
            tile_output_root=args.tile_output_root,
            rover_profile=rover_profile,
            downsample_factor=int(args.downsample_factor),
            default_resolution=float(
                (manifest.get("sources") or [{"resolution_m_per_pixel": 2.0103810127427}])[0].get(
                    "resolution_m_per_pixel",
                    2.0103810127427,
                )
            ),
        )
        for map_spec in manifest.get("maps", [])
    ]
    level3_manifest = {
        "level": "mars_dtm",
        "source_project": str(source_root),
        "raw_data_dir": str(args.raw_output_dir.relative_to(PROJECT_ROOT)),
        "source_config_dir": str(args.source_config_output_dir.relative_to(PROJECT_ROOT)),
        "downsample_factor": int(args.downsample_factor),
        "copied_files": {name: str(path.relative_to(PROJECT_ROOT)) for name, path in copied.items()},
        "tiles": tiles,
    }
    out_manifest = args.tile_output_root / "mars_dtm_level3_manifest.json"
    out_manifest.write_text(json.dumps(level3_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Copied Mars raw data: {args.raw_output_dir}")
    print(f"Wrote Level 3 tile manifest: {out_manifest}")
    for tile in tiles:
        print(
            f"{tile['tile_id']}: archetype={tile['archetype']} "
            f"shape=100x100 obstacle={tile['obstacle_cell_ratio']:.4f} "
            f"slope_mean={tile['slope_mean_deg']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

