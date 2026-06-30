"""Prepare 40/60/80 Level 3 Mars DTEED difficulty tiles from the full DEM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extract_real_dem_tile import (  # noqa: E402
    derive_planner_layers,
    save_layer_npz,
    save_preview_figures,
    slope_degrees,
    local_roughness,
    write_readme,
)


DEFAULT_DEM = PROJECT_ROOT / "maps" / "DTEED_076968_1475_076823_1475_A01.tif"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "maps" / "real_dem_tiles"
DEFAULT_ROVER_PROFILE = PROJECT_ROOT / "configs" / "rovers" / "mars_rover.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "maps" / "real_dem_tiles" / "mars_dtm_level3_manifest.json"
SOURCE_RESOLUTION_M_PER_PIXEL = 2.0103810127427
DOWNSAMPLE_FACTOR = 5
NODATA_THRESHOLD = -1.0e20


TILE_SPECS = {
    "easy": {
        "tile_id": "marsdteed_40_tile",
        "source_row": 200,
        "source_col": 1000,
        "final_pixels": 40,
        "archetype": "smooth",
        "task_sampling_mode": "distance",
        "min_distance_ratio": 0.60,
    },
    "medium": {
        "tile_id": "marsdteed_60_tile",
        "source_row": 1350,
        "source_col": 300,
        "final_pixels": 60,
        "archetype": "medium_roughness",
        "task_sampling_mode": "distance",
        "min_distance_ratio": 0.68,
    },
    "hard": {
        "tile_id": "marsdteed_80_tile",
        "source_row": 1000,
        "source_col": 200,
        "final_pixels": 80,
        "archetype": "ridge_boundary",
        "task_sampling_mode": "risk_corridor",
        "min_distance_ratio": 0.75,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Mars DTEED 40/60/80 difficulty tiles.")
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rover-profile", type=Path, default=DEFAULT_ROVER_PROFILE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_dem(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        nodata_text = image.tag_v2.get(42113)
        dem = np.asarray(image, dtype=np.float32)
    nodata_value = None
    if nodata_text is not None:
        try:
            nodata_value = float(nodata_text)
        except (TypeError, ValueError):
            nodata_value = None
    valid = np.isfinite(dem) & (dem > NODATA_THRESHOLD)
    if nodata_value is not None:
        valid &= dem != np.float32(nodata_value)
    out = dem.astype(np.float32, copy=True)
    out[~valid] = np.nan
    return out


def block_nanmean_downsample(values: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return values.astype(np.float32, copy=True)
    rows = (values.shape[0] // factor) * factor
    cols = (values.shape[1] // factor) * factor
    cropped = values[:rows, :cols]
    blocks = cropped.reshape(rows // factor, factor, cols // factor, factor)
    counts = np.sum(np.isfinite(blocks), axis=(1, 3))
    sums = np.nansum(blocks, axis=(1, 3))
    out = np.full((rows // factor, cols // factor), np.nan, dtype=np.float32)
    valid = counts > 0
    out[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    return out


def finite_filled(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    if finite.all():
        return array
    if not finite.any():
        raise ValueError("tile has no finite DEM cells")
    out = array.copy()
    out[~finite] = float(np.nanmedian(array[finite]))
    return out


def rover_slope_threshold(profile: dict[str, Any]) -> float:
    mobility = profile.get("mobility", {})
    if not isinstance(mobility, dict):
        return 30.0
    return float(mobility.get("max_traversable_slope_deg", 30.0))


def robust_stats(tile: np.ndarray, layers: dict[str, np.ndarray]) -> dict[str, float]:
    finite = np.isfinite(tile)
    values = tile[finite]
    slope = np.asarray(layers["slope_degrees"], dtype=np.float32)
    roughness = np.asarray(layers["roughness_m"], dtype=np.float32)
    return {
        "finite_ratio": float(finite.mean()),
        "elevation_min_m": float(np.nanmin(values)),
        "elevation_max_m": float(np.nanmax(values)),
        "elevation_mean_m": float(np.nanmean(values)),
        "elevation_std_m": float(np.nanstd(values)),
        "elevation_p05_m": float(np.nanpercentile(values, 5)),
        "elevation_median_m": float(np.nanmedian(values)),
        "elevation_p95_m": float(np.nanpercentile(values, 95)),
        "relief_p95_p05_m": float(np.nanpercentile(values, 95) - np.nanpercentile(values, 5)),
        "slope_mean_deg": float(np.nanmean(slope)),
        "slope_p95_deg": float(np.nanpercentile(slope, 95)),
        "slope_max_deg": float(np.nanmax(slope)),
        "roughness_mean_m": float(np.nanmean(roughness)),
        "roughness_p95_m": float(np.nanpercentile(roughness, 95)),
    }


def write_tile(
    dem: np.ndarray,
    spec_name: str,
    spec: dict[str, Any],
    args: argparse.Namespace,
    rover_profile: dict[str, Any],
) -> dict[str, Any]:
    final_pixels = int(spec["final_pixels"])
    source_pixels = final_pixels * DOWNSAMPLE_FACTOR
    source_row = int(spec["source_row"])
    source_col = int(spec["source_col"])
    source_tile = dem[source_row : source_row + source_pixels, source_col : source_col + source_pixels]
    if source_tile.shape != (source_pixels, source_pixels):
        raise ValueError(f"source window outside DEM for {spec_name}: {source_tile.shape}")
    if float(np.isfinite(source_tile).mean()) < 1.0:
        raise ValueError(f"selected source window contains NoData: {spec_name}")

    tile = block_nanmean_downsample(source_tile, DOWNSAMPLE_FACTOR)
    tile = finite_filled(tile)
    meters_per_pixel = SOURCE_RESOLUTION_M_PER_PIXEL * DOWNSAMPLE_FACTOR
    obstacle_slope_deg = rover_slope_threshold(rover_profile)
    layers = derive_planner_layers(
        tile=tile,
        meters_per_pixel=meters_per_pixel,
        obstacle_slope_deg=obstacle_slope_deg,
        rover_profile=rover_profile,
    )
    # Force slope/roughness from the final tile into metadata even if layer derivation changes later.
    layers["slope_degrees"] = slope_degrees(tile, meters_per_pixel).astype(np.float32)
    layers["roughness_m"] = local_roughness(tile).astype(np.float32)

    tile_id = str(spec["tile_id"])
    output_dir = args.output_root / tile_id
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "tile_id": tile_id,
        "level": "mars_dtm",
        "difficulty": spec_name,
        "scenario": "real_mars_dtm",
        "source_dem": str(args.dem.relative_to(PROJECT_ROOT)),
        "source_name": "MarsDTEED",
        "source_resolution_m_per_pixel": SOURCE_RESOLUTION_M_PER_PIXEL,
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "row": source_row,
        "col": source_col,
        "row_end_exclusive": source_row + source_pixels,
        "col_end_exclusive": source_col + source_pixels,
        "source_window_pixels": source_pixels,
        "tile_pixels": final_pixels,
        "tile_meters": float(final_pixels * meters_per_pixel),
        "requested_meters": float(final_pixels * meters_per_pixel),
        "meters_per_pixel": float(meters_per_pixel),
        "stride": 0,
        "seed": 0,
        "rover_profile_path": str(args.rover_profile.relative_to(PROJECT_ROOT)),
        "rover_name": rover_profile.get("name", "MARS_ROVER"),
        "obstacle_slope_deg": float(obstacle_slope_deg),
        "max_slope_p95_deg": float(max(obstacle_slope_deg - 1.0, 0.0)),
        "obstacle_cell_ratio": float(np.mean(layers["obstacle_mask"])),
        "archetype": str(spec["archetype"]),
        "task_sampling_mode": str(spec["task_sampling_mode"]),
        "min_distance_ratio": float(spec["min_distance_ratio"]),
        **robust_stats(tile, layers),
    }

    np.save(output_dir / "tile_dem.npy", tile.astype(np.float32))
    save_layer_npz(output_dir / "real_map_layers.npz", layers)
    (output_dir / "tile_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "rover_profile_used.json").write_text(
        json.dumps(rover_profile, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    save_preview_figures(output_dir, tile, layers, metadata)
    write_readme(output_dir, metadata)
    return {
        "difficulty": spec_name,
        "tile_id": tile_id,
        "archetype": metadata["archetype"],
        "layers": str((output_dir / "real_map_layers.npz").relative_to(PROJECT_ROOT)),
        "metadata": str((output_dir / "tile_metadata.json").relative_to(PROJECT_ROOT)),
        "tile_pixels": final_pixels,
        "tile_meters": metadata["tile_meters"],
        "meters_per_pixel": metadata["meters_per_pixel"],
        "source_row": source_row,
        "source_col": source_col,
        "source_window_pixels": source_pixels,
        "slope_mean_deg": metadata["slope_mean_deg"],
        "slope_p95_deg": metadata["slope_p95_deg"],
        "relief_p95_p05_m": metadata["relief_p95_p05_m"],
        "obstacle_cell_ratio": metadata["obstacle_cell_ratio"],
    }


def main() -> int:
    args = parse_args()
    rover_profile = read_json(args.rover_profile)
    dem = load_dem(args.dem)
    tiles = [
        write_tile(dem, name, spec, args, rover_profile)
        for name, spec in TILE_SPECS.items()
    ]
    manifest = {
        "level": "mars_dtm",
        "source_dem": str(args.dem.relative_to(PROJECT_ROOT)),
        "source_resolution_m_per_pixel": SOURCE_RESOLUTION_M_PER_PIXEL,
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "difficulty_grid_sizes": {"easy": 40, "medium": 60, "hard": 80},
        "tiles": tiles,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote Mars DTEED difficulty manifest: {args.manifest}")
    for tile in tiles:
        print(
            f"{tile['tile_id']}: {tile['tile_pixels']}x{tile['tile_pixels']} "
            f"slope_p95={tile['slope_p95_deg']:.3f} "
            f"relief={tile['relief_p95_p05_m']:.3f} "
            f"obstacle={tile['obstacle_cell_ratio']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
