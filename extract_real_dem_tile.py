"""Extract a 200 m x 200 m real DEM tile for rover-map experiments.

The current project mainly uses synthetic normalized cost-map layers. This
script keeps the DEM extraction separate from the training/evaluation code: it
selects a finite, moderately structured tile from a real GeoTIFF/float TIFF and
exports both the raw elevation patch and simple derived layers that can later be
connected to the planner environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter


DEFAULT_DEM_PATH = Path("maps") / "NPD_final_adj_5mpp_surf.tif"
DEFAULT_OUTPUT_DIR = Path("maps") / "real_dem_tiles" / "npd_plain_pgda_200m_tile"
DEFAULT_OBSTACLE_SLOPE_DEG = 25.0
DEFAULT_MAX_SLOPE_P95_DEG = 22.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a 200 m x 200 m tile from a lunar DEM TIFF.",
    )
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--meters", type=float, default=200.0)
    parser.add_argument("--meters-per-pixel", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument(
        "--rover-profile",
        type=Path,
        default=None,
        help="Optional rover parameter JSON, for example configs/rovers/viper.json.",
    )
    parser.add_argument("--min-finite-ratio", type=float, default=1.0)
    parser.add_argument("--min-relief-m", type=float, default=2.0)
    parser.add_argument("--max-slope-p95-deg", type=float, default=None)
    parser.add_argument("--obstacle-slope-deg", type=float, default=None)
    parser.add_argument("--max-obstacle-ratio", type=float, default=0.25)
    parser.add_argument("--row", type=int, default=None)
    parser.add_argument("--col", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory.",
    )
    return parser.parse_args()


def load_dem(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"DEM file not found: {path}")
    with Image.open(path) as image:
        dem = np.asarray(image, dtype=np.float32)
    if dem.ndim != 2:
        raise ValueError(f"Expected a single-band DEM, got shape {dem.shape}")
    return dem


def load_rover_profile(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Rover profile not found: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        profile = json.load(handle)
    if not isinstance(profile, dict):
        raise ValueError(f"Rover profile must be a JSON object: {path}")
    return profile


def rover_nested_value(
    profile: dict[str, Any] | None,
    section: str,
    name: str,
    default: float,
) -> float:
    if not profile:
        return float(default)
    section_data = profile.get(section, {})
    if not isinstance(section_data, dict):
        return float(default)
    value = section_data.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def resolve_slope_thresholds(
    obstacle_slope_deg: float | None,
    max_slope_p95_deg: float | None,
    rover_profile: dict[str, Any] | None,
) -> tuple[float, float]:
    rover_max_slope = rover_nested_value(
        rover_profile,
        "mobility",
        "max_traversable_slope_deg",
        DEFAULT_OBSTACLE_SLOPE_DEG,
    )
    resolved_obstacle = (
        float(obstacle_slope_deg)
        if obstacle_slope_deg is not None
        else float(rover_max_slope if rover_profile else DEFAULT_OBSTACLE_SLOPE_DEG)
    )
    resolved_p95 = (
        float(max_slope_p95_deg)
        if max_slope_p95_deg is not None
        else float(max(resolved_obstacle - 1.0, 0.0) if rover_profile else DEFAULT_MAX_SLOPE_P95_DEG)
    )
    return resolved_obstacle, resolved_p95


def normalize01(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    lo = float(np.nanmin(array[finite]))
    hi = float(np.nanmax(array[finite]))
    span = hi - lo
    if span < 1e-8:
        out = np.zeros_like(array, dtype=np.float32)
    else:
        out = (array - lo) / span
    out[~finite] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def slope_degrees(tile: np.ndarray, meters_per_pixel: float) -> np.ndarray:
    dz_drow, dz_dcol = np.gradient(tile.astype(np.float32), meters_per_pixel, meters_per_pixel)
    grade = np.hypot(dz_drow, dz_dcol)
    return np.degrees(np.arctan(grade)).astype(np.float32)


def local_roughness(tile: np.ndarray, window: int = 5) -> np.ndarray:
    tile = tile.astype(np.float32)
    mean = uniform_filter(tile, size=window, mode="reflect")
    mean_sq = uniform_filter(tile * tile, size=window, mode="reflect")
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0)).astype(np.float32)


def tile_stats(
    tile: np.ndarray,
    row: int,
    col: int,
    meters_per_pixel: float,
    obstacle_slope_deg: float,
) -> dict[str, Any]:
    finite = np.isfinite(tile)
    finite_ratio = float(finite.mean())
    if not finite.any():
        return {
            "row": row,
            "col": col,
            "finite_ratio": finite_ratio,
            "score": -math.inf,
        }

    values = tile[finite]
    slope = slope_degrees(tile, meters_per_pixel)
    roughness = local_roughness(tile)
    p05, p50, p95 = np.percentile(values, [5, 50, 95])
    relief_p95_p05 = float(p95 - p05)
    slope_mean = float(np.nanmean(slope))
    slope_p95 = float(np.nanpercentile(slope, 95))
    roughness_mean = float(np.nanmean(roughness))
    obstacle_cell_ratio = float(np.mean((slope > float(obstacle_slope_deg)) | ~np.isfinite(tile)))

    # Prefer non-flat, traversable-looking local terrain. The slope penalty
    # avoids selecting cliff-like patches that would become mostly no-go cells.
    score = relief_p95_p05 + 2.0 * roughness_mean + 0.5 * slope_mean
    if slope_p95 > 35.0:
        score -= 5.0 * (slope_p95 - 35.0)
    if obstacle_cell_ratio > 0.10:
        score -= 40.0 * (obstacle_cell_ratio - 0.10)

    return {
        "row": row,
        "col": col,
        "finite_ratio": finite_ratio,
        "score": float(score),
        "elevation_min_m": float(np.nanmin(values)),
        "elevation_max_m": float(np.nanmax(values)),
        "elevation_mean_m": float(np.nanmean(values)),
        "elevation_std_m": float(np.nanstd(values)),
        "elevation_p05_m": float(p05),
        "elevation_median_m": float(p50),
        "elevation_p95_m": float(p95),
        "relief_p95_p05_m": relief_p95_p05,
        "slope_mean_deg": slope_mean,
        "slope_p95_deg": slope_p95,
        "slope_max_deg": float(np.nanmax(slope)),
        "roughness_mean_m": roughness_mean,
        "roughness_p95_m": float(np.nanpercentile(roughness, 95)),
        "obstacle_cell_ratio": obstacle_cell_ratio,
    }


def iter_tile_origins(height: int, width: int, tile_pixels: int, stride: int):
    rows = list(range(0, max(height - tile_pixels + 1, 1), stride))
    cols = list(range(0, max(width - tile_pixels + 1, 1), stride))
    last_row = height - tile_pixels
    last_col = width - tile_pixels
    if rows[-1] != last_row:
        rows.append(last_row)
    if cols[-1] != last_col:
        cols.append(last_col)
    for row in rows:
        for col in cols:
            yield row, col


def find_candidate_tiles(
    dem: np.ndarray,
    tile_pixels: int,
    stride: int,
    meters_per_pixel: float,
    min_finite_ratio: float,
    min_relief_m: float,
    max_slope_p95_deg: float,
    obstacle_slope_deg: float,
    max_obstacle_ratio: float,
    top_k: int,
) -> list[dict[str, Any]]:
    height, width = dem.shape
    candidates: list[dict[str, Any]] = []
    for row, col in iter_tile_origins(height, width, tile_pixels, stride):
        tile = dem[row : row + tile_pixels, col : col + tile_pixels]
        stats = tile_stats(tile, row, col, meters_per_pixel, obstacle_slope_deg)
        if stats["finite_ratio"] < min_finite_ratio:
            continue
        if stats.get("relief_p95_p05_m", 0.0) < min_relief_m:
            continue
        if stats.get("slope_p95_deg", math.inf) > max_slope_p95_deg:
            continue
        if stats.get("obstacle_cell_ratio", 1.0) > max_obstacle_ratio:
            continue
        candidates.append(stats)

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[: max(int(top_k), 1)]


def derive_planner_layers(
    tile: np.ndarray,
    meters_per_pixel: float,
    obstacle_slope_deg: float,
    rover_profile: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    slope = slope_degrees(tile, meters_per_pixel)
    roughness = local_roughness(tile)
    height_norm = normalize01(tile)
    slope_norm = normalize01(slope)
    roughness_norm = normalize01(roughness)

    distance = np.ones_like(tile, dtype=np.float32)
    energy = normalize01(0.68 * slope_norm + 0.32 * roughness_norm)
    hazard = normalize01(0.60 * slope_norm + 0.30 * roughness_norm + 0.10 * height_norm)
    communication = normalize01(0.65 * height_norm + 0.35 * roughness_norm)

    # Approximate illumination from a fixed low sun angle. This is a diagnostic
    # layer only; proper sun geometry can be added once we wire real DEM maps
    # into the environment.
    dz_drow, dz_dcol = np.gradient(tile.astype(np.float32), meters_per_pixel, meters_per_pixel)
    normal_x = -dz_dcol
    normal_y = -dz_drow
    normal_z = np.ones_like(tile, dtype=np.float32)
    normal_norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2) + 1e-8
    normal_x /= normal_norm
    normal_y /= normal_norm
    normal_z /= normal_norm
    sun_elevation = np.deg2rad(12.0)
    sun_azimuth = np.deg2rad(45.0)
    sun = np.array(
        [
            np.cos(sun_elevation) * np.cos(sun_azimuth),
            np.cos(sun_elevation) * np.sin(sun_azimuth),
            np.sin(sun_elevation),
        ],
        dtype=np.float32,
    )
    illumination_quality = np.clip(
        normal_x * sun[0] + normal_y * sun[1] + normal_z * sun[2],
        0.0,
        None,
    )
    illumination = 1.0 - normalize01(illumination_quality)

    obstacle_mask = (slope > float(obstacle_slope_deg)) | ~np.isfinite(tile)
    extra_layers: dict[str, np.ndarray] = {}
    if rover_profile:
        max_slope = max(
            rover_nested_value(
                rover_profile,
                "mobility",
                "max_traversable_slope_deg",
                obstacle_slope_deg,
            ),
            1e-6,
        )
        mass_kg = rover_nested_value(rover_profile, "body", "rolling_mass_kg", 450.0)
        gravity = rover_nested_value(rover_profile, "simulation_assumptions", "gravity_mps2", math.nan)
        if not np.isfinite(gravity):
            gravity = rover_nested_value(rover_profile, "simulation_assumptions", "mars_gravity_mps2", math.nan)
        if not np.isfinite(gravity):
            gravity = rover_nested_value(rover_profile, "simulation_assumptions", "lunar_gravity_mps2", 1.62)
        rolling_resistance = rover_nested_value(
            rover_profile,
            "simulation_assumptions",
            "rolling_resistance_coeff",
            0.12,
        )
        nominal_speed = rover_nested_value(rover_profile, "mobility", "nominal_drive_speed_mps", 0.10)
        top_speed = rover_nested_value(rover_profile, "mobility", "top_speed_mps", max(nominal_speed, 1e-3))
        peak_power_w = rover_nested_value(rover_profile, "power", "peak_power_w", 450.0)

        slope_ratio = np.clip(slope / float(max_slope), 0.0, None).astype(np.float32)
        slope_margin = (1.0 - slope_ratio).astype(np.float32)
        speed_scale = np.clip(1.0 - 0.70 * np.minimum(slope_ratio, 1.5) ** 2, 0.15, 1.0)
        rover_speed_mps = np.clip(float(nominal_speed) * speed_scale, 0.0, float(top_speed)).astype(np.float32)
        rover_speed_mps = np.where(obstacle_mask, 0.0, rover_speed_mps).astype(np.float32)
        cell_time_s = np.where(
            rover_speed_mps > 1e-8,
            float(meters_per_pixel) / np.maximum(rover_speed_mps, 1e-8),
            np.inf,
        ).astype(np.float32)

        slope_rad = np.deg2rad(slope)
        traction_force_n = float(mass_kg) * float(gravity) * (
            float(rolling_resistance) + np.sin(slope_rad)
        )
        tractive_energy_j_per_m = np.maximum(traction_force_n, 0.0).astype(np.float32)
        tractive_energy_j_per_cell = (tractive_energy_j_per_m * float(meters_per_pixel)).astype(np.float32)
        power_limited_time_cost = np.where(
            np.isfinite(cell_time_s),
            float(peak_power_w) * cell_time_s,
            np.inf,
        ).astype(np.float32)

        finite_energy = np.where(np.isfinite(tractive_energy_j_per_cell), tractive_energy_j_per_cell, np.nan)
        finite_time_energy = np.where(np.isfinite(power_limited_time_cost), power_limited_time_cost, np.nan)
        energy = normalize01(0.65 * finite_energy + 0.35 * finite_time_energy)
        hazard = normalize01(
            0.70 * np.clip(slope_ratio, 0.0, 1.5)
            + 0.20 * roughness_norm
            + 0.10 * np.clip(-slope_margin, 0.0, 1.0)
        )

        extra_layers = {
            "rover_max_slope_deg": np.full_like(tile, float(max_slope), dtype=np.float32),
            "viper_slope_ratio": slope_ratio.astype(np.float32),
            "viper_slope_margin": slope_margin.astype(np.float32),
            "viper_speed_mps": rover_speed_mps.astype(np.float32),
            "viper_cell_traverse_time_s": cell_time_s,
            "viper_tractive_energy_j_per_m": tractive_energy_j_per_m,
            "viper_tractive_energy_j_per_cell": tractive_energy_j_per_cell,
            "viper_power_time_energy_proxy_j": power_limited_time_cost,
        }

    uncertainty_base = normalize01(0.45 * roughness_norm + 0.45 * slope_norm + 0.10 * np.abs(height_norm - 0.5))
    layers = {
        "height_map": tile.astype(np.float32),
        "height_norm": height_norm,
        "slope_degrees": slope.astype(np.float32),
        "slope_layer": slope_norm.astype(np.float32),
        "roughness_m": roughness.astype(np.float32),
        "roughness_layer": roughness_norm.astype(np.float32),
        "obstacle_mask": obstacle_mask.astype(bool),
        "layer_distance": distance,
        "layer_energy": energy.astype(np.float32),
        "layer_hazard": hazard.astype(np.float32),
        "layer_communication": communication.astype(np.float32),
        "layer_illumination": illumination.astype(np.float32),
        "uncertainty_distance": np.full_like(tile, 0.05, dtype=np.float32),
        "uncertainty_energy": uncertainty_base.astype(np.float32),
        "uncertainty_hazard": normalize01(0.70 * slope_norm + 0.30 * roughness_norm),
        "uncertainty_communication": normalize01(0.50 * communication + 0.50 * roughness_norm),
        "uncertainty_illumination": normalize01(0.50 * illumination + 0.50 * slope_norm),
    }
    layers.update(extra_layers)
    return layers


def save_candidates_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    if not candidates:
        return
    fieldnames = list(candidates[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)


def save_layer_npz(path: Path, layers: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **layers)


def save_preview_figures(
    output_dir: Path,
    tile: np.ndarray,
    layers: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    extent = [0.0, metadata["tile_meters"], metadata["tile_meters"], 0.0]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    plots = [
        ("DEM elevation (m)", tile, "terrain"),
        ("Slope (deg)", layers["slope_degrees"], "magma"),
        ("Roughness (m)", layers["roughness_m"], "viridis"),
        ("Energy cost", layers["layer_energy"], "inferno"),
        ("Hazard cost", layers["layer_hazard"], "Reds"),
        ("Obstacle mask", layers["obstacle_mask"].astype(float), "gray_r"),
    ]
    for ax, (title, data, cmap) in zip(axes.ravel(), plots):
        im = ax.imshow(data, cmap=cmap, extent=extent)
        ax.set_title(title)
        ax.set_xlabel("meters")
        ax.set_ylabel("meters")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"DEM tile row={metadata['row']} col={metadata['col']} "
        f"({metadata['tile_meters']:.0f} m x {metadata['tile_meters']:.0f} m)",
    )
    fig.savefig(output_dir / "tile_layers_preview.png", dpi=180)
    plt.close(fig)

    for name, data, cmap in (
        ("tile_dem_preview.png", tile, "terrain"),
        ("tile_slope_degrees.png", layers["slope_degrees"], "magma"),
        ("tile_hazard_layer.png", layers["layer_hazard"], "Reds"),
        ("tile_obstacle_mask.png", layers["obstacle_mask"].astype(float), "gray_r"),
    ):
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        im = ax.imshow(data, cmap=cmap, extent=extent)
        ax.set_title(name.replace("_", " ").replace(".png", ""))
        ax.set_xlabel("meters")
        ax.set_ylabel("meters")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(output_dir / name, dpi=180)
        plt.close(fig)

    if "viper_speed_mps" in layers:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        mobility_plots = [
            ("VIPER slope ratio", layers["viper_slope_ratio"], "magma"),
            ("VIPER slope margin", layers["viper_slope_margin"], "coolwarm"),
            ("VIPER speed (m/s)", layers["viper_speed_mps"], "viridis"),
            ("Cell traverse time (s)", np.where(np.isfinite(layers["viper_cell_traverse_time_s"]), layers["viper_cell_traverse_time_s"], np.nan), "plasma"),
            ("Energy proxy (J/cell)", np.where(np.isfinite(layers["viper_tractive_energy_j_per_cell"]), layers["viper_tractive_energy_j_per_cell"], np.nan), "inferno"),
            ("Obstacle mask", layers["obstacle_mask"].astype(float), "gray_r"),
        ]
        for ax, (title, data, cmap) in zip(axes.ravel(), mobility_plots):
            im = ax.imshow(data, cmap=cmap, extent=extent)
            ax.set_title(title)
            ax.set_xlabel("meters")
            ax.set_ylabel("meters")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle("VIPER mobility layers on selected DEM tile")
        fig.savefig(output_dir / "tile_viper_mobility_preview.png", dpi=180)
        plt.close(fig)


def write_readme(output_dir: Path, metadata: dict[str, Any]) -> None:
    rover_lines: list[str] = []
    if metadata.get("rover_name"):
        rover_lines = [
            "",
            "Rover profile:",
            "",
            f"- Rover: `{metadata['rover_name']}`",
            f"- Profile JSON: `{metadata.get('rover_profile_path', '')}`",
            f"- Max traversable slope used as obstacle threshold: `{metadata['obstacle_slope_deg']:.3f}` deg",
            f"- P95 slope filter: `{metadata['max_slope_p95_deg']:.3f}` deg",
            "- `real_map_layers.npz` includes `viper_*` mobility layers.",
            "- The energy layer is a proxy derived from public mass/speed/slope parameters, not a calibrated wheel-regolith dynamics model.",
        ]
    readme = output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Real DEM Tile",
                "",
                f"- Source DEM: `{metadata['source_dem']}`",
                f"- Pixel origin: row `{metadata['row']}`, col `{metadata['col']}`",
                f"- Tile size: `{metadata['tile_pixels']} x {metadata['tile_pixels']}` pixels",
                f"- Physical size: `{metadata['tile_meters']:.2f} m x {metadata['tile_meters']:.2f} m`",
                f"- Meters per pixel: `{metadata['meters_per_pixel']}`",
                f"- Elevation range: `{metadata['elevation_min_m']:.3f}` to `{metadata['elevation_max_m']:.3f}` m",
                f"- Robust relief p95-p05: `{metadata['relief_p95_p05_m']:.3f}` m",
                f"- Mean slope: `{metadata['slope_mean_deg']:.3f}` deg",
                f"- P95 slope: `{metadata['slope_p95_deg']:.3f}` deg",
                f"- Obstacle cell ratio at configured threshold: `{metadata['obstacle_cell_ratio']:.4f}`",
                *rover_lines,
                "",
                "Files:",
                "",
                "- `tile_dem.npy`: raw elevation tile in meters.",
                "- `real_map_layers.npz`: derived normalized layers with project-style names.",
                "- `rover_profile_used.json`: rover parameter snapshot, if `--rover-profile` was provided.",
                "- `tile_metadata.json`: extraction and statistics metadata.",
                "- `candidate_tiles.csv`: top candidate windows scanned from the DEM.",
                "- `tile_layers_preview.png`: quick visual check.",
                "",
                "This tile is not automatically wired into PPO training yet; it is a reproducible real-map input candidate.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    rover_profile = load_rover_profile(args.rover_profile)
    obstacle_slope_deg, max_slope_p95_deg = resolve_slope_thresholds(
        obstacle_slope_deg=args.obstacle_slope_deg,
        max_slope_p95_deg=args.max_slope_p95_deg,
        rover_profile=rover_profile,
    )
    if args.meters_per_pixel <= 0:
        raise ValueError("--meters-per-pixel must be positive")
    tile_pixels = int(round(float(args.meters) / float(args.meters_per_pixel)))
    if tile_pixels <= 1:
        raise ValueError("requested tile is too small")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use --overwrite or choose another --output-dir."
        )

    dem = load_dem(args.dem)
    height, width = dem.shape
    if tile_pixels > height or tile_pixels > width:
        raise ValueError(
            f"Requested {tile_pixels} px tile does not fit DEM shape {dem.shape}"
        )

    if args.row is not None or args.col is not None:
        if args.row is None or args.col is None:
            raise ValueError("--row and --col must be provided together")
        row = int(args.row)
        col = int(args.col)
        if row < 0 or col < 0 or row + tile_pixels > height or col + tile_pixels > width:
            raise ValueError("manual tile origin is outside the DEM")
        candidates = [
            tile_stats(
                dem[row : row + tile_pixels, col : col + tile_pixels],
                row,
                col,
                args.meters_per_pixel,
                obstacle_slope_deg,
            )
        ]
    else:
        candidates = find_candidate_tiles(
            dem=dem,
            tile_pixels=tile_pixels,
            stride=max(int(args.stride), 1),
            meters_per_pixel=float(args.meters_per_pixel),
            min_finite_ratio=float(args.min_finite_ratio),
            min_relief_m=float(args.min_relief_m),
            max_slope_p95_deg=float(max_slope_p95_deg),
            obstacle_slope_deg=float(obstacle_slope_deg),
            max_obstacle_ratio=float(args.max_obstacle_ratio),
            top_k=int(args.top_k),
        )
        if not candidates:
            raise RuntimeError(
                "No candidate tile passed the filters. Try lowering --min-relief-m "
                "or increasing --max-slope-p95-deg."
            )

    best = candidates[0]
    row = int(best["row"])
    col = int(best["col"])
    tile = dem[row : row + tile_pixels, col : col + tile_pixels].astype(np.float32)
    layers = derive_planner_layers(
        tile=tile,
        meters_per_pixel=float(args.meters_per_pixel),
        obstacle_slope_deg=float(obstacle_slope_deg),
        rover_profile=rover_profile,
    )

    metadata = {
        "source_dem": str(args.dem),
        "dem_shape": [int(height), int(width)],
        "row": row,
        "col": col,
        "row_end_exclusive": row + tile_pixels,
        "col_end_exclusive": col + tile_pixels,
        "tile_pixels": tile_pixels,
        "tile_meters": float(tile_pixels * args.meters_per_pixel),
        "requested_meters": float(args.meters),
        "meters_per_pixel": float(args.meters_per_pixel),
        "stride": int(args.stride),
        "seed": int(args.seed),
        "rover_profile_path": str(args.rover_profile) if args.rover_profile else None,
        "rover_name": rover_profile.get("name") if rover_profile else None,
        "obstacle_slope_deg": float(obstacle_slope_deg),
        "max_slope_p95_deg": float(max_slope_p95_deg),
        "obstacle_cell_ratio": float(np.mean(layers["obstacle_mask"])),
        **best,
    }
    metadata["obstacle_cell_ratio"] = float(np.mean(layers["obstacle_mask"]))

    np.save(output_dir / "tile_dem.npy", tile)
    save_layer_npz(output_dir / "real_map_layers.npz", layers)
    save_candidates_csv(output_dir / "candidate_tiles.csv", candidates)
    if rover_profile:
        (output_dir / "rover_profile_used.json").write_text(
            json.dumps(rover_profile, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    (output_dir / "tile_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    save_preview_figures(output_dir, tile, layers, metadata)
    write_readme(output_dir, metadata)

    print(f"DEM: {args.dem}")
    if rover_profile:
        print(f"Rover profile: {rover_profile.get('name', args.rover_profile)} ({args.rover_profile})")
        print(f"Resolved VIPER slope threshold: obstacle={obstacle_slope_deg:.3f} deg, p95 filter={max_slope_p95_deg:.3f} deg")
    print(f"DEM shape: {height} x {width}")
    print(f"Selected tile origin: row={row}, col={col}")
    print(f"Tile size: {tile_pixels} px = {metadata['tile_meters']:.1f} m")
    print(f"Relief p95-p05: {metadata['relief_p95_p05_m']:.3f} m")
    print(f"Slope mean/p95/max: {metadata['slope_mean_deg']:.3f} / {metadata['slope_p95_deg']:.3f} / {metadata['slope_max_deg']:.3f} deg")
    print(f"Obstacle cell ratio: {metadata['obstacle_cell_ratio']:.4f}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
