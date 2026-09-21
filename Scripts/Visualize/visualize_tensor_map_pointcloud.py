#!/usr/bin/env python3
"""Export a tensor-map point cloud and build an offline 3D viewer."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def load_point_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = ("points//pos_Tw", "points//color")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Missing tensor-map keys: {', '.join(missing)}")
        xyz = np.asarray(data[required[0]], dtype=np.float32)
        rgb = np.asarray(data[required[1]])

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected positions with shape (N, 3), got {xyz.shape}")
    if rgb.ndim != 2 or rgb.shape != xyz.shape:
        raise ValueError(f"Expected colors with shape {xyz.shape}, got {rgb.shape}")
    return xyz, np.clip(rgb, 0, 255).astype(np.uint8)


def filter_point_cloud(
    xyz: np.ndarray,
    rgb: np.ndarray,
    keep_quantile: float = 0.995,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    finite_mask = np.isfinite(xyz).all(axis=1)
    finite_xyz, finite_rgb = xyz[finite_mask], rgb[finite_mask]
    if len(finite_xyz) == 0:
        raise ValueError("No finite points remain")

    center = np.median(finite_xyz, axis=0)
    distance = np.linalg.norm(finite_xyz - center, axis=1)
    threshold = np.quantile(distance, keep_quantile)
    keep = distance <= threshold
    filtered_xyz, filtered_rgb = finite_xyz[keep], finite_rgb[keep]
    if len(filtered_xyz) == 0:
        raise ValueError("No points remain after outlier filtering")
    stats = {"input": len(xyz), "finite": len(finite_xyz), "kept": len(filtered_xyz)}
    return filtered_xyz, filtered_rgb, stats


def write_ascii_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(
        (
            "ply",
            "format ascii 1.0",
            f"element vertex {len(xyz)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        )
    ) + "\n"
    rows = np.column_stack((xyz, rgb))
    with path.open("w", encoding="ascii") as stream:
        stream.write(header)
        np.savetxt(stream, rows, fmt="%.6f %.6f %.6f %d %d %d")


def sample_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(xyz) <= max_points:
        return xyz, rgb
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(xyz), max_points, replace=False)
    return xyz[indices], rgb[indices]


def build_viewer(
    xyz: np.ndarray,
    rgb: np.ndarray,
    full_count: int,
) -> go.Figure:
    colors = [f"rgb({r},{g},{b})" for r, g, b in rgb]
    figure = go.Figure(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers",
            marker={"size": 1.35, "color": colors, "opacity": 0.9},
            hoverinfo="skip",
            name="V203 points",
        )
    )
    figure.update_layout(
        title={
            "text": f"V203 Point Cloud · viewer {len(xyz):,} / full {full_count:,} points",
            "x": 0.02,
            "y": 0.97,
            "font": {"size": 20, "color": "#d8f3f0"},
        },
        paper_bgcolor="#071316",
        plot_bgcolor="#071316",
        font={"color": "#b9d7d4"},
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        showlegend=False,
        scene={
            "aspectmode": "data",
            "bgcolor": "#071316",
            "xaxis": {"title": "X", "gridcolor": "#1d3b3d"},
            "yaxis": {"title": "Y", "gridcolor": "#1d3b3d"},
            "zaxis": {"title": "Z", "gridcolor": "#1d3b3d"},
        },
        uirevision="v203-point-cloud",
    )
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--ply-output", type=Path, required=True)
    parser.add_argument("--html-output", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-quantile", type=float, default=0.995)
    args = parser.parse_args()
    if args.max_points < 0:
        parser.error("--max-points must be non-negative")
    if not 0 < args.keep_quantile <= 1:
        parser.error("--keep-quantile must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    xyz, rgb = load_point_cloud(args.input)
    xyz, rgb, stats = filter_point_cloud(xyz, rgb, args.keep_quantile)
    viewer_xyz, viewer_rgb = sample_points(xyz, rgb, args.max_points, args.seed)

    write_ascii_ply(args.ply_output, xyz, rgb)
    args.html_output.parent.mkdir(parents=True, exist_ok=True)
    build_viewer(viewer_xyz, viewer_rgb, len(xyz)).write_html(
        args.html_output,
        include_plotlyjs=True,
        full_html=True,
    )

    print(
        f"Input: {stats['input']:,}; finite: {stats['finite']:,}; "
        f"kept: {stats['kept']:,}; viewer: {len(viewer_xyz):,}"
    )
    print(f"PLY: {args.ply_output}")
    print(f"HTML: {args.html_output}")


if __name__ == "__main__":
    main()
