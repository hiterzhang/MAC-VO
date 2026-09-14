"""Bidirectional image and covariance validation for proximity ICP factors."""

from dataclasses import dataclass
import math

import pypose as pp
import torch
import torch.nn.functional as F

from Module.Optimization.MatchICP import build_edge_from_correspondences
from Module.Optimization.ProximityFactorStore import ProximityFactorRecord
from Module.Optimization.WindowICP import Edge
from Utility.Point import filterPointsInRange


@dataclass(frozen=True)
class ProximityValidationConfig:
    max_forward_backward_error_px: float = 2.0
    min_forward_backward_inliers: int = 30
    min_forward_backward_ratio: float = 0.50
    min_depth_valid_ratio: float = 0.50
    grid_rows: int = 4
    grid_cols: int = 6
    min_occupied_grid_cells: int = 6
    mahalanobis_threshold: float = 3.5
    min_mahalanobis_inliers: int = 30
    min_mahalanobis_inlier_ratio: float = 0.50

    def __post_init__(self):
        if (
            not math.isfinite(self.max_forward_backward_error_px)
            or self.max_forward_backward_error_px < 0
            or not math.isfinite(self.mahalanobis_threshold)
            or self.mahalanobis_threshold <= 0
        ):
            raise ValueError("invalid proximity residual thresholds")
        for name in (
            "min_forward_backward_ratio",
            "min_depth_valid_ratio",
            "min_mahalanobis_inlier_ratio",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be within [0, 1]")
        for name in (
            "min_forward_backward_inliers",
            "grid_rows",
            "grid_cols",
            "min_occupied_grid_cells",
            "min_mahalanobis_inliers",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class MahalanobisFilterResult:
    edge: Edge | None
    uv_a: torch.Tensor
    uv_b: torch.Tensor
    inliers: int
    ratio: float


@dataclass(frozen=True)
class ProximityValidationResult:
    record: ProximityFactorRecord | None
    reason: str | None
    metrics: dict


def sample_map_bilinear(pixel_uv, value_map):
    height, width = value_map.shape[-2:]
    x = pixel_uv[:, 0] * 2 / max(width - 1, 1) - 1
    y = pixel_uv[:, 1] * 2 / max(height - 1, 1) - 1
    grid = torch.stack((x, y), dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        value_map,
        grid.to(device=value_map.device, dtype=value_map.dtype),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[0, :, :, 0].T


def occupied_grid_cells(pixel_uv, width, height, rows, cols):
    if len(pixel_uv) == 0:
        return 0
    column = torch.clamp(
        (pixel_uv[:, 0] / width * cols).long(), 0, cols - 1
    )
    row = torch.clamp(
        (pixel_uv[:, 1] / height * rows).long(), 0, rows - 1
    )
    return int(torch.unique(row * cols + column).numel())


def filter_mahalanobis_inliers(edge, uv_a, uv_b, poses, threshold):
    pose = pp.SE3(torch.as_tensor(poses).double())
    qa = pose[edge.a].Act(edge.points_a)
    qb = pose[edge.b].Act(edge.points_b)
    Ra = pose[edge.a].rotation().matrix()
    Rb = pose[edge.b].rotation().matrix()
    covariance = Ra @ edge.cov_a @ Ra.T + Rb @ edge.cov_b @ Rb.T
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    residual = qa - qb
    white = torch.linalg.solve_triangular(
        torch.linalg.cholesky(covariance),
        residual[..., None],
        upper=False,
    ).squeeze(-1)
    mask = white.norm(dim=-1) <= threshold
    inliers = int(mask.sum())
    ratio = inliers / len(edge.points_a)
    if inliers == 0:
        return MahalanobisFilterResult(None, uv_a[:0], uv_b[:0], 0, 0.0)
    filtered = Edge(
        edge.a,
        edge.b,
        edge.points_a[mask],
        edge.points_b[mask],
        edge.cov_a[mask],
        edge.cov_b[mask],
    )
    return MahalanobisFilterResult(
        filtered,
        uv_a[mask],
        uv_b[mask],
        inliers,
        ratio,
    )


def _failure(reason, metrics):
    return ProximityValidationResult(None, reason, metrics | {"reason": reason})


@torch.inference_mode()
def validate_proximity_match(
    *,
    candidate,
    forward,
    backward,
    source_frame,
    target_frame,
    source_depth,
    target_depth,
    poses,
    frontend,
    keypoint_selector,
    covariance_model,
    num_point,
    edge_width,
    match_cov_default,
    device,
    config,
):
    source_stereo = source_frame.stereo
    target_stereo = target_frame.stereo
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(
            1000003 + candidate.source * 1009 + candidate.target
        )
        uv_a = keypoint_selector.select_point(
            source_stereo, num_point, source_depth, target_depth, forward
        )
    forward_flow = frontend.retrieve_pixels(uv_a, forward.flow).T
    uv_b = uv_a + forward_flow
    inbound = torch.isfinite(uv_b).all(-1) & filterPointsInRange(
        uv_b,
        (edge_width, target_stereo.width - edge_width),
        (edge_width, target_stereo.height - edge_width),
    )
    uv_a = uv_a[inbound]
    uv_b = uv_b[inbound]
    forward_flow = forward_flow[inbound]
    metrics = {
        "selected_points": int(len(inbound)),
        "inbound_points": int(inbound.sum()),
    }
    if len(uv_a) == 0:
        return _failure("insufficient_forward_backward_inliers", metrics)
    backward_flow = sample_map_bilinear(uv_b, backward.flow)
    closure = (forward_flow + backward_flow).norm(dim=-1)
    fb_mask = torch.isfinite(closure) & (
        closure <= config.max_forward_backward_error_px
    )
    fb_inliers = int(fb_mask.sum())
    fb_ratio = fb_inliers / len(uv_a)
    metrics.update({
        "forward_backward_inliers": fb_inliers,
        "forward_backward_ratio": fb_ratio,
        "forward_backward_median_px": (
            None if fb_inliers == 0 else float(closure[fb_mask].median())
        ),
    })
    if (
        fb_inliers < config.min_forward_backward_inliers
        or fb_ratio < config.min_forward_backward_ratio
    ):
        return _failure("insufficient_forward_backward_inliers", metrics)
    uv_a = uv_a[fb_mask]
    uv_b = uv_b[fb_mask]

    depth_a = frontend.retrieve_pixels(uv_a, source_depth.depth).squeeze(0)
    depth_b = frontend.retrieve_pixels(uv_b, target_depth.depth).squeeze(0)
    depth_valid = (
        torch.isfinite(depth_a)
        & torch.isfinite(depth_b)
        & (depth_a > 0)
        & (depth_b > 0)
    )
    depth_ratio = float(depth_valid.double().mean())
    metrics["depth_valid_ratio"] = depth_ratio
    if depth_ratio < config.min_depth_valid_ratio:
        return _failure("insufficient_valid_depth", metrics)
    uv_a = uv_a[depth_valid]
    uv_b = uv_b[depth_valid]
    grid_cells = occupied_grid_cells(
        uv_a.detach().cpu(),
        source_stereo.width,
        source_stereo.height,
        config.grid_rows,
        config.grid_cols,
    )
    metrics["occupied_grid_cells"] = grid_cells
    if grid_cells < config.min_occupied_grid_cells:
        return _failure("insufficient_grid_coverage", metrics)

    built = build_edge_from_correspondences(
        a=candidate.source,
        b=candidate.target,
        uv_a=uv_a,
        uv_b=uv_b,
        stereo_a=source_stereo,
        stereo_b=target_stereo,
        depth_a=source_depth,
        depth_b=target_depth,
        match=forward,
        frontend=frontend,
        covariance_model=covariance_model,
        min_num_point=1,
        match_cov_default=match_cov_default,
        device=device,
    )
    if built.edge is None:
        return _failure(built.reason or "invalid_edge", metrics)
    filtered = filter_mahalanobis_inliers(
        built.edge,
        built.uv_a,
        built.uv_b,
        poses,
        config.mahalanobis_threshold,
    )
    metrics.update({
        "mahalanobis_inliers": filtered.inliers,
        "mahalanobis_inlier_ratio": filtered.ratio,
    })
    if (
        filtered.edge is None
        or filtered.inliers < config.min_mahalanobis_inliers
        or filtered.ratio < config.min_mahalanobis_inlier_ratio
    ):
        return _failure("insufficient_mahalanobis_inliers", metrics)
    final_grid_cells = occupied_grid_cells(
        filtered.uv_a,
        source_stereo.width,
        source_stereo.height,
        config.grid_rows,
        config.grid_cols,
    )
    metrics["final_occupied_grid_cells"] = final_grid_cells
    if final_grid_cells < config.min_occupied_grid_cells:
        return _failure("insufficient_grid_coverage", metrics)
    confidence = max(0.0, min(1.0, 0.25 * (
        candidate.mean_overlap
        + candidate.depth_consistency
        + fb_ratio
        + filtered.ratio
    )))
    record = ProximityFactorRecord(
        edge=filtered.edge,
        score=candidate.score,
        confidence=confidence,
        age=0,
        state="accepted",
        candidate_metrics=candidate.as_dict(),
        validation_metrics=metrics,
    )
    return ProximityValidationResult(record, None, metrics)
