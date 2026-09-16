"""Sparse fixed-scale SE(3) geometry for loop verification."""

from dataclasses import dataclass

import pypose as pp
import torch


@dataclass(frozen=True)
class SE3RansacResult:
    measurement: torch.Tensor | None
    mask: torch.Tensor
    inliers: int
    ratio: float
    median_mahalanobis: float | None
    p90_mahalanobis: float | None
    singular_values: torch.Tensor | None
    reason: str | None


def _failure(count, reason, singular_values=None):
    return SE3RansacResult(
        None,
        torch.zeros(count, dtype=torch.bool),
        0,
        0.0,
        None,
        None,
        singular_values,
        reason,
    )


def _geometry_singular_values(points):
    centered = points - points.mean(dim=0)
    return torch.linalg.svdvals(centered)


def _geometry_valid(points_a, points_b, minimum_ratio):
    singular_a = _geometry_singular_values(points_a)
    singular_b = _geometry_singular_values(points_b)
    singular = torch.minimum(singular_a, singular_b)
    denominator = singular[0].clamp_min(1e-15)
    return bool(singular[1] / denominator >= minimum_ratio), singular


def _weighted_kabsch(points_a, points_b, covariance):
    weights = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
    weights = weights.clamp_min(1e-15).reciprocal()
    weights = weights / weights.sum()
    center_a = (weights[:, None] * points_a).sum(dim=0)
    center_b = (weights[:, None] * points_b).sum(dim=0)
    a = points_a - center_a
    b = points_b - center_b
    cross = (weights[:, None] * b).T @ a
    U, _, Vh = torch.linalg.svd(cross)
    rotation = Vh.T @ U.T
    if torch.linalg.det(rotation) < 0:
        Vh = Vh.clone()
        Vh[-1] *= -1
        rotation = Vh.T @ U.T
    translation = center_a - rotation @ center_b
    matrix = torch.eye(4, dtype=torch.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return pp.from_matrix(matrix, pp.SE3_type).tensor()


def _mahalanobis(measurement, points_a, points_b, cov_a, cov_b):
    transform = pp.SE3(measurement)
    rotation = transform.rotation().matrix()
    residual = points_a - transform.Act(points_b)
    covariance = cov_a + rotation @ cov_b @ rotation.T
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    cholesky, info = torch.linalg.cholesky_ex(covariance)
    if int(info.max()) != 0:
        raise ValueError("invalid combined covariance")
    white = torch.linalg.solve_triangular(
        cholesky, residual[..., None], upper=False
    ).squeeze(-1)
    return white.norm(dim=-1)


@torch.no_grad()
def estimate_se3_ransac(
    points_a,
    points_b,
    cov_a,
    cov_b,
    *,
    iterations,
    mahalanobis_threshold,
    min_geometry_ratio,
    seed,
):
    points_a = torch.as_tensor(points_a).detach().cpu().double()
    points_b = torch.as_tensor(points_b).detach().cpu().double()
    cov_a = torch.as_tensor(cov_a).detach().cpu().double()
    cov_b = torch.as_tensor(cov_b).detach().cpu().double()
    count = len(points_a)
    if (
        points_a.shape != (count, 3)
        or points_b.shape != (count, 3)
        or cov_a.shape != (count, 3, 3)
        or cov_b.shape != (count, 3, 3)
    ):
        raise ValueError("sparse SE3 inputs have incompatible shapes")
    if count < 3:
        return _failure(count, "too_few_correspondences")
    if not all(torch.isfinite(value).all() for value in (
        points_a, points_b, cov_a, cov_b
    )):
        return _failure(count, "nonfinite_input")
    if (
        int(torch.linalg.cholesky_ex(cov_a).info.max()) != 0
        or int(torch.linalg.cholesky_ex(cov_b).info.max()) != 0
    ):
        return _failure(count, "invalid_covariance")
    geometry_valid, singular = _geometry_valid(
        points_a, points_b, min_geometry_ratio
    )
    if not geometry_valid:
        return _failure(count, "degenerate_geometry", singular)

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    best_measurement = None
    best_mask = None
    best_count = -1
    best_median = float("inf")
    valid_samples = 0
    covariance_sum = cov_a + cov_b
    for _ in range(int(iterations)):
        sample = torch.randperm(count, generator=generator)[:3]
        sample_valid, _ = _geometry_valid(
            points_a[sample], points_b[sample], min_geometry_ratio
        )
        if not sample_valid:
            continue
        valid_samples += 1
        try:
            measurement = _weighted_kabsch(
                points_a[sample], points_b[sample], covariance_sum[sample]
            )
            distance = _mahalanobis(
                measurement, points_a, points_b, cov_a, cov_b
            )
        except (RuntimeError, ValueError):
            continue
        mask = distance <= mahalanobis_threshold
        inliers = int(mask.sum())
        median = (
            float(distance[mask].median()) if inliers else float("inf")
        )
        if inliers > best_count or (inliers == best_count and median < best_median):
            best_measurement = measurement
            best_mask = mask
            best_count = inliers
            best_median = median
    if valid_samples == 0:
        return _failure(count, "degenerate_geometry", singular)
    if best_measurement is None or best_mask is None or best_count < 3:
        return _failure(count, "no_consensus", singular)

    for _ in range(2):
        measurement = _weighted_kabsch(
            points_a[best_mask],
            points_b[best_mask],
            (cov_a + cov_b)[best_mask],
        )
        distance = _mahalanobis(
            measurement, points_a, points_b, cov_a, cov_b
        )
        best_mask = distance <= mahalanobis_threshold
        if int(best_mask.sum()) < 3:
            return _failure(count, "no_consensus", singular)

    final_valid, singular = _geometry_valid(
        points_a[best_mask], points_b[best_mask], min_geometry_ratio
    )
    if not final_valid:
        return _failure(count, "degenerate_geometry", singular)
    inlier_distance = distance[best_mask]
    inliers = int(best_mask.sum())
    return SE3RansacResult(
        measurement=measurement,
        mask=best_mask,
        inliers=inliers,
        ratio=inliers / count,
        median_mahalanobis=float(inlier_distance.median()),
        p90_mahalanobis=float(torch.quantile(inlier_distance, 0.9)),
        singular_values=singular,
        reason=None,
    )
