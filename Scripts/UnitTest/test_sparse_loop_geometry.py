import unittest

import pypose as pp
import torch

from Module.LoopClosure.SparseGeometry import estimate_se3_ransac


def synthetic_problem(seed=4, outliers=40):
    generator = torch.Generator().manual_seed(seed)
    points_target = torch.randn(100, 3, generator=generator, dtype=torch.float64)
    points_target[:, 0] += 4.0
    tangent = torch.tensor(
        [0.25, -0.12, 0.08, 0.04, -0.03, 0.05], dtype=torch.float64
    )
    truth = pp.se3(tangent).Exp()
    points_source = truth.Act(points_target)
    points_source += torch.randn(
        points_source.shape, generator=generator, dtype=torch.float64
    ) * 0.002
    if outliers:
        points_source[:outliers] = torch.randn(
            outliers, 3, generator=generator, dtype=torch.float64
        ) * 3.0
    covariance = torch.eye(3, dtype=torch.float64)[None].repeat(100, 1, 1)
    covariance *= 1e-4
    return points_source, points_target, covariance, covariance.clone(), truth


class SparseSE3RansacTests(unittest.TestCase):
    def test_recovers_rigid_transform_with_forty_percent_outliers(self):
        points_a, points_b, cov_a, cov_b, truth = synthetic_problem()

        result = estimate_se3_ransac(
            points_a,
            points_b,
            cov_a,
            cov_b,
            iterations=256,
            mahalanobis_threshold=3.5,
            min_geometry_ratio=1e-3,
            seed=17,
        )

        self.assertIsNotNone(result.measurement)
        self.assertGreaterEqual(result.inliers, 55)
        error = (pp.SE3(result.measurement).Inv() @ truth).Log().tensor()
        self.assertLess(float(error[:3].norm()), 0.02)
        self.assertLess(float(error[3:].norm()), float(torch.deg2rad(torch.tensor(1.0))))

    def test_repeated_seed_produces_identical_result(self):
        args = synthetic_problem(seed=9)

        first = estimate_se3_ransac(
            *args[:4], iterations=128, mahalanobis_threshold=3.5,
            min_geometry_ratio=1e-3, seed=31,
        )
        second = estimate_se3_ransac(
            *args[:4], iterations=128, mahalanobis_threshold=3.5,
            min_geometry_ratio=1e-3, seed=31,
        )

        self.assertTrue(torch.equal(first.measurement, second.measurement))
        self.assertTrue(torch.equal(first.mask, second.mask))

    def test_rejects_too_few_points(self):
        points = torch.zeros(2, 3, dtype=torch.float64)
        covariance = torch.eye(3, dtype=torch.float64)[None].repeat(2, 1, 1)

        result = estimate_se3_ransac(
            points, points, covariance, covariance,
            iterations=10, mahalanobis_threshold=3.5,
            min_geometry_ratio=1e-3, seed=1,
        )

        self.assertIsNone(result.measurement)
        self.assertEqual(result.reason, "too_few_correspondences")

    def test_rejects_collinear_geometry(self):
        x = torch.linspace(1, 5, 20, dtype=torch.float64)
        points = torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)), dim=-1)
        covariance = torch.eye(3, dtype=torch.float64)[None].repeat(20, 1, 1) * 1e-4

        result = estimate_se3_ransac(
            points, points, covariance, covariance,
            iterations=64, mahalanobis_threshold=3.5,
            min_geometry_ratio=1e-3, seed=2,
        )

        self.assertIsNone(result.measurement)
        self.assertEqual(result.reason, "degenerate_geometry")

    def test_rejects_non_positive_definite_covariance(self):
        points_a, points_b, cov_a, cov_b, _ = synthetic_problem(outliers=0)
        cov_a[0, 0, 0] = -1

        result = estimate_se3_ransac(
            points_a, points_b, cov_a, cov_b,
            iterations=64, mahalanobis_threshold=3.5,
            min_geometry_ratio=1e-3, seed=3,
        )

        self.assertIsNone(result.measurement)
        self.assertEqual(result.reason, "invalid_covariance")


if __name__ == "__main__":
    unittest.main()
