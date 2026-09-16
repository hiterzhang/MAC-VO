import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pypose as pp
import torch

from Module.Frontend.SerializedFrontend import SerializedFrontend
from Module.LoopClosure.ORBBoW import (
    ORBLoopCandidate,
    ORBLoopCandidateBatch,
)
from Module.LoopClosure.LoopHypothesis import (
    HypothesisConfig,
    LoopHypothesisTracker,
)
from Module.LoopClosure.SparseGeometry import (
    SparseGeometryConfig,
    SparseLoopResult,
    conservative_sparse_factor,
)
from Module.Optimization.PairwiseICP import PairwiseCompressionResult
from Module.Optimization.PoseGraph import PoseGraphFactor
from Module.Optimization.WindowICP import Edge
from Odometry.OnlineLoopWindowMACVO import OnlineLoopWindowMACVO
from Odometry.WindowMACVO import WindowMACVO


def edge(a=0, b=1):
    points = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float64)
    covariance = torch.eye(3, dtype=torch.float64)[None] * 0.001
    return Edge(a, b, points, points.clone(), covariance, covariance.clone())


def factor(a=0, b=1, kind="adjacent"):
    return PoseGraphFactor(
        a, b, pp.identity_SE3(dtype=torch.float64).tensor(),
        torch.eye(6, dtype=torch.float64), kind, 1.0, 10,
    )


def sparse_candidate(source, target, count=50):
    matches = np.zeros((count, 5), dtype=np.float64)
    matches[:, 0] = np.linspace(10, 50, count)
    matches[:, 1] = np.linspace(10, 40, count)
    matches[:, 2:4] = matches[:, :2]
    return ORBLoopCandidate(
        source=source,
        score=0.2,
        rank=0,
        raw_knn_matches=count + 20,
        ratio_matches=count + 10,
        matches=matches,
    )


def sparse_validation(source, target, translation=0.0):
    measurement = pp.se3(torch.tensor(
        [translation, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64
    )).Exp().tensor()
    return SparseLoopResult(
        source=source,
        target=target,
        measurement=measurement,
        inlier_mask=torch.ones(40, dtype=torch.bool),
        metrics={
            "ransac_inliers": 40,
            "ransac_ratio": 0.8,
            "final_grid_cells": 10,
            "reprojection_forward_p90_px": 1.0,
            "reprojection_backward_p90_px": 1.0,
        },
        reason=None,
    )


class OnlineLoopWindowMACVOTests(unittest.TestCase):
    def test_window_hook_is_default_noop(self):
        system = WindowMACVO.__new__(WindowMACVO)

        self.assertIsNone(system._after_window_step(None, None, None, None))

    def test_frontend_wrapper_reuses_original_object(self):
        original = SimpleNamespace(model=object(), cuda_graph=object())
        system = OnlineLoopWindowMACVO.__new__(OnlineLoopWindowMACVO)
        system.Frontend = original

        system._wrap_frontend_once()
        first = system.Frontend
        system._wrap_frontend_once()

        self.assertIsInstance(first, SerializedFrontend)
        self.assertIs(system.Frontend, first)
        self.assertIs(first.frontend, original)

    def test_compressed_factor_is_inserted_once_and_increments_version(self):
        system = OnlineLoopWindowMACVO.__new__(OnlineLoopWindowMACVO)
        system.pose_factors = {}
        system.pose_graph_version = 0
        system.compression_records = []
        system.pairwise_iterations = 5
        system.pairwise_huber_delta = 3.0
        system.compress_edge = Mock(return_value=PairwiseCompressionResult(
            factor(0, 1), "compressed", None, 1.0, 0.5, 10.0
        ))
        system.linearize_edge = Mock(return_value=PairwiseCompressionResult(
            factor(0, 1), "compressed", None, 0.5, 0.5, 10.0
        ))
        system.graph = SimpleNamespace(frames=SimpleNamespace(
            data={"pose": SimpleNamespace(tensor=pp.identity_SE3(2).tensor())}
        ))

        first = system._compress_and_store(edge(), "adjacent")
        second = system._compress_and_store(edge(), "adjacent")

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(system.pose_graph_version, 1)
        self.assertEqual(len(system.pose_factors), 1)
        system.linearize_edge.assert_called_once()
        system.compress_edge.assert_not_called()

    def test_runtime_folder_creates_bounded_online_stores(self):
        system = OnlineLoopWindowMACVO.__new__(OnlineLoopWindowMACVO)
        system.loop_enabled = True
        system.max_pending_targets = 3
        system.loop_provider = None
        system.orb_sidecar = "/missing"
        system.orb_vocabulary = "/missing"
        system.loop_min_temporal_gap = 30
        system.bow_top_k = 3
        system.max_candidate_queue = 8

        with tempfile.TemporaryDirectory() as directory:
            system.set_runtime_folder(Path(directory))

            self.assertTrue(Path(directory, "online_loop_keyframes").is_dir())
            self.assertEqual(system.pending_targets.capacity, 3)
            self.assertFalse(system.loop_provider.enabled)

    def test_orb_response_without_eligible_candidate_releases_target(self):
        system = OnlineLoopWindowMACVO.__new__(OnlineLoopWindowMACVO)
        system.loop_provider = SimpleNamespace(poll=lambda: [
            ORBLoopCandidateBatch(
                1, 40, (ORBLoopCandidate(0, 0.001, 0),)
            )
        ])
        system.min_bow_score = 0.005
        system.existing_loop_pairs = set()
        system.pending_loop_pairs = set()
        system.rejected_loop_pairs = set()
        system.candidate_nms_frames = 10
        system.max_candidates_per_target = 1
        system.candidate_filter_counts = __import__("collections").Counter()
        system.loop_candidates = __import__("collections").deque(maxlen=8)
        system.pending_targets = SimpleNamespace(pop=Mock())

        system._poll_orb_candidates()

        system.pending_targets.pop.assert_called_once_with(40)

    def test_sparse_candidate_with_too_few_matches_skips_gpu_depth(self):
        system = OnlineLoopWindowMACVO.__new__(OnlineLoopWindowMACVO)
        system.sparse_geometry_config = SparseGeometryConfig(
            min_mutual_matches=40
        )
        system.Frontend = SimpleNamespace(estimate_loop_depth=Mock())
        system.sparse_validator = Mock()
        source = SimpleNamespace(stereo=object())
        target = SimpleNamespace(stereo=object(), depth=object())

        result = system._validate_sparse_candidate(
            source, target, sparse_candidate(0, 50, count=20)
        )

        self.assertEqual(result.reason, "insufficient_mutual_matches")
        system.Frontend.estimate_loop_depth.assert_not_called()
        system.sparse_validator.assert_not_called()

    def test_second_sparse_support_stores_one_direct_loop_factor(self):
        system = OnlineLoopWindowMACVO.__new__(OnlineLoopWindowMACVO)
        system.graph = SimpleNamespace(frames=SimpleNamespace(data={
            "pose": SimpleNamespace(tensor=pp.identity_SE3(30).tensor())
        }))
        system.hypothesis_tracker = LoopHypothesisTracker(HypothesisConfig(
            min_supports=2,
            strong_supports=3,
            source_cluster_frames=10,
            target_support_frames=10,
            max_correction_translation_m=0.25,
            max_correction_rotation_deg=10.0,
        ))
        system.sparse_factor_builder = conservative_sparse_factor
        system.sparse_translation_sigma = 0.25
        system.sparse_rotation_sigma_deg = 10.0
        system.pose_factors = {}
        system.pose_graph_version = 0
        system.existing_loop_pairs = set()
        system.hypothesis_records = []
        system.loop_records = []
        system.pose_backend = SimpleNamespace(submit=Mock())
        system._pose_graph_snapshot = Mock(return_value="snapshot")
        system.compress_edge = Mock()

        first = system._register_sparse_support(
            sparse_candidate(10, 20), sparse_validation(10, 20)
        )
        second = system._register_sparse_support(
            sparse_candidate(15, 25), sparse_validation(15, 25)
        )

        self.assertFalse(first)
        self.assertTrue(second)
        self.assertEqual(len(system.pose_factors), 1)
        stored = next(iter(system.pose_factors.values()))
        self.assertEqual((stored.a, stored.b, stored.kind), (10, 20, "loop"))
        self.assertAlmostEqual(float(stored.information[0, 0]), 16.0)
        system.compress_edge.assert_not_called()
        system.pose_backend.submit.assert_called_once_with("snapshot")


if __name__ == "__main__":
    unittest.main()
