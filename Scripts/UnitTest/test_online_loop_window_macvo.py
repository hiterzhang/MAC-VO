import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pypose as pp
import torch

from Module.Frontend.SerializedFrontend import SerializedFrontend
from Module.LoopClosure.ORBBoW import (
    ORBLoopCandidate,
    ORBLoopCandidateBatch,
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


if __name__ == "__main__":
    unittest.main()
