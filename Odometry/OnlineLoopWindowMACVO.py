"""WindowICP with asynchronous ORB retrieval and serialized online loop closure."""

from collections import Counter, deque
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pypose as pp
import torch

from Module.Frontend.SerializedFrontend import SerializedFrontend
from Module.LoopClosure.ORBBoW import (
    ORBLoopCandidateProvider,
    filter_loop_candidates,
)
from Module.LoopClosure.OnlineLoopStore import (
    LoopKeyframeStore,
    PendingLoopTarget,
    PendingLoopTargetStore,
)
from Module.Optimization.AsyncPoseGraph import (
    AsyncPoseGraphBackend,
    PoseGraphSnapshot,
)
from Module.Optimization.CovisibilitySelector import CovisibilityMetrics
from Module.Optimization.PairwiseICP import (
    compress_edge_to_pose_factor,
    linearize_edge_to_pose_factor,
)
from Module.Optimization.PoseGraph import (
    PoseGraphArchive,
    optimize_pose_graph,
    save_pose_graph_archive,
)
from Module.Optimization.ProximityICP import (
    ProximityValidationConfig,
    depth_output_to_device,
    validate_proximity_match,
)
from Odometry.WindowMACVO import WindowMACVO, reanchor_points


class OnlineLoopWindowMACVO(WindowMACVO):
    def __init__(
        self,
        *,
        online_loop,
        pairwise_icp,
        pose_graph,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loop_enabled = bool(online_loop.enabled)
        self.loop_keyframe_stride = int(online_loop.keyframe_stride)
        self.loop_min_temporal_gap = int(online_loop.min_temporal_gap)
        self.bow_top_k = int(online_loop.bow_top_k)
        self.max_candidates_per_target = int(
            online_loop.max_candidates_per_target
        )
        self.candidate_nms_frames = int(online_loop.candidate_nms_frames)
        self.max_candidate_queue = int(online_loop.max_candidate_queue)
        self.max_pending_targets = int(online_loop.max_pending_targets)
        self.min_bow_score = float(getattr(online_loop, "min_bow_score", 0.02))
        self.min_frames_between_loop_matches = int(
            getattr(online_loop, "min_frames_between_loop_matches", 25)
        )
        self.orb_sidecar = str(online_loop.orb_sidecar)
        self.orb_vocabulary = str(online_loop.vocabulary)
        self.pairwise_iterations = int(pairwise_icp.iterations)
        self.pairwise_huber_delta = float(pairwise_icp.huber_delta)
        self.pose_graph_iterations = int(pose_graph.iterations)
        self.pose_graph_huber_delta = float(pose_graph.huber_delta)
        self.skip_information_cap = float(pose_graph.skip_information_cap)
        self.switch_prior = float(pose_graph.switch_prior)
        self.compress_edge = compress_edge_to_pose_factor
        self.linearize_edge = linearize_edge_to_pose_factor
        self.pose_factors = {}
        self.pose_graph_version = 0
        self.compression_records = []
        self.loop_records = []
        self.loop_candidates = deque(maxlen=self.max_candidate_queue)
        self.existing_loop_pairs = set()
        self.pending_loop_pairs = set()
        self.rejected_loop_pairs = set()
        self.candidate_filter_counts = Counter()
        self.expired_targets = 0
        self.last_loop_match_frame = -10**9
        self.pose_graph_writebacks = 0
        self.pose_graph_results = []
        self.runtime_folder = None
        self.loop_keyframes = None
        self.pending_targets = None
        self.loop_provider = None
        self.pose_backend = AsyncPoseGraphBackend(
            solver=lambda poses, factors: optimize_pose_graph(
                poses,
                factors,
                max_iters=self.pose_graph_iterations,
                huber_delta=self.pose_graph_huber_delta,
                skip_information_cap=self.skip_information_cap,
                switch_prior=self.switch_prior,
            )
        )
        self._wrap_frontend_once()

    @classmethod
    def is_valid_config(cls, config):
        assert config is not None
        base_args = SimpleNamespace(**{
            key: value
            for key, value in vars(config.args).items()
            if key not in {"online_loop", "pairwise_icp", "pose_graph"}
        })
        base = SimpleNamespace(**vars(config))
        base.args = base_args
        WindowMACVO.is_valid_config(base)
        for name in ("online_loop", "pairwise_icp", "pose_graph"):
            if not hasattr(config.args, name):
                raise ValueError(f"Online loop config is missing {name}")

    def _wrap_frontend_once(self):
        if not isinstance(self.Frontend, SerializedFrontend):
            self.Frontend = SerializedFrontend(self.Frontend)

    def set_runtime_folder(self, folder):
        self.runtime_folder = Path(folder)
        self.loop_keyframes = LoopKeyframeStore(
            self.runtime_folder / "online_loop_keyframes"
        )
        self.pending_targets = PendingLoopTargetStore(
            self.max_pending_targets
        )
        if self.loop_provider is None:
            if self.loop_enabled:
                self.loop_provider = ORBLoopCandidateProvider(
                    executable=Path(self.orb_sidecar),
                    vocabulary=Path(self.orb_vocabulary),
                    min_temporal_gap=self.loop_min_temporal_gap,
                    top_k=self.bow_top_k,
                    queue_size=self.max_candidate_queue,
                )
            else:
                self.loop_provider = None
        if self.loop_provider is not None and not self.loop_provider.enabled:
            self.loop_enabled = False

    def _compress_and_store(self, edge, kind, confidence=1.0):
        if edge is None:
            return False
        key = (edge.a, edge.b, kind)
        if key in self.pose_factors:
            return False
        if kind == "loop":
            result = self.compress_edge(
                edge,
                kind=kind,
                max_iters=self.pairwise_iterations,
                huber_delta=self.pairwise_huber_delta,
            )
        else:
            result = self.linearize_edge(
                edge,
                self.graph.frames.data["pose"].tensor.double(),
                kind=kind,
                huber_delta=self.pairwise_huber_delta,
            )
        self.compression_records.append({
            "a": edge.a,
            "b": edge.b,
            "kind": kind,
            "status": result.status,
            "reason": result.reason,
            "condition_number": result.condition_number,
        })
        if result.factor is None:
            return False
        self.pose_factors[key] = replace(
            result.factor, confidence=float(confidence)
        )
        self.pose_graph_version += 1
        return True

    def initialize(self, frame0):
        super().initialize(frame0)
        if self.loop_enabled:
            self._publish_loop_keyframe(frame0, self.prev_keyframe[2], 0)

    def _publish_loop_keyframe(self, frame, depth, frame_id):
        if (
            not self.loop_enabled
            or self.loop_keyframes is None
            or self.pending_targets is None
            or frame_id % self.loop_keyframe_stride
        ):
            return
        pose = self.graph.frames.data["pose"][frame_id].double()
        stored = self.loop_keyframes.add(frame_id, frame.stereo, pose)
        expired = self.pending_targets.add(
            PendingLoopTarget(frame_id, frame.stereo, depth)
        )
        if expired is not None:
            self.expired_targets += 1
        try:
            self.loop_provider.submit(
                frame_id=frame_id, left_image_path=stored.left_path
            )
        except Exception as error:
            self.loop_enabled = False
            self.loop_records.append({
                "status": "provider_failed", "reason": str(error)
            })

    def _poll_orb_candidates(self):
        if self.loop_provider is None:
            return
        for response in self.loop_provider.poll():
            raw = [
                (candidate.source, candidate.score)
                for candidate in response.candidates
                if candidate.score >= self.min_bow_score
            ]
            selected, counts = filter_loop_candidates(
                target=response.target,
                candidates=raw,
                existing_pairs=self.existing_loop_pairs,
                pending_pairs=self.pending_loop_pairs,
                rejected_pairs=self.rejected_loop_pairs,
                nms_frames=self.candidate_nms_frames,
                maximum=self.max_candidates_per_target,
            )
            self.candidate_filter_counts.update(counts)
            if not selected:
                self.pending_targets.pop(response.target)
            for candidate in selected:
                pair = (candidate.source, response.target)
                if len(self.loop_candidates) == self.loop_candidates.maxlen:
                    dropped_target, dropped = self.loop_candidates.popleft()
                    dropped_pair = (dropped.source, dropped_target)
                    self.pending_loop_pairs.discard(dropped_pair)
                    self.pending_targets.pop(dropped_target)
                self.pending_loop_pairs.add(pair)
                self.loop_candidates.append((response.target, candidate))

    def _process_one_loop_candidate(self, current_frame_id):
        if (
            not self.loop_candidates
            or current_frame_id - self.last_loop_match_frame
            < self.min_frames_between_loop_matches
        ):
            return False
        target_id, candidate = self.loop_candidates.popleft()
        pair = (candidate.source, target_id)
        self.pending_loop_pairs.discard(pair)
        target = self.pending_targets.get(target_id)
        if target is None or candidate.source not in self.loop_keyframes.records:
            self.rejected_loop_pairs.add(pair)
            self.loop_records.append({
                "source": candidate.source,
                "target": target_id,
                "status": "expired",
            })
            return False
        source = self.loop_keyframes.load(candidate.source)
        source.stereo.imageL = source.stereo.imageL.to(target.stereo.imageL)
        source.stereo.imageR = source.stereo.imageR.to(target.stereo.imageR)
        source_depth, forward, backward = self.Frontend.estimate_bidirectional(
            source.stereo, target.stereo
        )
        metric = CovisibilityMetrics(
            source=candidate.source,
            target=target_id,
            overlap_forward=1.0,
            overlap_backward=1.0,
            mean_overlap=1.0,
            depth_consistency=1.0,
            median_motion_px=0.0,
            score=max(0.0, 1.0 - candidate.score),
            status="eligible",
            reason=None,
        )
        validation_config = ProximityValidationConfig(
            mahalanobis_threshold=1e9,
            min_mahalanobis_inliers=30,
            min_mahalanobis_inlier_ratio=0.5,
        )
        validation = validate_proximity_match(
            candidate=metric,
            forward=forward,
            backward=backward,
            source_frame=source,
            target_frame=SimpleNamespace(stereo=target.stereo),
            source_depth=source_depth,
            target_depth=depth_output_to_device(target.depth, self.device),
            poses=self.graph.frames.data["pose"].tensor.double(),
            frontend=self.Frontend,
            keypoint_selector=self.KeypointSelector,
            covariance_model=self.ObsCovModel,
            num_point=self.num_point,
            edge_width=self.edge_width,
            match_cov_default=self.match_cov_default,
            device=self.device,
            config=validation_config,
        )
        self.last_loop_match_frame = current_frame_id
        self.pending_targets.pop(target_id)
        if validation.record is None:
            self.rejected_loop_pairs.add(pair)
            self.loop_records.append({
                "source": candidate.source,
                "target": target_id,
                "bow_score": candidate.score,
                "status": "rejected",
                "reason": validation.reason,
            })
            return False
        accepted = self._compress_and_store(
            validation.record.edge,
            "loop",
            validation.record.confidence,
        )
        if not accepted:
            self.rejected_loop_pairs.add(pair)
            return False
        self.existing_loop_pairs.add(pair)
        self.loop_records.append({
            "source": candidate.source,
            "target": target_id,
            "gap": target_id - candidate.source,
            "bow_score": candidate.score,
            "status": "accepted",
            "observations": len(validation.record.edge.points_a),
        })
        self.pose_backend.submit(self._pose_graph_snapshot())
        return True

    def _pose_graph_snapshot(self):
        poses = self.graph.frames.data["pose"].tensor.double().clone()
        factors = tuple(
            self.pose_factors[key] for key in sorted(self.pose_factors)
        )
        return PoseGraphSnapshot(
            self.pose_graph_version, len(poses), poses, factors
        )

    def _apply_pose_graph_result(self, asynchronous):
        if asynchronous.status != "refined" or asynchronous.result is None:
            self.pose_graph_results.append({
                "status": asynchronous.status,
                "reason": asynchronous.reason,
            })
            return False
        solved = asynchronous.result
        if isinstance(solved, dict):
            return False
        if solved.diagnostics.get("status") != "refined":
            return False
        old = self.graph.frames.data["pose"].tensor.double().clone()
        count = asynchronous.frame_count
        if count > len(old) or len(solved.poses) != count:
            return False
        after = old.clone()
        after[:count] = solved.poses
        if count < len(old):
            correction = pp.SE3(after[count - 1]) @ pp.SE3(
                old[count - 1]
            ).Inv()
            after[count:] = (correction @ pp.SE3(old[count:])).tensor()
        reanchor_points(self.graph, list(range(len(old))), old, after)
        self.graph.frames.data["pose"].tensor[:] = after.float()
        if self.prev_keyframe is not None:
            self.MotionEstimator.update(pp.SE3(after[-1]))
        self.pose_graph_writebacks += 1
        self.pose_graph_results.append(solved.diagnostics)
        return True

    def _poll_pose_backend(self):
        results = self.pose_backend.poll()
        if results:
            self._apply_pose_graph_result(results[-1])

    def _after_window_step(self, frame, depth, adjacent, skip):
        self._compress_and_store(adjacent, "adjacent")
        self._compress_and_store(skip, "skip2")
        frame_id = self.prev_keyframe[1]
        self._publish_loop_keyframe(frame, depth, frame_id)
        self._poll_orb_candidates()
        self._process_one_loop_candidate(frame_id)
        self._poll_pose_backend()

    def terminate(self):
        if self.terminated:
            return
        if self.loop_provider is not None:
            self.loop_provider.close()
            self._poll_orb_candidates()
            while self.loop_candidates:
                self.last_loop_match_frame = -10**9
                if not self._process_one_loop_candidate(
                    len(self.graph.frames) - 1
                ):
                    continue
        super().terminate()
        if self.pose_factors:
            self.pose_backend.terminate(self._pose_graph_snapshot())
        else:
            self.pose_backend.close()
        results = self.pose_backend.poll()
        if results:
            self._apply_pose_graph_result(results[-1])

    def save_window_diagnostics(self, folder):
        super().save_window_diagnostics(folder)
        folder = Path(folder)
        factors = tuple(
            self.pose_factors[key] for key in sorted(self.pose_factors)
        )
        if factors:
            graph = self.graph.frames.data
            save_pose_graph_archive(
                folder / "pose_graph_factors.npz",
                PoseGraphArchive(
                    graph["pose"].tensor.double(),
                    graph["time_ns"].tensor.cpu().numpy(),
                    factors,
                    {"graph_version": self.pose_graph_version},
                ),
            )
        payload = {
            "loop_enabled": self.loop_enabled,
            "provider": None if self.loop_provider is None else self.loop_provider.status,
            "frontend": self.Frontend.diagnostics,
            "pose_graph_version": self.pose_graph_version,
            "pose_factors": len(factors),
            "factor_kinds": dict(Counter(factor.kind for factor in factors)),
            "loop_records": self.loop_records,
            "compression_records": self.compression_records,
            "candidate_filter_counts": dict(self.candidate_filter_counts),
            "expired_targets": self.expired_targets,
            "pose_backend": self.pose_backend.diagnostics,
            "pose_graph_writebacks": self.pose_graph_writebacks,
            "pose_graph_results": self.pose_graph_results,
            "cuda_max_memory_reserved": (
                torch.cuda.max_memory_reserved()
                if torch.cuda.is_available() else 0
            ),
        }
        (folder / "online_loop_diagnostics.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
        )
