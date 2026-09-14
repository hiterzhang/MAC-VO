"""Compact relative-pose graph data structures and serialization."""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np
import torch

from Module.Optimization.WindowICP import normalized_pose


POSE_FACTOR_KINDS = frozenset({"adjacent", "skip2", "loop"})


@dataclass(frozen=True)
class PoseGraphFactor:
    a: int
    b: int
    measurement: torch.Tensor
    information: torch.Tensor
    kind: str
    confidence: float
    observation_count: int

    def __post_init__(self):
        measurement = normalized_pose(
            torch.as_tensor(self.measurement).detach().cpu().double().clone()
        )
        information = torch.as_tensor(
            self.information
        ).detach().cpu().double().clone()
        if self.a < 0 or self.b <= self.a:
            raise ValueError("pose factor endpoints must be strictly forward")
        if measurement.shape != (7,) or not torch.isfinite(measurement).all():
            raise ValueError("pose factor measurement must be finite SE3")
        if information.shape != (6, 6) or not torch.isfinite(information).all():
            raise ValueError("pose factor information must be finite 6x6")
        if not torch.allclose(information, information.T, atol=1e-9):
            raise ValueError("pose factor information must be symmetric")
        if torch.linalg.cholesky_ex(information).info.item() != 0:
            raise ValueError("pose factor information must be positive definite")
        if self.kind not in POSE_FACTOR_KINDS:
            raise ValueError("unknown pose factor kind")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("pose factor confidence must be within [0, 1]")
        if self.observation_count < 1:
            raise ValueError("pose factor observation count must be positive")
        object.__setattr__(self, "measurement", measurement)
        object.__setattr__(self, "information", information)


@dataclass(frozen=True)
class PoseGraphArchive:
    poses: torch.Tensor
    time_ns: np.ndarray
    factors: tuple[PoseGraphFactor, ...]
    metadata: dict

    def __post_init__(self):
        poses = normalized_pose(
            torch.as_tensor(self.poses).detach().cpu().double().clone()
        )
        times = np.asarray(self.time_ns, dtype=np.int64).copy()
        factors = tuple(self.factors)
        if poses.ndim != 2 or poses.shape[1] != 7 or not torch.isfinite(poses).all():
            raise ValueError("pose graph poses must be finite Nx7")
        if times.shape != (len(poses),):
            raise ValueError("pose graph timestamps must match poses")
        if any(factor.b >= len(poses) for factor in factors):
            raise ValueError("pose graph factor endpoint exceeds poses")
        keys = [(factor.a, factor.b, factor.kind) for factor in factors]
        if len(keys) != len(set(keys)):
            raise ValueError("pose graph contains duplicate factors")
        object.__setattr__(self, "poses", poses)
        object.__setattr__(self, "time_ns", times)
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "metadata", dict(self.metadata))


def save_pose_graph_archive(path, archive):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    factors = archive.factors
    payload = {
        "poses": archive.poses.numpy(),
        "time_ns": archive.time_ns,
        "a": np.asarray([factor.a for factor in factors], dtype=np.int64),
        "b": np.asarray([factor.b for factor in factors], dtype=np.int64),
        "measurement": np.asarray(
            [factor.measurement.numpy() for factor in factors], dtype=np.float64
        ).reshape(-1, 7),
        "information": np.asarray(
            [factor.information.numpy() for factor in factors], dtype=np.float64
        ).reshape(-1, 6, 6),
        "kind": np.asarray([factor.kind for factor in factors], dtype="<U16"),
        "confidence": np.asarray(
            [factor.confidence for factor in factors], dtype=np.float64
        ),
        "observation_count": np.asarray(
            [factor.observation_count for factor in factors], dtype=np.int64
        ),
        "metadata_json": np.array(
            json.dumps(archive.metadata, sort_keys=True, allow_nan=False)
        ),
    }
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_pose_graph_archive(path):
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    count = len(payload["a"])
    if any(len(payload[key]) != count for key in (
        "b", "measurement", "information", "kind",
        "confidence", "observation_count",
    )):
        raise ValueError("pose graph factor arrays have different lengths")
    factors = tuple(
        PoseGraphFactor(
            int(payload["a"][index]),
            int(payload["b"][index]),
            torch.from_numpy(payload["measurement"][index].copy()),
            torch.from_numpy(payload["information"][index].copy()),
            str(payload["kind"][index]),
            float(payload["confidence"][index]),
            int(payload["observation_count"][index]),
        )
        for index in range(count)
    )
    metadata = json.loads(str(payload["metadata_json"].item()))
    return PoseGraphArchive(
        torch.from_numpy(payload["poses"].copy()),
        payload["time_ns"].copy(),
        factors,
        metadata,
    )
