# V203 Point Cloud Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export the latest complete V203 tensor map as a filtered full-resolution colored PLY and a responsive offline Plotly 3D viewer.

**Architecture:** Add one focused conversion/viewer script under `Scripts/Visualize` with pure functions for loading, filtering, PLY writing, sampling, and figure construction. Cover those functions with a small synthetic NPZ test, then run the script against the selected V203 result and verify both generated artifacts and HTTP delivery.

**Tech Stack:** Python 3, NumPy, Plotly, pytest, Python `http.server`

---

### Task 1: Add tested tensor-map conversion primitives

**Files:**
- Create: `Scripts/Visualize/visualize_tensor_map_pointcloud.py`
- Create: `Tests/test_visualize_tensor_map_pointcloud.py`

- [ ] **Step 1: Write failing tests for loading, validation, finite filtering, outlier filtering, and PLY output**

```python
from pathlib import Path
import importlib.util
import numpy as np


SCRIPT = Path("Scripts/Visualize/visualize_tensor_map_pointcloud.py")


def load_module():
    spec = importlib.util.spec_from_file_location("tensor_cloud", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_filter_and_write_ply(tmp_path):
    mod = load_module()
    source = tmp_path / "map.npz"
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [np.nan, 0, 0], [999, 999, 999]], dtype=np.float32)
    color = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=np.uint8)
    np.savez(source, **{"points//pos_Tw": pos, "points//color": color})

    xyz, rgb = mod.load_point_cloud(source)
    xyz, rgb, stats = mod.filter_point_cloud(xyz, rgb, keep_quantile=0.75)
    assert xyz.shape == (3, 3)
    assert rgb.tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert stats == {"input": 5, "finite": 4, "kept": 3}

    output = tmp_path / "cloud.ply"
    mod.write_ascii_ply(output, xyz, rgb)
    text = output.read_text()
    assert "element vertex 3" in text
    assert text.rstrip().endswith("0.000000 1.000000 0.000000 7 8 9")


def test_load_rejects_missing_color(tmp_path):
    mod = load_module()
    source = tmp_path / "bad.npz"
    np.savez(source, **{"points//pos_Tw": np.zeros((1, 3), dtype=np.float32)})
    try:
        mod.load_point_cloud(source)
    except ValueError as exc:
        assert "points//color" in str(exc)
    else:
        raise AssertionError("missing color key was accepted")
```

- [ ] **Step 2: Run tests and verify the module is missing**

Run: `pytest -q Tests/test_visualize_tensor_map_pointcloud.py`

Expected: FAIL because `Scripts/Visualize/visualize_tensor_map_pointcloud.py` does not exist.

- [ ] **Step 3: Implement the conversion primitives**

Create `Scripts/Visualize/visualize_tensor_map_pointcloud.py` with:

```python
#!/usr/bin/env python3
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


def filter_point_cloud(xyz: np.ndarray, rgb: np.ndarray, keep_quantile: float = 0.995):
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
    return filtered_xyz, filtered_rgb, {"input": len(xyz), "finite": len(finite_xyz), "kept": len(filtered_xyz)}


def write_ascii_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(("ply", "format ascii 1.0", f"element vertex {len(xyz)}", "property float x", "property float y", "property float z", "property uchar red", "property uchar green", "property uchar blue", "end_header")) + "\n"
    rows = np.column_stack((xyz, rgb))
    with path.open("w", encoding="ascii") as stream:
        stream.write(header)
        np.savetxt(stream, rows, fmt="%.6f %.6f %.6f %d %d %d")
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest -q Tests/test_visualize_tensor_map_pointcloud.py`

Expected: `2 passed`.

### Task 2: Add the offline interactive viewer and CLI

**Files:**
- Modify: `Scripts/Visualize/visualize_tensor_map_pointcloud.py`
- Modify: `Tests/test_visualize_tensor_map_pointcloud.py`

- [ ] **Step 1: Add a failing sampling/viewer test**

```python
def test_sample_and_build_viewer():
    mod = load_module()
    xyz = np.arange(900, dtype=np.float32).reshape(300, 3)
    rgb = np.tile(np.array([[10, 20, 30]], dtype=np.uint8), (300, 1))
    sampled_xyz, sampled_rgb = mod.sample_points(xyz, rgb, max_points=200, seed=42)
    assert sampled_xyz.shape == (200, 3)
    assert sampled_rgb.shape == (200, 3)
    fig = mod.build_viewer(sampled_xyz, sampled_rgb, full_count=300)
    assert fig.layout.title.text.startswith("V203 Point Cloud")
    assert len(fig.data[0].x) == 200
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `pytest -q Tests/test_visualize_tensor_map_pointcloud.py::test_sample_and_build_viewer`

Expected: FAIL because `sample_points` is not defined.

- [ ] **Step 3: Implement sampling, Plotly construction, and command-line entry point**

Add deterministic sampling, RGB strings for marker colors, a dark equal-aspect 3D scene, `write_html(..., include_plotlyjs=True)`, and arguments `--input`, `--ply-output`, `--html-output`, `--max-points`, `--seed`, and `--keep-quantile`. The CLI must print input/finite/kept/viewer counts and both output paths.

- [ ] **Step 4: Run the full unit test file**

Run: `pytest -q Tests/test_visualize_tensor_map_pointcloud.py`

Expected: `3 passed`.

- [ ] **Step 5: Commit the implementation and tests**

Run: `git add Scripts/Visualize/visualize_tensor_map_pointcloud.py Tests/test_visualize_tensor_map_pointcloud.py && git commit -m "feat: visualize tensor map point clouds"`

### Task 3: Generate and verify the V203 artifacts

**Files:**
- Create: `Results/V203_pointcloud_latest.ply`
- Create: `Results/V203_pointcloud_3d.html`

- [ ] **Step 1: Generate the artifacts from the selected complete run**

Run:

```bash
python Scripts/Visualize/visualize_tensor_map_pointcloud.py \
  --input 'Results/SparseORBLoop_V203_alpha10000_t030/20260919_181925_f362b2/window_orb_loop_sparse_t030/outputs/MACVO-Fast-WindowICP5-ORBLoop-Sparse-t030@V203/09_19_181927/tensor_map.npz' \
  --ply-output Results/V203_pointcloud_latest.ply \
  --html-output Results/V203_pointcloud_3d.html \
  --max-points 200000 \
  --seed 42 \
  --keep-quantile 0.995
```

Expected: reports 363,706 input points, nonzero filtered output, 200,000 viewer points, and both output paths.

- [ ] **Step 2: Verify artifact structure and content**

Run a read-only Python check that parses the PLY header and data, asserts finite XYZ and byte-range RGB, checks the exact vertex row count, and asserts the HTML contains `V203 Point Cloud`, `scatter3d`, and embedded Plotly JavaScript.

Expected: prints `V203 artifacts verified` and exits with status 0.

- [ ] **Step 3: Start a local HTTP server**

Run: `python -m http.server 8765 --directory Results`

Expected: server listens on port 8765.

- [ ] **Step 4: Verify HTTP delivery**

Run: `curl -I http://127.0.0.1:8765/V203_pointcloud_3d.html`

Expected: HTTP status `200 OK`.
