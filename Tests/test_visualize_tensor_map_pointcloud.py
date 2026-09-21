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
    pos = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [np.nan, 0, 0], [999, 999, 999]],
        dtype=np.float32,
    )
    color = np.array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
        dtype=np.uint8,
    )
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
