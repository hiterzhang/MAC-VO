import json
from pathlib import Path
import tempfile
import unittest

from Scripts.Experiment.CompareOnlineORBLoop import (
    build_parser,
    online_invariants,
    write_rows,
)


class CompareOnlineORBLoopTests(unittest.TestCase):
    def test_online_invariants_require_single_model_graph_and_concurrency(self):
        diagnostics = {
            "frontend": {
                "wrapped_instances_created": 1,
                "wrapped_graphs_captured": 1,
                "max_concurrent_calls": 1,
            },
            "cuda_max_memory_reserved": 5 * 1024**3,
        }

        result = online_invariants(diagnostics)

        self.assertTrue(result["passed"])
        diagnostics["frontend"]["max_concurrent_calls"] = 2
        self.assertFalse(online_invariants(diagnostics)["passed"])

    def test_write_rows_preserves_mode_order(self):
        rows = [
            {"mode": "window_skip", "RMSE_ATE": 0.2},
            {"mode": "window_pose_graph", "RMSE_ATE": 0.15},
            {"mode": "window_orb_loop", "RMSE_ATE": 0.1},
        ]
        with tempfile.TemporaryDirectory() as directory:
            write_rows(directory, rows)
            saved = json.loads(Path(directory, "metrics.json").read_text())

        self.assertEqual(
            [row["mode"] for row in saved],
            ["window_skip", "window_pose_graph", "window_orb_loop"],
        )

    def test_parser_accepts_complete_sequence(self):
        args = build_parser().parse_args([
            "--sequence", "V203", "--seq-from", "0"
        ])

        self.assertEqual(args.sequence, "V203")
        self.assertIsNone(args.seq_to)


if __name__ == "__main__":
    unittest.main()
