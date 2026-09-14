import json
from pathlib import Path
import tempfile
import unittest

from Scripts.Experiment.CompareOnlineORBLoop import (
    build_parser,
    loop_switch_summary,
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
        self.assertEqual(args.modes, [
            "window_skip", "window_pose_graph", "window_orb_loop",
        ])

    def test_loop_switch_summary_counts_only_effective_factors(self):
        diagnostics = {
            "pose_graph_results": [{
                "loop_switches": [
                    {"a": 0, "b": 30, "switch": 0.5},
                    {"a": 100, "b": 500, "switch": 0.02},
                    {"a": 200, "b": 900, "switch": 1e-5},
                ],
            }],
        }

        self.assertEqual(loop_switch_summary(diagnostics), {
            "max_loop_switch": 0.5,
            "effective_loops_001": 2,
            "effective_long_loops_001": 1,
        })


if __name__ == "__main__":
    unittest.main()
