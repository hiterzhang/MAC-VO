import threading
import time
import unittest

import pypose as pp

from Module.Optimization.AsyncPoseGraph import (
    AsyncPoseGraphBackend,
    PoseGraphSnapshot,
)


def snapshot(version):
    return PoseGraphSnapshot(
        graph_version=version,
        frame_count=2,
        poses=pp.identity_SE3(2).tensor(),
        factors=(),
    )


class BlockingSolver:
    def __init__(self):
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, poses, factors):
        self.calls.append(len(self.calls))
        self.started.set()
        self.release.wait(timeout=2)
        return {"poses": poses, "status": "refined"}


class AsyncPoseGraphTests(unittest.TestCase):
    def test_busy_backend_coalesces_to_newest_pending_snapshot(self):
        solver = BlockingSolver()
        backend = AsyncPoseGraphBackend(solver=solver)
        backend.submit(snapshot(1))
        self.assertTrue(solver.started.wait(timeout=1))
        backend.submit(snapshot(2))
        backend.submit(snapshot(3))
        solver.release.set()
        deadline = time.time() + 2
        results = []
        while len(results) < 2 and time.time() < deadline:
            results.extend(backend.poll())
            time.sleep(0.01)
        backend.close()

        self.assertEqual([result.graph_version for result in results], [1, 3])
        self.assertEqual(backend.diagnostics["coalesced_submissions"], 1)

    def test_solver_exception_becomes_failed_result(self):
        backend = AsyncPoseGraphBackend(
            solver=lambda *_: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        backend.submit(snapshot(4))
        deadline = time.time() + 2
        results = []
        while not results and time.time() < deadline:
            results = backend.poll()
            time.sleep(0.01)
        backend.close()

        self.assertEqual(results[0].status, "failed")
        self.assertIn("boom", results[0].reason)

    def test_terminate_solves_final_newest_snapshot(self):
        versions = []

        def solver(poses, factors):
            versions.append(len(poses))
            return {"poses": poses, "status": "refined"}

        backend = AsyncPoseGraphBackend(solver=solver)
        backend.submit(snapshot(1))
        backend.terminate(snapshot(9))
        results = backend.poll()

        self.assertEqual(results[-1].graph_version, 9)
        self.assertFalse(backend.is_alive)


if __name__ == "__main__":
    unittest.main()
