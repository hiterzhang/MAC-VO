import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build/orb_bow/macvo_orb_bow"


def tiny_vocabulary(path):
    zero = " ".join(["0"] * 32)
    full = " ".join(["255"] * 32)
    path.write_text(
        f"2 1 0 0\n0 1 {zero} 1.0\n0 1 {full} 1.0",
        encoding="utf-8",
    )


class ORBBoWSidecarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = os.environ.copy()
        env["ORB_SLAM3_ROOT"] = "/home/zzh/ORB_SLAM3"
        subprocess.run(
            [str(ROOT / "Scripts/Build/build_orb_bow_sidecar.sh")],
            cwd=ROOT,
            env=env,
            check=True,
        )

    def test_handshake_query_temporal_filter_and_malformed_command(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            vocabulary = directory / "tiny_vocabulary.txt"
            tiny_vocabulary(vocabulary)
            rng = np.random.default_rng(9)
            image = (rng.random((128, 128)) * 255).astype(np.uint8)
            image_path = directory / "image.png"
            cv2.imwrite(str(image_path), image)
            process = subprocess.Popen(
                [str(BUILD), "--vocabulary", str(vocabulary)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            assert process.stdin is not None and process.stdout is not None
            ready = process.stdout.readline().strip().split("\t")
            self.assertEqual(ready[0], "READY")
            self.assertEqual(ready[1], "1")

            process.stdin.write(f"QUERY\t1\t0\t{image_path}\t30\t3\n")
            process.stdin.flush()
            first = process.stdout.readline().strip().split("\t")
            self.assertEqual(first[:4], ["RESULT", "1", "0", "0"])

            process.stdin.write(f"QUERY\t2\t40\t{image_path}\t30\t3\n")
            process.stdin.flush()
            second = process.stdout.readline().strip().split("\t")
            self.assertEqual(second[:4], ["RESULT", "2", "40", "1"])
            source, score = second[4].split(":")
            self.assertEqual(source, "0")
            self.assertGreater(float(score), 0)

            process.stdin.write("BAD\n")
            process.stdin.flush()
            self.assertTrue(process.stdout.readline().startswith("ERROR\t"))

            process.stdin.write("STOP\n")
            process.stdin.flush()
            self.assertEqual(process.stdout.readline().strip(), "BYE")
            self.assertEqual(process.wait(timeout=5), 0)


if __name__ == "__main__":
    unittest.main()
