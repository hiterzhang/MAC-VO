import io
from pathlib import Path
import sys
import tempfile
import unittest

from Scripts.Experiment.CompareWindowICP import run_and_tee


class LiveOutputTests(unittest.TestCase):
    def test_subprocess_output_is_written_to_terminal_and_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory, "run.log")
            terminal = io.BytesIO()
            run_and_tee(
                [sys.executable, "-c", "import sys; sys.stdout.write('frame 1\\rframe 2\\n'); sys.stdout.flush()"],
                Path(directory), None, log, terminal,
            )
            expected = b"frame 1\rframe 2\n"
            self.assertEqual(terminal.getvalue(), expected)
            self.assertEqual(log.read_bytes(), expected)


if __name__ == "__main__":
    unittest.main()
