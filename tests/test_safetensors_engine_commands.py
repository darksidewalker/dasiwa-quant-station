import os
import tempfile
import unittest
from unittest import mock

from core.safetensors_engine import run_safe_conversion


class _FakeStdout:
    def read(self, _size=1):
        return ""


class _FakeProcess:
    def __init__(self, cmd, *args, **kwargs):
        self.cmd = cmd
        self.stdout = _FakeStdout()
        self.returncode = 0
        output_path = cmd[cmd.index("-o") + 1]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(b"fake")

    def poll(self):
        return 0

    def wait(self):
        return 0


class SafetensorsEngineCommandTests(unittest.TestCase):
    def test_krea2_nvfp4_uses_dedicated_nvfp4_path_not_custom_type(self):
        commands = []

        def fake_popen(cmd, *args, **kwargs):
            commands.append(cmd)
            return _FakeProcess(cmd, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            source = os.path.join(tmpdir, "source.safetensors")
            with open(source, "wb") as f:
                f.write(b"source")

            with mock.patch("core.safetensors_engine.verify_architecture_match", return_value=(True, "ok")), \
                 mock.patch("core.safetensors_engine.subprocess.Popen", side_effect=fake_popen), \
                 mock.patch("core.safetensors_engine.inject_metadata", return_value=(True, "ok")), \
                 mock.patch("core.safetensors_engine.calculate_civitai_hashes", return_value={}), \
                 mock.patch("core.safetensors_engine.save_log"):
                list(run_safe_conversion(
                    tmpdir,
                    source,
                    ["NVFP4"],
                    "KreaTest",
                    "Krea 2",
                    "prodigy",
                    "Simple",
                    "",
                ))

        self.assertEqual(len(commands), 1)
        cmd = commands[0]
        self.assertIn("--nvfp4", cmd)
        self.assertIn("--krea2", cmd)
        self.assertIn("--layer-config", cmd)
        self.assertNotIn("--custom-type", cmd)
        self.assertNotIn("nvfp4", cmd[cmd.index("--nvfp4") + 1:cmd.index("--krea2")])


if __name__ == "__main__":
    unittest.main()
