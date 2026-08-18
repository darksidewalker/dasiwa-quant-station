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
    def _capture_commands(self, architecture, formats=("NVFP4",), strategy="Simple"):
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
                    list(formats),
                    "CommandTest",
                    architecture,
                    "prodigy",
                    strategy,
                    "",
                ))

        return commands

    def test_krea2_nvfp4_uses_dedicated_nvfp4_path_not_custom_type(self):
        commands = self._capture_commands("Krea 2")

        self.assertEqual(len(commands), 1)
        cmd = commands[0]
        self.assertIn("--nvfp4", cmd)
        self.assertIn("--krea2", cmd)
        self.assertIn("--layer-config", cmd)
        self.assertNotIn("--custom-type", cmd)
        self.assertNotIn("nvfp4", cmd[cmd.index("--nvfp4") + 1:cmd.index("--krea2")])

    def test_all_nvfp4_architectures_use_dedicated_nvfp4_path(self):
        expected_flags = {
            "Not set": None,
            "WAN 2.2": "--wan",
            "LTX-2.3": "--ltxv2",
            "Krea 2": "--krea2",
            "Flux.2": "--flux2",
            "Hunyuan Video": "--hunyuan",
            "Qwen Image": "--qwen",
            "Z-Image": "--zimage",
            "Z-Image Refiner": "--zimage_refiner",
            "Anima": "--anima",
            "Radiance": "--radiance",
            "Distillation Large": "--distillation_large",
            "Distillation Small": "--distillation_small",
            "NeRF Large": "--nerf_large",
            "NeRF Small": "--nerf_small",
            "T5-XXL": "--t5xxl",
            "Qwen 3.5": "--qwen35",
            "Mistral": "--mistral",
            "Visual": "--visual",
            "Generic Text": "--generic_text",
        }

        for architecture, expected_flag in expected_flags.items():
            with self.subTest(architecture=architecture):
                commands = self._capture_commands(architecture)
                self.assertEqual(len(commands), 1)
                cmd = commands[0]
                self.assertIn("--nvfp4", cmd)
                self.assertIn("--comfy_quant", cmd)
                self.assertNotIn("--custom-type", cmd)
                if expected_flag is None:
                    self.assertNotIn("--layer-config", cmd)
                else:
                    self.assertIn(expected_flag, cmd)

    def test_optimizer_driven_nvfp4_also_uses_dedicated_path(self):
        for architecture in ("WAN 2.2", "LTX-2.3", "Krea 2", "Flux.2"):
            with self.subTest(architecture=architecture):
                commands = self._capture_commands(architecture, strategy="Optimizer-driven")
                self.assertEqual(len(commands), 1)
                cmd = commands[0]
                self.assertIn("--nvfp4", cmd)
                self.assertIn("--comfy_quant", cmd)
                self.assertNotIn("--custom-type", cmd)
                self.assertIn("--optimizer", cmd)
                self.assertNotIn("--simple", cmd)

    def test_layer_config_only_arch_h3_passes_guard_with_zero_arch_flags(self):
        # "MiniMax H3" has flag=None: it is a layer-config-only arch. The
        # command carries a --layer-config (DaSiWa's H3 preserve/rescue
        # patterns carry all quality) and NO upstream architecture flag.
        # The safety guard must not reject it with "missing architecture flag".
        all_arch_flags = [
            "--wan", "--ltxv2", "--krea2", "--flux2", "--hunyuan", "--qwen",
            "--zimage", "--zimage_refiner", "--anima", "--radiance",
            "--distillation_large", "--distillation_small", "--nerf_large",
            "--nerf_small", "--t5xxl", "--qwen35", "--mistral", "--visual",
            "--generic_text",
        ]
        commands = self._capture_commands("MiniMax H3")
        self.assertEqual(len(commands), 1, "guard must not abort the H3 conversion")
        cmd = commands[0]
        # A layer config was attached (quality carrier), despite flag=None.
        self.assertIn("--layer-config", cmd)
        # Zero upstream architecture flags in the assembled command.
        present_arch = [f for f in all_arch_flags if f in cmd]
        self.assertEqual(present_arch, [])
        # NVFP4 dedicated path (not the unified FP8-mislabeled path).
        self.assertIn("--nvfp4", cmd)
        self.assertNotIn("--custom-type", cmd)


if __name__ == "__main__":
    unittest.main()
