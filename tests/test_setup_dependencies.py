import pathlib
import tomllib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class SetupDependencyTests(unittest.TestCase):
    def test_runtime_dependencies_have_one_uv_source_of_truth(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        dependencies = config["project"]["dependencies"]
        names = {dependency.split("[", 1)[0].split("=", 1)[0].lower() for dependency in dependencies}

        self.assertIn("torch", names)
        self.assertIn("comfy-kitchen", names)
        self.assertIn("convert-to-quant", names)
        self.assertIn("packaging", names)
        self.assertNotIn("torchvision", names)
        self.assertNotIn("torchaudio", names)
        self.assertNotIn("sentencepiece", names)

    def test_setup_resolves_and_installs_python_packages_once(self):
        script = (ROOT / "start-linux.sh").read_text()

        self.assertEqual(script.count("uv sync"), 1)
        self.assertNotIn("uv pip install", script)
        self.assertNotIn("git+https://github.com/Comfy-Org/comfy-kitchen", script)
        self.assertNotIn("comfy-kitchen[cublas]", script)
        self.assertIn("uv pip check", script)
        self.assertIn("uv self update", script)
        self.assertIn("GGUFY_VERSION_FILE", script)
        self.assertIn("releases/latest", script)
        self.assertIn("/uv.lock", (ROOT / ".gitignore").read_text())


if __name__ == "__main__":
    unittest.main()
