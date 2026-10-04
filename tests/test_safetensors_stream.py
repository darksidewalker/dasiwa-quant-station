import os
import tempfile
import unittest
from pathlib import Path
from core.safetensors_stream import destination


class DestinationTests(unittest.TestCase):
    def test_uppercase_suffix_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'source.safetensors'
            source.touch()
            output = str(Path(d) / 'out.SAFETENSORS')
            self.assertEqual(destination({'output_path': output}, 'unused', [str(source)]), output)

    def test_job_owned_staging_is_respected(self):
        from unittest.mock import patch
        from core.safetensors_stream import TensorSpool
        with tempfile.TemporaryDirectory() as d:
            owned = Path(d) / '.h3_job_owned'
            owned.mkdir()
            with patch.dict(os.environ, {'DASIWA_H3_STAGE_DIR': str(owned)}):
                with TensorSpool(str(Path(d) / 'out.safetensors')) as spool:
                    self.assertEqual(Path(spool.temp.name).parent, owned)
            self.assertEqual(list(owned.iterdir()), [])

    def test_bridge_inspect_exposes_variant_and_coordinates(self):
        import json
        import subprocess
        import sys
        from safetensors.torch import save_file
        from tests.test_h3_variant_contract import full_tensors
        from core.h3_curve import TIME_KEYS
        import torch
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'pruned.safetensors'
            tensors = {k:v for k,v in full_tensors().items() if k not in TIME_KEYS}
            tensors['adaln_t_table'] = torch.zeros(65, 3)
            save_file(tensors, str(path))
            result = subprocess.run([sys.executable, str(root/'scripts/go_bridge.py'), 'inspect', str(path)], capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            self.assertEqual(data['h3']['variant'], 'pruned')
            self.assertEqual(len(data['h3']['adaln_coordinate_table_sha256']), 64)

    def test_bridge_reports_installed_ctq_capability(self):
        import json
        import subprocess
        import sys
        from core.safetensors_engine import h3_ctq_capability
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, str(root/'scripts/go_bridge.py'), 'capabilities'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        supported, detail = h3_ctq_capability()
        self.assertEqual(data['h3_ctq'], {'supported': supported, 'detail': detail})

    def test_exact_recipe_blocks_publication(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'source.safetensors'
            source.touch()
            output = str(Path(d) / 'out.safetensors')
            Path(output+'.txt').touch()
            with self.assertRaises(FileExistsError):
                destination({'output_path': output}, 'unused', [str(source)])


if __name__ == '__main__':
    unittest.main()
