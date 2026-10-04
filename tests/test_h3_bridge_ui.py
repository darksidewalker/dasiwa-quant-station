"""Cross-stack payload contracts without loading real model tensors."""
import ast
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

class H3BridgeContract(unittest.TestCase):
    def test_new_commands_stream_backend_generators(self):
        tree = ast.parse((ROOT / 'scripts/go_bridge.py').read_text())
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for command in ('h3_prune', 'h3_adapter_convert'):
            self.assertIn('cmd_' + command, functions)
            text = ast.unparse(functions['cmd_' + command])
            self.assertIn('run_' + command + '(payload)', text)
            self.assertIn('_emit(event)', text)
    def test_quant_kwargs_are_consumed(self):
        tree = ast.parse((ROOT / 'scripts/go_bridge.py').read_text())
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'run_safe_conversion')
        keys = {k.arg for k in call.keywords}
        self.assertIn('h3_quant_policy', keys)
        self.assertIn('verbose_level', keys)

if __name__ == '__main__': unittest.main()
