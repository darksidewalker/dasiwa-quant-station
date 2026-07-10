import re
import unittest

from core.layer_config_builder import build_layer_config_dict


class LayerConfigBuilderTests(unittest.TestCase):
    def test_krea2_nvfp4_patterns_match_raw_tensor_keys(self):
        config, _ = build_layer_config_dict("Krea 2", "NVFP4")

        def action_for(key):
            for pattern, action in config.items():
                if pattern.startswith("_"):
                    continue
                if re.fullmatch(pattern, key):
                    return action
            return config["_default"]

        for key in (
            "first.weight",
            "first.bias",
            "last.linear.weight",
            "last.linear.bias",
            "tproj.1.weight",
            "tmlp.0.weight",
            "txtmlp.1.weight",
            "txtfusion.projector.weight",
            "txtfusion.layerwise_blocks.0.attn.wq.weight",
        ):
            self.assertEqual(action_for(key), {"skip": True}, key)

        for key in (
            "blocks.0.attn.wq.weight",
            "blocks.0.attn.wk.weight",
            "blocks.0.attn.wv.weight",
            "blocks.0.attn.wo.weight",
            "blocks.0.attn.gate.weight",
            "blocks.0.mlp.gate.weight",
            "blocks.0.mlp.up.weight",
            "blocks.0.mlp.down.weight",
        ):
            self.assertEqual(
                action_for(key),
                {"format": "float8_e4m3fn", "scaling_mode": "tensor"},
                key,
            )


if __name__ == "__main__":
    unittest.main()
