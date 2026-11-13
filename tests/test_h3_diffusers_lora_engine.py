"""Diffusers -> native MiniMax H3 LoRA conversion contract (synthetic tiny tensors, CPU only)."""
import os
import unittest
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import load_file, save_file

from core.h3_diffusers_lora_engine import run_h3_diffusers_lora

HID, HEADS, HD, FFN, LAYERS, REFINER, CURVE = 16, 2, 8, 24, 2, 1, 8
INNER = HEADS * HD
MODULES = {"attn.to_q": (HID, INNER), "attn.to_k": (HID, INNER), "attn.to_v": (HID, INNER),
           "attn.to_out.0": (INNER, HID), "ff.net.0.proj": (HID, 2 * FFN), "ff.net.2": (FFN, HID)}
STACKS = [(f"transformer_blocks.{n}", f"blocks.{n}") for n in range(LAYERS)] + \
         [(f"token_refiner.refiner_blocks.{n}", f"token_refiner.blocks.{n}") for n in range(REFINER)]


def native_checkpoint(curve=True):
    z = lambda *s: torch.zeros(s, dtype=torch.bfloat16)
    sd = {"rope.inv_freq": z(2)}
    if curve:
        sd["adaln_t_table"] = z(33, CURVE)
    for _, b in STACKS:
        sd |= {f"{b}.attn.qkv_proj.weight": z(3 * INNER, HID), f"{b}.attn.out_proj.weight": z(HID, INNER),
               f"{b}.mlp.fc1.weight": z(2 * FFN, HID), f"{b}.mlp.fc2.weight": z(HID, FFN)}
        if not b.startswith("token"):
            sd[f"{b}.adaln_proj.linear.weight"] = z(18 * HID, CURVE)
    return sd


def diffusers_lora(prefix="transformer.", down="lora_A", up="lora_B", alpha=False, rank=3):
    g = torch.Generator().manual_seed(0)
    sd = {}
    for stack, _ in STACKS:
        for i, (mod, (fan_in, fan_out)) in enumerate(MODULES.items()):
            r = rank + i % 2  # uneven ranks exercise the block-diagonal fuse
            sd[f"{prefix}{stack}.{mod}.{down}.weight"] = torch.randn(r, fan_in, generator=g)
            sd[f"{prefix}{stack}.{mod}.{up}.weight"] = torch.randn(fan_out, r, generator=g)
            if alpha:
                sd[f"{prefix}{stack}.{mod}.alpha"] = torch.tensor(float(2 * r + i))
    return sd


def curve_lora(rank=4):
    g = torch.Generator().manual_seed(1)
    sd = {}
    for n in range(LAYERS):
        s = f"transformer.transformer_blocks.{n}.adaln_proj.linear"
        sd[f"{s}.curve_A.weight"] = torch.randn(rank, CURVE, generator=g)
        sd[f"{s}.curve_B.weight"] = torch.randn(18 * HID, rank, generator=g)
    return sd


def source_delta(sd, module, prefix="transformer.", down="lora_A", up="lora_B"):
    a, b = sd[f"{prefix}{module}.{down}.weight"].float(), sd[f"{prefix}{module}.{up}.weight"].float()
    alpha = sd.get(f"{prefix}{module}.alpha")
    return (b @ a) * (1.0 if alpha is None else alpha.item() / a.shape[0])


def expected(src, curve=None, **naming):
    """Native deltas written independently of the converter."""
    out = {}
    for stack, native in STACKS:
        d = lambda m: source_delta(src, f"{stack}.{m}", **naming)
        out[f"diffusion_model.{native}.attn.qkv_proj"] = torch.cat([d(m) for m in ("attn.to_q", "attn.to_k", "attn.to_v")])
        out[f"diffusion_model.{native}.attn.out_proj"] = d("attn.to_out.0")
        value, gate = d("ff.net.0.proj").chunk(2)
        out[f"diffusion_model.{native}.mlp.fc1"] = torch.cat([gate, value])
        out[f"diffusion_model.{native}.mlp.fc2"] = d("ff.net.2")
    for n in range(LAYERS if curve else 0):
        s = f"transformer.transformer_blocks.{n}.adaln_proj.linear"
        out[f"diffusion_model.blocks.{n}.adaln_proj.linear"] = curve[f"{s}.curve_B.weight"] @ curve[f"{s}.curve_A.weight"]
    return out


def delta(sd, module):
    return sd[f"{module}.lora_B.weight"].float() @ sd[f"{module}.lora_A.weight"].float()


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


class DiffusersH3LoraTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.dir = self.tmp.name
        self.src, self.curve = diffusers_lora(), curve_lora()
        save_file(self.src, self.path("lora"))
        save_file(self.curve, self.path("curve"))
        save_file(native_checkpoint(True), self.path("ckpt_curve"))
        save_file(native_checkpoint(False), self.path("ckpt_plain"))

    def tearDown(self):
        self.tmp.cleanup()

    def path(self, name):
        return os.path.join(self.dir, name + ".safetensors")

    def run_convert(self, out="out", curve=True, ckpt="ckpt_curve", **extra):
        payload = dict(adapter_path=self.path("lora"), output_path=self.path(out),
                       curve_path=self.path("curve") if curve else "", checkpoint_path=self.path(ckpt) if ckpt else "", **extra)
        events = list(run_h3_diffusers_lora(payload))
        log = "".join(e.get("text", "") for e in events)
        return (load_file(self.path(out)) if not extra.get("dry_run") else None), log, events

    def test_every_family_maps_and_deltas_match(self):
        out, log, events = self.run_convert()
        want = expected(self.src, self.curve)
        self.assertEqual({k.rsplit(".lora_", 1)[0] for k in out}, set(want))
        self.assertEqual(events[-1]["status"], "finished")
        self.assertTrue(all(t.dtype == torch.bfloat16 for t in out.values()))
        self.assertFalse(any(k.endswith(".alpha") for k in out))
        for module, d in want.items():
            self.assertLess(rel(delta(out, module), d), 1e-2, module)  # bf16 factors
        recipe = open(self.path("out") + ".txt").read()
        self.assertTrue(recipe.startswith("DaSiWa H3 Diffusers LoRA Conversion Recipe"))
        self.assertIn("MiniMax H3 Community License", recipe)

    def test_converted_curve_bakes_only_into_its_checkpoint_table(self):
        from safetensors import safe_open
        from core.h3_curve import table_sha256
        from core.lora_merge_engine import run_lora_merge

        checkpoint = native_checkpoint()
        checkpoint['adaln_t_table'] = torch.linspace(-1, 1, 33 * CURVE).reshape(33, CURVE)
        save_file(checkpoint, self.path('ckpt_curve'))
        for representation in ('exact', 'svd'):
            with self.subTest(representation=representation):
                name = 'bound_' + representation
                converted, _, _ = self.run_convert(name, representation=representation)
                payload = dict(base_path=self.path('ckpt_curve'), output_path=self.path('baked_' + representation),
                               architecture='MiniMax H3', loras=[dict(path=self.path(name))],
                               h3_turbo_complete=True, merge_device='cpu')
                events = list(run_lora_merge(payload))
                self.assertEqual(events[-1]['status'], 'finished')
                baked = load_file(payload['output_path'])
                for n in range(LAYERS):
                    module = f'blocks.{n}.adaln_proj.linear'
                    want = delta(converted, 'diffusion_model.' + module).bfloat16()
                    torch.testing.assert_close(baked[module + '.weight'], want)
                with safe_open(self.path(name), framework='pt') as handle:
                    self.assertEqual(handle.metadata()['adaln_coordinate_table_sha256'],
                                     table_sha256(checkpoint['adaln_t_table']))
                other = dict(checkpoint)
                other['adaln_t_table'] = checkpoint['adaln_t_table'].clone()
                other['adaln_t_table'][0, 0] += 1
                save_file(other, self.path('other'))
                with self.assertRaisesRegex(ValueError, 'gauge'):
                    list(run_lora_merge(dict(payload, base_path=self.path('other'),
                                             output_path=self.path('rejected_' + representation))))
                self.assertFalse(os.path.exists(self.path('rejected_' + representation)))

    def test_qkv_is_exact_block_diagonal_stack(self):
        out, _, _ = self.run_convert(curve=False, ckpt="")
        s, t = "transformer.transformer_blocks.1.attn", "diffusion_model.blocks.1.attn.qkv_proj"
        a = torch.cat([self.src[f"{s}.to_{p}.lora_A.weight"] for p in "qkv"])
        b = torch.block_diag(*[self.src[f"{s}.to_{p}.lora_B.weight"] for p in "qkv"])
        torch.testing.assert_close(out[f"{t}.lora_A.weight"], a.bfloat16(), rtol=0, atol=0)
        torch.testing.assert_close(out[f"{t}.lora_B.weight"], b.bfloat16(), rtol=0, atol=0)
        rq = self.src[f"{s}.to_q.lora_A.weight"].shape[0]
        self.assertTrue((out[f"{t}.lora_B.weight"][:INNER, rq:] == 0).all())

    def test_fc1_swaps_value_gate_halves(self):
        out, _, _ = self.run_convert(curve=False, ckpt="")
        b_src = self.src["transformer.transformer_blocks.0.ff.net.0.proj.lora_B.weight"]
        b_out = out["diffusion_model.blocks.0.mlp.fc1.lora_B.weight"].float()
        torch.testing.assert_close(b_out[:FFN], b_src[FFN:].bfloat16().float())
        torch.testing.assert_close(b_out[FFN:], b_src[:FFN].bfloat16().float())

    def test_prefix_naming_and_alpha_variants(self):
        for prefix, down, up in (("", "lora_down", "lora_up"), ("base_model.model.", "lora_A", "lora_B")):
            with self.subTest(prefix=prefix, down=down):
                src = diffusers_lora(prefix, down, up, alpha=True)
                save_file(src, self.path("lora"))
                name = "out_" + down
                out, _, _ = self.run_convert(name, curve=False)
                self.assertFalse(any(k.endswith(".alpha") for k in out))
                for module, d in expected(src, prefix=prefix, down=down, up=up).items():
                    self.assertLess(rel(delta(out, module), d), 1e-2, module)

    def test_unknown_orphan_missing_sibling_are_listed_errors_without_output(self):
        bad = dict(self.src, **{"transformer.proj_out.lora_A.weight": torch.zeros(3, HID),
                                "transformer.proj_out.lora_B.weight": torch.zeros(HID, 3)})
        del bad["transformer.transformer_blocks.0.ff.net.2.lora_B.weight"]
        del bad["transformer.transformer_blocks.1.attn.to_v.lora_A.weight"]
        del bad["transformer.transformer_blocks.1.attn.to_v.lora_B.weight"]
        save_file(bad, self.path("lora"))
        events = []
        with self.assertRaisesRegex(ValueError, "unsupported or invalid"):
            for e in run_h3_diffusers_lora(dict(adapter_path=self.path("lora"), output_path=self.path("out"), dry_run=True)):
                events.append(e)
        report = "".join(e.get("text", "") for e in events)
        self.assertIn("unsupported module: transformer.proj_out", report)
        self.assertIn("orphan factor or unknown key): transformer.transformer_blocks.0.ff.net.2.lora_A.weight", report)
        self.assertIn("blocks.1.attn.qkv_proj: missing q/k/v sibling(s) ['attn.to_v']", report)
        with self.assertRaises(ValueError):
            self.run_convert(curve=False)
        self.assertFalse(os.path.exists(self.path("out")))
        self.assertFalse(any(n.startswith(".h3_stage_") for n in os.listdir(self.dir)))

    def test_qkv_unequal_rows_rejected_even_when_fused_shape_matches(self):
        src = {}
        for p, rows in zip('qkv', (1, 2, 3)):
            module = f'transformer.transformer_blocks.0.attn.to_{p}'
            src[module + '.lora_A.weight'] = torch.ones(1, 2)
            src[module + '.lora_B.weight'] = torch.ones(rows, 1)
        save_file(src, self.path('lora'))
        save_file({'blocks.0.attn.qkv_proj.weight': torch.zeros(6, 2)}, self.path('ckpt_curve'))
        events = []
        with self.assertRaisesRegex(ValueError, 'unsupported or invalid'):
            for e in run_h3_diffusers_lora(dict(adapter_path=self.path('lora'),
                                               checkpoint_path=self.path('ckpt_curve'), dry_run=True)):
                events.append(e)
        self.assertIn('equal q/k/v output rows', ''.join(e.get('text', '') for e in events))
        self.assertFalse(os.path.exists(self.path('out')))

    def test_shape_mismatch_against_checkpoint_fails(self):
        ck = native_checkpoint(True)
        ck["blocks.0.mlp.fc2.weight"] = torch.zeros(HID, FFN + 1)
        save_file(ck, self.path("ckpt_curve"))
        with self.assertRaisesRegex(ValueError, "unsupported or invalid"):
            self.run_convert(curve=False)
        self.assertFalse(os.path.exists(self.path("out")))

    def test_already_native_input_is_rejected(self):
        save_file({"diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": torch.zeros(2, HID),
                   "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": torch.zeros(3 * INNER, 2)}, self.path("lora"))
        with self.assertRaisesRegex(ValueError, "already ComfyUI-native, nothing to convert"):
            self.run_convert(curve=False, ckpt="")

    def test_curve_skipped_on_non_curve_checkpoint(self):
        out, log, _ = self.run_convert(ckpt="ckpt_plain")
        self.assertIn("curve branches: target checkpoint has no adaln_t_table", log)
        self.assertFalse(any("adaln_proj" in k for k in out))

    def test_curve_requires_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "target checkpoint"):
            self.run_convert(ckpt="")

    def test_refuses_overwrite(self):
        with open(self.path("out"), "wb") as f:
            f.write(b"keep")
        with self.assertRaises(FileExistsError):
            self.run_convert()
        self.assertEqual(open(self.path("out"), "rb").read(), b"keep")

    def test_dry_run_reports_and_writes_nothing(self):
        _, log, events = self.run_convert(dry_run=True)
        self.assertEqual(events[-1]["status"], "dry-run complete")
        self.assertIn('"attn.qkv_proj": 3', log)
        self.assertIn("exact output", log)
        self.assertEqual(sorted(os.listdir(self.dir)), sorted(n + ".safetensors" for n in ("lora", "curve", "ckpt_curve", "ckpt_plain")))

    def test_svd_rerank_is_near_exact_at_sufficient_rank_and_reports(self):
        out, log, _ = self.run_convert(representation="svd", energy=1.0)
        for module, d in expected(self.src, self.curve).items():
            self.assertLess(rel(delta(out, module), d), 1e-2, module)
        self.assertIn("median_relative_error", log)
        recipe = open(self.path("out") + ".txt").read()
        self.assertIn('"per_module"', recipe)
        capped, log, _ = self.run_convert("capped", curve=False, representation="svd", max_rank=2)
        self.assertEqual(capped["diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight"].shape[0], 2)
        self.assertIn('"at_rank_cap"', log)


if __name__ == "__main__":
    unittest.main()
