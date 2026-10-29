import tempfile
import unittest
from pathlib import Path

from compiler.inductor_capture import _launch_sites, _ttir_function_symbols


class InductorArtifactLinkTests(unittest.TestCase):
    def test_generated_launch_symbol_links_to_dumped_ttir(self):
        generated_wrapper = [
            "triton_poi_fused_add_relu_0.run(x, y, out, grid=grid(1));\n"
        ]
        sites = _launch_sites(generated_wrapper)
        self.assertEqual([site["kernel_symbol"] for site in sites], [
            "triton_poi_fused_add_relu_0"
        ])
        self.assertEqual(sites[0]["triton_function_symbol"], "triton_poi_fused_add_relu_0")

        with tempfile.TemporaryDirectory() as tmp:
            ttir = Path(tmp) / "kernel.ttir"
            ttir.write_text(
                "module { tt.func public @triton_poi_fused_add_relu_0() }\n"
            )
            self.assertEqual(
                _ttir_function_symbols(ttir), {sites[0]["kernel_symbol"]}
            )

    def test_custom_kernel_launch_uses_explicit_compiler_binding(self):
        sites = _launch_sites([
            "flash_attention_kernel_0 = async_compile.triton(\n"
            "    'flash_attention_kernel', 'source', device_str='cuda')\n"
            "flash_attention_kernel_0.run(q, k, v, out)\n"
            "unbound_kernel_0.run(out)\n"
        ])
        self.assertEqual(sites[0]["kernel_symbol"], "flash_attention_kernel_0")
        self.assertEqual(sites[0]["triton_function_symbol"], "flash_attention_kernel")
        # A similar suffix without a compiler binding is not an association.
        self.assertEqual(sites[1]["triton_function_symbol"], "unbound_kernel_0")


if __name__ == "__main__":
    unittest.main()
