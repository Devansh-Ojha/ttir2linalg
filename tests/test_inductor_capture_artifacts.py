import tempfile
import unittest
import json
import os
from pathlib import Path
from types import SimpleNamespace
from io import StringIO
from unittest.mock import patch

from compiler.inductor_capture import (
    _configure_capture_environment, _launch_sites, _prepare_capture_directory,
    _ttir_function_symbols, capture_inductor,
    main as capture_main,
)
from compiler.inductor_to_atlas import map_capture


class InductorArtifactLinkTests(unittest.TestCase):
    def test_aot_api_is_enabled_and_caches_remain_enabled(self):
        import torch._dynamo.config as dynamo_config
        import torch._inductor.config as inductor_config
        import torch

        sample = object()
        calls = []

        def precompile(inputs):
            self.assertFalse(torch.is_grad_enabled())
            calls.append(inputs)

        def compile_model(model, **kwargs):
            self.assertEqual(kwargs, {"backend": "inductor", "fullgraph": True})
            self.assertTrue(dynamo_config.enable_aot_compile)
            self.assertFalse(inductor_config.force_disable_caches)
            self.assertEqual(dynamo_config.debug_dir_root, os.environ["TORCH_COMPILE_DEBUG_DIR"])
            return SimpleNamespace(forward=SimpleNamespace(aot_compile=precompile))

        # This tests API configuration, not genuine compilation or numerical output.
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ), patch(
                "torch.backends.cuda.is_built", return_value=True), patch(
                "compiler.inductor_capture._make_fake_cuda_model",
                return_value=(object(), sample)), patch(
                "torch.export.export", side_effect=RuntimeError("context graph intentionally unavailable")), patch(
                "torch.compile", side_effect=compile_model), dynamo_config.patch(
                enable_aot_compile=False), inductor_config.patch(force_disable_caches=True):
            report = capture_inductor(out_dir=Path(tmp) / "capture")
            self.assertFalse(dynamo_config.enable_aot_compile)
            self.assertTrue(inductor_config.force_disable_caches)
        self.assertEqual(calls, [((sample,), {})])
        self.assertIsNone(report["inductor_error"])
        self.assertEqual(report["inductor_status"], "code_capture_failed")
        self.assertEqual(report["inductor_ttir_artifacts"], [])

    def test_cli_failure_has_nonzero_exit_status(self):
        with patch("sys.argv", ["inductor_capture"]), patch(
                "compiler.inductor_capture.capture_inductor",
                return_value={"inductor_status": "failed"}), patch("sys.stdout", StringIO()):
            with self.assertRaises(SystemExit) as error:
                capture_main()
            self.assertEqual(error.exception.code, 1)

    def test_capture_directory_is_absolute_and_never_reused(self):
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            relative = Path(tmp) / "capture"
            capture = _prepare_capture_directory(relative)
            self.assertTrue(capture.is_absolute())
            sentinel = capture / "output_code.py"
            sentinel.write_text("old evidence")
            with self.assertRaises(FileExistsError):
                _prepare_capture_directory(relative)
            self.assertEqual(sentinel.read_text(), "old evidence")

    def test_dump_and_required_caches_are_invocation_local(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            first = _prepare_capture_directory(Path(tmp) / "first")
            second = _prepare_capture_directory(Path(tmp) / "second")
            for directory in (first, second):
                dump = _configure_capture_environment(directory, "aot-cuda")
                self.assertTrue(dump.is_relative_to(directory))
                self.assertEqual(os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"], "0")
                for key in ("TRITON_DUMP_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
                            "TORCH_COMPILE_DEBUG_DIR"):
                    self.assertTrue(Path(os.environ[key]).is_relative_to(directory))
                self.assertEqual(os.environ["TRITON_ALWAYS_COMPILE"], "1")

    def test_absolute_capture_relative_and_legacy_cwd_paths(self):
        # These are synthetic path fixtures, not evidence of an Inductor compile.
        for spelling in ("absolute", "capture_relative", "legacy_cwd"):
            with self.subTest(spelling=spelling), tempfile.TemporaryDirectory(dir=".") as tmp:
                directory = (Path(tmp) / "capture").resolve()
                source = directory / "inductor_ttir" / "kernel.ttir"
                source.parent.mkdir(parents=True)
                source.write_text("synthetic path fixture")
                path = {
                    "absolute": source,
                    "capture_relative": source.relative_to(directory),
                    "legacy_cwd": source.relative_to(Path.cwd()),
                }[spelling]
                (directory / "capture.json").write_text(json.dumps({
                    "model": "test",
                    "inductor_wrapper_referenced_ttir_artifacts": [str(path)],
                }))
                with patch("compiler.inductor_to_atlas.convert_file", return_value="test-only") as convert:
                    summary = map_capture(directory)
                self.assertEqual(convert.call_args.args[0], source)
                self.assertEqual(summary["mapped_kernel_count"], 1)
                self.assertEqual(summary["kernels"][0]["source_ttir"], str(source))

    def test_missing_dump_stays_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "capture.json").write_text(json.dumps({
                "inductor_wrapper_referenced_ttir_artifacts": ["missing.ttir"],
            }))
            with patch("compiler.inductor_to_atlas.convert_file") as convert:
                summary = map_capture(directory)
            convert.assert_not_called()
            self.assertEqual(summary["unsupported_kernel_count"], 1)
            self.assertIn("missing", summary["kernels"][0]["error"])

    def test_unassociated_ttir_and_hgraph_are_never_substituted(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "capture.json").write_text(json.dumps({
                "hgraph_json_artifact": "workload.json",
                "inductor_unassociated_ttir_artifacts": ["unassociated.ttir"],
                "inductor_wrapper_referenced_ttir_artifacts": [],
            }))
            with patch("compiler.inductor_to_atlas.convert_file") as convert:
                summary = map_capture(directory)
            convert.assert_not_called()
            self.assertEqual(summary["input_status"], "no_associated_ttir")
            self.assertEqual(summary["mapped_kernel_count"], 0)
            self.assertEqual(summary["unassociated_ttir_artifacts"], ["unassociated.ttir"])
            self.assertTrue((directory / "atlas/mapping.json").is_file())
            with self.assertRaises(FileExistsError):
                map_capture(directory)

    def test_failed_recognizers_retain_all_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "kernel.ttir").write_text("synthetic unsupported kernel")
            (directory / "capture.json").write_text(json.dumps({
                "inductor_wrapper_referenced_ttir_artifacts": ["kernel.ttir"],
            }))
            with patch("compiler.inductor_to_atlas.convert_file",
                       side_effect=NotImplementedError("unsupported source operation")):
                summary = map_capture(directory)
            self.assertEqual(summary["unsupported_kernel_count"], 1)
            self.assertEqual(len(summary["kernels"][0]["attempts"]), 4)
            self.assertTrue(all("unsupported source operation" in attempt["reason"]
                                for attempt in summary["kernels"][0]["attempts"]))
            self.assertFalse(list((directory / "atlas").glob("*.mlir")))

    def test_setup_failure_still_writes_evidence_reports(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ), patch(
                "compiler.inductor_capture._make_fake_cuda_model",
                side_effect=RuntimeError("test device unavailable")), patch(
                "torch.backends.cuda.is_built", return_value=True):
            report = capture_inductor(out_dir=Path(tmp) / "capture")
            self.assertEqual(report["inductor_status"], "failed")
            self.assertIn("test device unavailable", report["setup_error"])
            self.assertFalse(report["atlas_virtual_ops_emitted"])
            self.assertEqual(report["inductor_ttir_artifacts"], [])
            mapping = json.loads(Path(report["atlas_mapping_report"]).read_text())
            self.assertEqual(mapping["input_status"], "no_associated_ttir")

    def test_generated_launch_symbol_links_to_dumped_ttir(self):
        generated_wrapper = [
            "triton_poi_fused_add_relu_0.run(x, y, out, grid=grid(1));\n"
        ]
        sites = _launch_sites(generated_wrapper)
        self.assertEqual([site["kernel_symbol"] for site in sites], [
            "triton_poi_fused_add_relu_0"
        ])

        with tempfile.TemporaryDirectory() as tmp:
            ttir = Path(tmp) / "kernel.ttir"
            ttir.write_text(
                "module { tt.func public @triton_poi_fused_add_relu_0() }\n"
            )
            self.assertEqual(
                _ttir_function_symbols(ttir), {sites[0]["kernel_symbol"]}
            )


if __name__ == "__main__":
    unittest.main()
