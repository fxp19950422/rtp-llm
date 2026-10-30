import ast
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parent


class PackageContractTest(unittest.TestCase):
    def test_local_include_closure(self):
        for path in list((ROOT / "csrc").rglob("*")) + list((ROOT / "deep_gemm_mega/include").rglob("*")):
            if not path.is_file():
                continue
            for quoted, angle in re.findall(r'#\s*include\s*(?:"([^"]+)"|<([^>]+)>)', path.read_text()):
                name = quoted or angle
                if name.startswith("deep_gemm_mega/"):
                    self.assertTrue((ROOT / "deep_gemm_mega/include" / name).is_file(), (path, name))
                if quoted.startswith("../") or (quoted and "/" not in quoted):
                    self.assertTrue((path.parent / quoted).is_file(), (path, quoted))

    def test_isolated_imports_and_jit(self):
        init = (ROOT / "deep_gemm_mega/__init__.py").read_text()
        self.assertLess(init.index("import torch"), init.index("from . import _C"))
        for path in (ROOT / "deep_gemm_mega").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    self.assertFalse(any(n.name.split(".")[0] == "deep_gemm" for n in node.names))
                elif isinstance(node, ast.ImportFrom):
                    self.assertNotEqual((node.module or "").split(".")[0], "deep_gemm")
        compiler = (ROOT / "csrc/jit/compiler.hpp").read_text()
        self.assertIn('".deep_gemm_mega"', compiler)
        self.assertIn('"DG_MEGA_JIT_CACHE_DIR"', compiler)
        self.assertNotIn('"DG_JIT_CACHE_DIR"', compiler)
        parser = (ROOT / "csrc/jit/include_parser.hpp").read_text()
        self.assertIn('filename.rfind("deep_gemm_mega/", 0) == 0', parser)
        lut = (ROOT / "csrc/jit_kernels/heuristics/runtime_lut.hpp").read_text()
        self.assertIn('"RTP_PPU_MEGA_LUT_PATH"', lut)
        self.assertNotIn('"SAIL_DEEPGEMM_LUT_PATH"', lut)

    def test_only_mega_api_registered(self):
        api = (ROOT / "csrc/python_api.cpp").read_text()
        self.assertLess(api.index("<hggc_runtime.h>"), api.index('"apis/mega_moe.hpp"'))
        self.assertNotIn('"apis/gemm.hpp"', api)
        self.assertNotIn('"apis/attention.hpp"', api)
        self.assertIn('deep_gemm_mega::mega::register_apis', api)


if __name__ == "__main__":
    unittest.main()
