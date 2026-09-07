"""Source-only contract for wiring the masked SwiGLU+MXFP4 kernel into DeepEP LL.

The measured version of this path loaded a kernel from an out-of-repo ``.so``
that SGLang's JIT had built, and pre-allocated the packed payload and the scale
in Python.  The in-tree kernel allocates both itself and is reached through
``rtp_llm_ops``, so the wheel carries the whole path.  These assertions keep the
JIT loader from creeping back and keep the call on the shapes the kernel reads:
the grouped ``[E, capacity, 2H]`` tensor and the clamped per-expert counts.

Substring checks would pass on a call that had the right tokens in the wrong
argument positions, so the call itself is inspected through the AST.
"""

import ast
from pathlib import Path
import unittest

_SOURCE_PATH = (
    Path(__file__).resolve().parents[1] / "moe" / "strategies" / "deepep.py"
)

_MASKED_OP = "ppu_silu_and_mul_masked_post_quant_mxfp4"
_UNMASKED_OP = "ppu_silu_and_mul_post_quant_mxfp4"


def _tree():
    return ast.parse(_SOURCE_PATH.read_text())


def _calls(tree, attr):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
    ]


def _module_assign(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
    raise AssertionError("module-level %s not found" % name)


class MaskedSiluCallContractTest(unittest.TestCase):
    def test_reached_through_the_native_op_registry(self):
        calls = _calls(_tree(), _MASKED_OP)
        self.assertEqual(len(calls), 1)
        owner = calls[0].func.value
        self.assertIsInstance(owner, ast.Name)
        self.assertEqual(owner.id, "rtp_llm_ops")

    def test_called_on_the_grouped_tensor_and_the_clamped_counts(self):
        # The kernel indexes rows by expert, so it needs the [E, capacity, 2H]
        # tensor -- not the flat ``gate_up`` view the unmasked op takes -- and it
        # reads row counts straight off the device, so they have to be the
        # already-clamped ``safe_counts`` rather than the raw dispatch counts.
        call = _calls(_tree(), _MASKED_OP)[0]
        self.assertEqual(call.keywords, [])
        self.assertEqual(
            [a.id for a in call.args if isinstance(a, ast.Name)],
            ["gate_up_grouped", "safe_counts", "swiglu_limit"],
        )
        self.assertEqual(len(call.args), 4)
        hint = call.args[3]
        self.assertIsInstance(hint, ast.IfExp)
        self.assertEqual(hint.test.id, "_BUCKET_EXPECTED_M")
        self.assertEqual(hint.body.id, "expected_m")
        self.assertEqual(hint.orelse.id, "compute_capacity")

    def test_the_op_allocates_both_outputs(self):
        # The measured patch pre-allocated the payload and the scale in Python
        # and passed them in.  Two sources of truth for the padded scale width is
        # how the buffer ends up narrower than what the kernel writes.
        tree = _tree()
        call = _calls(tree, _MASKED_OP)[0]
        parent = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and node.value is call:
                parent = node
        self.assertIsNotNone(parent)
        self.assertEqual(len(parent.targets), 1)
        target = parent.targets[0]
        self.assertIsInstance(target, ast.Tuple)
        self.assertEqual(
            [e.id for e in target.elts],
            ["hidden_fp4_grouped", "hidden_scale_grouped"],
        )

    def test_scale_is_consumed_as_returned(self):
        # [E, S, M] is the kernel's native scale layout and the op hands back the
        # mn-major view for free; re-materializing it would spend the win.
        text = _SOURCE_PATH.read_text()
        _, _, after_call = text.partition("rtp_llm_ops." + _MASKED_OP)
        branch, _, _ = after_call.partition("        else:")
        self.assertNotIn(".contiguous()", branch)
        self.assertNotIn("permute", branch)
        self.assertNotIn("as_strided", branch)

    def test_unmasked_path_survives(self):
        # The masked kernel is a switch, not a replacement: it has never run on
        # this device outside a bench harness.
        self.assertEqual(len(_calls(_tree(), _UNMASKED_OP)), 1)


class MaskedSiluGatingTest(unittest.TestCase):
    def test_off_unless_asked_for(self):
        source = ast.unparse(_module_assign(_tree(), "_MASKED_SILU"))
        self.assertIn("DSV4_MOE_MASKED_SILU", source)
        self.assertIn("'0'", source)

    def test_block_width_matches_the_kernel(self):
        # kPpuSiluMulMaskedMxfp4BlockN in the kernel header.  A wider padding
        # there and a narrower guard here is a silent stride mismatch.
        value = _module_assign(_tree(), "_MASKED_SILU_BLOCK_N")
        self.assertEqual(ast.literal_eval(value), 256)

    def test_odd_inter_falls_back_instead_of_guessing(self):
        text = _SOURCE_PATH.read_text()
        self.assertIn("if use_masked_silu and inter % _MASKED_SILU_BLOCK_N:", text)
        self.assertIn("use_masked_silu = False", text)

    def test_no_out_of_repo_loader(self):
        # Identifiers, attributes, imports and string literals only.  Comments
        # are excluded on purpose: the unmasked branch credits the SGLang runner
        # it was ported from, and that attribution is not a dependency.
        tokens = []
        for node in ast.walk(_tree()):
            if isinstance(node, ast.Name):
                tokens.append(node.id)
            elif isinstance(node, ast.Attribute):
                tokens.append(node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                tokens.append(node.value)
            elif isinstance(node, ast.Import):
                tokens.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                tokens.append(node.module or "")
        code = "\n".join(tokens).lower()
        for banned in ("tvm_ffi", "load_module", ".so", "masked_silu_so", "sglang"):
            self.assertNotIn(banned, code)

    def test_coupling_to_no_compact_stays_documented(self):
        # Enabling no-compact without this kernel took the activation from
        # 27.8 us to 436.1 us.  Whoever reads these two flags next needs to see
        # that they are one candidate before flipping either alone.
        text = _SOURCE_PATH.read_text()
        note = text[: text.index("_MASKED_SILU = os.environ.get")]
        for number in ("27.8 us", "436.1 us", "13.8 us"):
            self.assertIn(number, note)
        self.assertIn("_LL_NO_COMPACT", note)


if __name__ == "__main__":
    unittest.main()
