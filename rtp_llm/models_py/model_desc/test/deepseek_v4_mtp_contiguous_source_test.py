import ast
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "deepseek_v4_mtp_model.py"


class DeepSeekV4MtpContiguousSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        cls.model_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DeepSeekV4MtpModel"
        )

    def assert_contiguous_before_enorm(self, method_name: str, target: str) -> None:
        method = next(
            node
            for node in self.model_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method_name
        )
        assignments = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(assignment_target, ast.Name)
                and assignment_target.id == target
                for assignment_target in node.targets
            )
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "contiguous"
        ]
        self.assertEqual(len(assignments), 1)

        assignment = assignments[0]
        where_call = assignment.value.func.value
        self.assertIsInstance(where_call, ast.Call)
        self.assertIsInstance(where_call.func, ast.Attribute)
        self.assertIsInstance(where_call.func.value, ast.Name)
        self.assertEqual(where_call.func.value.id, "torch")
        self.assertEqual(where_call.func.attr, "where")

        norm_calls = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr == "enorm"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == target
        ]
        self.assertEqual(len(norm_calls), 1)
        self.assertLess(assignment.lineno, norm_calls[0].lineno)

    def test_chunked_embedding_is_contiguous_before_enorm(self) -> None:
        self.assert_contiguous_before_enorm("_build_fused_chunked", "embed_chunk")

    def test_non_chunked_embedding_is_contiguous_before_enorm(self) -> None:
        self.assert_contiguous_before_enorm("_build_fused", "inputs_embeds")


if __name__ == "__main__":
    unittest.main(verbosity=2)
