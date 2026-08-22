import unittest
from types import SimpleNamespace
from unittest import mock

from rtp_llm.utils import import_util


class OptionalInternalEntrypointTest(unittest.TestCase):
    def test_absent_internal_source_is_a_noop(self):
        with mock.patch.object(import_util, "has_internal_source", return_value=False):
            with mock.patch.object(import_util.importlib, "import_module") as imported:
                self.assertFalse(
                    import_util.call_optional_internal_source_entrypoint(
                        "models.runtime_init", "before_model_construction"
                    )
                )
        imported.assert_not_called()

    def test_calls_present_hook_with_exact_arguments(self):
        callback = mock.Mock()
        module = SimpleNamespace(before_model_construction=callback)
        with mock.patch.object(import_util, "has_internal_source", return_value=True):
            with mock.patch.object(
                import_util.importlib, "import_module", return_value=module
            ) as imported:
                self.assertTrue(
                    import_util.call_optional_internal_source_entrypoint(
                        "models.runtime_init",
                        "before_model_construction",
                        "positional",
                        ep_size=1,
                    )
                )
        imported.assert_called_once_with("internal_source.rtp_llm.models.runtime_init")
        callback.assert_called_once_with("positional", ep_size=1)

    def test_absent_optional_module_is_a_noop(self):
        error = ModuleNotFoundError("optional module absent")
        error.name = "internal_source.rtp_llm.models.runtime_init"
        with mock.patch.object(import_util, "has_internal_source", return_value=True):
            with mock.patch.object(
                import_util.importlib, "import_module", side_effect=error
            ):
                self.assertFalse(
                    import_util.call_optional_internal_source_entrypoint(
                        "models.runtime_init", "before_model_construction"
                    )
                )

    def test_nested_import_failure_propagates(self):
        error = ModuleNotFoundError("required dependency absent")
        error.name = "required_dependency"
        with mock.patch.object(import_util, "has_internal_source", return_value=True):
            with mock.patch.object(
                import_util.importlib, "import_module", side_effect=error
            ):
                with self.assertRaises(ModuleNotFoundError) as raised:
                    import_util.call_optional_internal_source_entrypoint(
                        "models.runtime_init", "before_model_construction"
                    )
        self.assertIs(raised.exception, error)

    def test_present_hook_must_be_callable(self):
        module = SimpleNamespace(before_model_construction=None)
        with mock.patch.object(import_util, "has_internal_source", return_value=True):
            with mock.patch.object(
                import_util.importlib, "import_module", return_value=module
            ):
                with self.assertRaisesRegex(TypeError, "is not callable"):
                    import_util.call_optional_internal_source_entrypoint(
                        "models.runtime_init", "before_model_construction"
                    )


if __name__ == "__main__":
    unittest.main()
