"""Tests for the DSV4 runtime switch surface (``runtime_config.py``).

Runs with plain ``python3 test_runtime_config.py`` — no torch, no bazel —
by loading ``runtime_config.py`` straight from its source path (the package
``__init__`` chain pulls heavy deps that do not exist outside the serving
container).  The wheel-layout case additionally proves the module imports
from a package-root tree shaped like the deployed wheel, which is the
empty-switch failure mode this surface exists to prevent (ROADMAP 3.1:
files under ``package-root/internal_source/...`` that the import path never
sees).
"""

import importlib
import importlib.util
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path

_DSV4_DIR = Path(__file__).resolve().parents[1]
RC_PATH = _DSV4_DIR / "runtime_config.py"
DEEPEP_PATH = _DSV4_DIR / "moe" / "strategies" / "deepep.py"

_SWITCHES_UNDER_TEST = (
    "DSV4_UNITTEST_PROBE",
    "DSV4_UNITTEST_BOOL",
    "DSV4_UNITTEST_INT",
    "DSV4_UNITTEST_PCTL",
    "DSV4_UNITTEST_WHEEL_PROBE",
)


def _load_rc():
    """Load a fresh runtime_config module (independent one-shot caches)."""
    spec = importlib.util.spec_from_file_location(
        "rtp_llm.models_py.modules.dsv4.runtime_config", RC_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _EnvSandbox:
    """Save/restore every switch the suite touches."""

    def __init__(self, env=None):
        self._saved = {k: os.environ.get(k) for k in _SWITCHES_UNDER_TEST}
        for k, v in (env or {}).items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


class RuntimeConfigTest(unittest.TestCase):
    def setUp(self):
        self._sandbox = _EnvSandbox()
        self._sandbox.__enter__()
        self.rc = _load_rc()

    def tearDown(self):
        self._sandbox.__exit__(None, None, None)

    def test_unset_switch_returns_default(self):
        with _EnvSandbox({"DSV4_UNITTEST_PROBE": None}):
            self.assertEqual(self.rc.get_switch("DSV4_UNITTEST_PROBE", 7), 7)

    def test_env_value_wins_and_is_read_exactly_once(self):
        with _EnvSandbox({"DSV4_UNITTEST_PROBE": "41"}):
            self.assertEqual(
                self.rc.get_switch(
                    "DSV4_UNITTEST_PROBE", 0, self.rc.parse_nonneg_int
                ),
                41,
            )
            # The one-shot cache must ignore later env mutations: prod never
            # re-reads the env on the hot path.
            os.environ["DSV4_UNITTEST_PROBE"] = "9999"
            self.assertEqual(
                self.rc.get_switch(
                    "DSV4_UNITTEST_PROBE", 0, self.rc.parse_nonneg_int
                ),
                41,
            )

    def test_invalid_value_fails_loud_with_readable_message(self):
        with _EnvSandbox({"DSV4_UNITTEST_INT": "not-a-number"}):
            with self.assertRaises(ValueError) as ctx:
                self.rc.get_switch("DSV4_UNITTEST_INT", 0, self.rc.parse_nonneg_int)
            msg = str(ctx.exception)
            # A readable failure names the switch, the raw value, the default.
            self.assertIn("DSV4_UNITTEST_INT", msg)
            self.assertIn("not-a-number", msg)
            self.assertIn("0", msg)

    def test_negative_int_is_rejected(self):
        with _EnvSandbox({"DSV4_UNITTEST_INT": "-3"}):
            with self.assertRaises(ValueError):
                self.rc.get_switch("DSV4_UNITTEST_INT", 0, self.rc.parse_nonneg_int)

    def test_audit_log_prints_once_per_switch(self):
        with self.assertLogs(level=logging.INFO) as logs:
            self.rc.get_switch("DSV4_UNITTEST_PROBE", 3)
            self.rc.get_switch("DSV4_UNITTEST_PROBE", 3)
            self.rc.get_switch("DSV4_UNITTEST_PROBE", 3)
        lines = [r for r in logs.output if "[DSV4_CONFIG]" in r]
        self.assertEqual(len(lines), 1)
        self.assertIn("[DSV4_CONFIG] DSV4_UNITTEST_PROBE=3", lines[0])
        self.assertIn("default", lines[0])

    def test_audit_log_marks_env_sourced_values(self):
        with _EnvSandbox({"DSV4_UNITTEST_PROBE": "5"}):
            with self.assertLogs(level=logging.INFO) as logs:
                self.rc.get_switch(
                    "DSV4_UNITTEST_PROBE", 3, self.rc.parse_nonneg_int
                )
        self.assertTrue(
            any(
                "[DSV4_CONFIG] DSV4_UNITTEST_PROBE=5" in r and "(env)" in r
                for r in logs.output
            )
        )

    def test_unknown_switch_without_default_is_a_typo(self):
        with self.assertRaises(KeyError):
            self.rc.get_switch("DSV4_UNITTEST_NEVER_REGISTERED")

    def test_register_switch_validates_at_registration_time(self):
        with _EnvSandbox({"DSV4_UNITTEST_INT": "bogus"}):
            with self.assertRaises(ValueError):
                self.rc.register_switch(
                    "DSV4_UNITTEST_INT", 0, self.rc.parse_nonneg_int
                )

    def test_registered_switch_can_be_fetched_without_default(self):
        with _EnvSandbox({"DSV4_UNITTEST_PROBE": None}):
            self.rc.register_switch("DSV4_UNITTEST_PROBE", 9)
            self.assertEqual(self.rc.get_switch("DSV4_UNITTEST_PROBE"), 9)

    def test_parse_bool_variants(self):
        parse = self.rc.parse_bool
        for truthy in ("1", "true", "ON", "Yes"):
            self.assertTrue(parse(truthy, False))
        for falsy in ("0", "false", "OFF", "no"):
            self.assertFalse(parse(falsy, True))
        with self.assertRaises(ValueError):
            parse("perhaps", False)

    def test_parse_percentile_list_variants(self):
        parse = self.rc.parse_percentile_list
        self.assertEqual(parse("p50,p99", ()), (50, 99))
        self.assertEqual(parse(" 50,99 ", ()), (50, 99))
        self.assertEqual(parse("p100", ()), (100,))
        # Empty means "off" — the current behaviour of every consumer.
        self.assertEqual(parse("", ()), ())
        self.assertEqual(parse("   ", ()), ())
        for bad in ("p101", "p-1", "latency", "p50,p"):
            with self.assertRaises(ValueError):
                parse(bad, ())

    def test_module_lives_on_the_wheel_import_path(self):
        # The empty-switch lesson (ROADMAP 3.1): a reader under
        # internal_source/ that package-root/rtp_llm/... does not carry is a
        # silent no-op.  runtime_config must sit on the real import path.
        resolved = str(Path(self.rc.__file__).resolve())
        self.assertIn(
            os.path.join("rtp_llm", "models_py", "modules", "dsv4"), resolved
        )
        self.assertNotIn("internal_source", resolved)

    def test_importable_from_wheel_package_root_layout(self):
        """Import the module from a tree shaped like the deployed wheel.

        Simulates ``package-root/rtp_llm/models_py/modules/dsv4/runtime_config.py``
        (empty ``__init__.py`` chain) on a clean sys.path, exactly the layout
        the engine loads in production.  Import failure here == another
        empty-switch incident.
        """
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "rtp_llm" / "models_py" / "modules" / "dsv4"
            pkg.mkdir(parents=True)
            (pkg / "runtime_config.py").write_bytes(RC_PATH.read_bytes())
            base = Path(tmp) / "rtp_llm"
            (base / "__init__.py").write_text("")
            for sub in ("models_py", "modules", "dsv4"):
                base = base / sub
                (base / "__init__.py").write_text("")

            saved_path = list(sys.path)
            saved_rtp = {
                k: v
                for k, v in sys.modules.items()
                if k == "rtp_llm" or k.startswith("rtp_llm.")
            }
            for k in saved_rtp:
                del sys.modules[k]
            sys.path.insert(0, tmp)
            try:
                mod = importlib.import_module(
                    "rtp_llm.models_py.modules.dsv4.runtime_config"
                )
                # The freshly imported copy must be functional, not just
                # importable: resolve a switch through it.
                with _EnvSandbox({"DSV4_UNITTEST_WHEEL_PROBE": "13"}):
                    self.assertEqual(
                        mod.get_switch(
                            "DSV4_UNITTEST_WHEEL_PROBE", 0,
                            mod.parse_nonneg_int,
                        ),
                        13,
                    )
            finally:
                sys.path[:] = saved_path
                for k in [
                    k
                    for k in sys.modules
                    if k == "rtp_llm" or k.startswith("rtp_llm.")
                ]:
                    del sys.modules[k]
                sys.modules.update(saved_rtp)

    def test_deepep_registers_three_switches_with_legacy_defaults(self):
        """The three new switches exist and default to legacy behaviour."""
        text = DEEPEP_PATH.read_text()
        for name, default in (
            ("DSV4_MOE_OVERFLOW_REPLAY_LOG_EVERY", "0"),
            ("DSV4_MOE_ROUTER_PERCENTILES", "()"),
            ("DSV4_MOE_ROUTER_PERCENTILE_LOG_EVERY", "0"),
        ):
            self.assertIn(f'"{name}"', text, f"{name} missing from deepep.py")
        # Defaults on the get_switch calls must be 0 / () / 0 — anything else
        # would change default decode behaviour.
        self.assertIn('"DSV4_MOE_OVERFLOW_REPLAY_LOG_EVERY", 0', text)
        self.assertIn('"DSV4_MOE_ROUTER_PERCENTILES", ()', text)
        self.assertIn('"DSV4_MOE_ROUTER_PERCENTILE_LOG_EVERY", 0', text)
        # And the switches must resolve through runtime_config, not ad-hoc
        # os.environ reads (the strangler entry point).
        self.assertIn("get_switch(", text)

    def test_deprecated_overflow_switch_name_is_gone(self):
        """The pre-review name must not linger anywhere.

        ``DSV4_MOE_OVERFLOW_LOG_EVERY`` differs from the legacy
        ``DSV4_MOE_CAPACITY_OVERFLOW_LOG_EVERY`` by a single word; exporting
        the old name would silently do nothing.  The rename review item
        demanded grep-clean code, so pin it here.
        """
        for path in (DEEPEP_PATH, RC_PATH):
            self.assertNotIn(
                "DSV4_MOE_OVERFLOW_LOG_EVERY",
                path.read_text(),
                f"stale pre-rename switch name still referenced in {path}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
