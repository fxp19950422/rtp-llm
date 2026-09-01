"""Guard against py_test targets that report PASSED without running a case.

bazel executes a py_test by running its main source as a script.  A source that
only defines ``unittest.TestCase`` subclasses -- or bare pytest functions -- and
carries no entrypoint therefore imports, defines, and exits 0: bazel prints
PASSED while zero assertions ran.  Three targets here were green that way
(``test_ll_no_compact``, ``test_target_verify_contract``,
``test_varlen_prefill_oracle``), and the second was additionally hiding a
missing ``data`` dependency that surfaced the moment its cases actually ran.

Scanning the whole repo from inside a test is not possible: bazel only stages
declared inputs, and ``glob()`` cannot cross package boundaries.  The three
affected packages are wired in through ``data`` instead -- this one by glob, the
other two by filegroup.  ``test_scope_is_populated`` fails loudly if one of
those inputs ever goes missing, because an empty scan is the same failure mode
this file exists to catch.
"""

import os
import re
import unittest

# A source defines cases if it has ``def test_...`` at any indent level.
_DEFINES_CASES = re.compile(r"^[ \t]*def test_", re.MULTILINE)
# bazel only ever runs the source as a script, so the one thing every test source
# needs is a ``__main__`` entrypoint.  What goes inside it varies by author --
# ``unittest.main()``, ``pytest.main()``, a hand-written list of calls, and a
# reflective loop over module globals are all in use in these packages -- so the
# guard deliberately does not police the style, only that an entrypoint exists.
_HAS_ENTRYPOINT = re.compile(r"^if __name__ == [\"']__main__[\"']\s*:", re.MULTILINE)

# Relative to the inner ``rtp_llm`` package root inside runfiles.
_SCANNED_PACKAGES = (
    "models_py/modules/dsv4/test",
    "models_py/modules/dsv4/fp8/test",
    "models_py/speculative/test",
)


def _package_root():
    # <runfiles>/rtp_llm/rtp_llm/models_py/modules/dsv4/test/<this file>
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "..", ".."))


def _scanned_dirs():
    root = _package_root()
    return [os.path.join(root, package) for package in _SCANNED_PACKAGES]


def _test_sources(directory):
    if not os.path.isdir(directory):
        return []
    return [
        name
        for name in sorted(os.listdir(directory))
        if name.endswith(".py")
        and (name.startswith("test_") or name.endswith("_test.py"))
    ]


class PyTestEntrypointGuard(unittest.TestCase):
    def test_scope_is_populated(self):
        for directory in _scanned_dirs():
            self.assertTrue(
                os.path.isdir(directory), "not staged into runfiles: %s" % directory
            )
            self.assertTrue(
                _test_sources(directory), "no test sources under %s" % directory
            )

    def test_every_test_source_drives_a_runner(self):
        offenders = []
        for directory in _scanned_dirs():
            for name in _test_sources(directory):
                path = os.path.join(directory, name)
                with open(path, encoding="utf-8") as handle:
                    text = handle.read()
                if _DEFINES_CASES.search(text) and not _HAS_ENTRYPOINT.search(text):
                    offenders.append(os.path.relpath(path, _package_root()))
        self.assertEqual(
            [],
            offenders,
            "these sources define test cases but have no __main__ entrypoint, so "
            "bazel imports them and exits 0 -- PASSED with zero cases executed; "
            'add `if __name__ == "__main__":` with whichever runner the file '
            "already uses: %s" % offenders,
        )


if __name__ == "__main__":
    unittest.main()
