# DSV4 python tests target the CUDA13/Blackwell stack (deep_gemm >= 2.6, dsv4
# kernels, pytest closure). They are skipped as incompatible on other platforms
# instead of failing on pools that can never satisfy those dependencies.
_DSV4_CUDA13_ONLY = select({
    "@//:using_cuda13_x86": [],
    "@//:using_cuda13_arm": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

py_test(
    name = "test_dspark_proposer",
    srcs = ["test_dspark_proposer.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_target_verify_contract",
    srcs = ["test_target_verify_contract.py"],
    # test_legacy_cache_module_reexports_t13_leaf reads this file off disk and
    # AST-parses it, so it has to be a declared data input -- undeclared, it is
    # simply absent from runfiles and the case raises FileNotFoundError.
    data = ["//rtp_llm:models/dsv4_kv_cache.py"],
    deps = [
        "//rtp_llm:dsv4_contracts",
        "//rtp_llm/models_py:target_verify_contract",
    ],
)

# Read as text (never imported) by
# //rtp_llm/models_py/modules/dsv4/test:test_py_test_entrypoint_guard, because
# bazel's glob() cannot reach across a package boundary.
filegroup(
    name = "speculative_test_sources",
    srcs = glob(["*.py"]),
    visibility = ["//rtp_llm/models_py/modules/dsv4/test:__pkg__"],
)
