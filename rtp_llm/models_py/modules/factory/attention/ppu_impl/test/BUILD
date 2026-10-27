load("@arch_config//:arch_select.bzl", "requirement")

py_test(
    name = "test_fa3_unused_cache",
    srcs = ["test_fa3_unused_cache.py"],
    exec_properties = {"gpu": "PPU-ZW810E"},
    tags = ["PPU-ZW810E", "manual"],
    deps = ["//rtp_llm/models_py:ppu_modules", ":torch"],
)

requirement([
    "triton",
    "torch",
    "numpy",
])

py_test(
    name = "test_fa3_decode_padding",
    srcs = ["test_fa3_decode_padding.py"],
    exec_properties = {"gpu": "PPU-ZW810E"},
    tags = [
        "PPU-ZW810E",
        "manual",
    ],
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/models_py:ppu_modules",
    ] + [":torch"],
)
