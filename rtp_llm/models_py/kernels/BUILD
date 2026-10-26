load("@arch_config//:arch_select.bzl", "requirement")

requirement(["torch", "triton"])

py_library(
    name = "internal_kernels",
    srcs = glob(["ppu/**/*.py"]),
    data = glob(["ppu/**/*.json"]),
    deps = [
        ":torch",
        ":triton",
    ],
    visibility = ["//visibility:public"],
)
