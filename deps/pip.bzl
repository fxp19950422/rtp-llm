load("@rules_python//python:pip.bzl", "pip_parse")
load(":ppu_requirements.bzl", "ppu_requirements")

PIP_EXTRA_ARGS = [
    "--cache-dir=~/.cache/pip",
    "--extra-index-url=https://mirrors.aliyun.com/pypi/simple/",
    "--verbose",
]

def pip_deps():
    ppu_requirements(name = "ppu_requirements")
    ppu_requirements(
        name = "ppu_sdk22_requirements",
        requirements = "@rtp_deps//:requirements_torch_ppu_sdk22.txt",
        lock = "@rtp_deps//:requirements_lock_torch_ppu_sdk22.txt",
        allow_local_wheelhouse = True,
    )

    pip_parse(
        name = "pip_cpu_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_torch_cpu.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
    )

    pip_parse(
        name = "pip_arm_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_torch_arm.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
    )

    pip_parse(
        name = "pip_ppu_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_torch_ppu.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
    )

    pip_parse(
        name = "pip_ppu_sdk22_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_torch_ppu_sdk22.txt",
        # Metadata is loaded on every platform; wheel extraction is lazy.
        # SDK22 images provide Python 3.12 as python3 on PATH.
        python_interpreter = "python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
    )

    pip_parse(
        name = "pip_gpu_cuda12_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_torch_gpu_cuda12.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_gpu_cuda12_9_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_torch_gpu_cuda12_9.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_gpu_cuda13_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_torch_gpu_cuda13.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS + ["--quiet"],
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_cuda12_arm_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_cuda12_arm.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_cuda13_arm_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_cuda13_arm.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_gpu_rocm_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_rocm.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 12000,
    )
