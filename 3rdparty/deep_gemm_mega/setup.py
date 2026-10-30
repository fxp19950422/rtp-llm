import os
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = Path(__file__).resolve().parent
SDK = Path(os.environ.get("PPU_SDK", "/usr/local/PPU_SDK"))
ACTLIZE = ROOT / "third-party/actlize"
FMT = ROOT / "third-party/fmt"


class BuildPy(build_py):
    def run(self):
        super().run()
        dest = Path(self.build_lib) / "deep_gemm_mega"
        includes = dest / "include/actlize_v1.0.0"
        shutil.copytree(ACTLIZE / "include", includes, dirs_exist_ok=True)
        shutil.copytree(ACTLIZE / "tools", includes / "tools", dirs_exist_ok=True)
        licenses = dest / "licenses"
        licenses.mkdir(exist_ok=True)
        shutil.copy2(ACTLIZE / "LICENSE.txt", licenses / "actlize-LICENSE.txt")
        shutil.copy2(FMT / "LICENSE", licenses / "fmt-LICENSE")


setup(
    name="deep_gemm_mega",
    version="0.1.0",
    description="RTP-LLM PPU MXFP4 MegaMoE kernels",
    packages=["deep_gemm_mega", "deep_gemm_mega.mega"],
    package_data={"deep_gemm_mega": ["include/**/*", "configs/**/*.lut"]},
    python_requires=">=3.10",
    license_files=["LICENSE", "NOTICE"],
    ext_modules=[
        CppExtension(
            "deep_gemm_mega._C",
            sources=["csrc/python_api.cpp"],
            include_dirs=[
                str(SDK / "targets/x86_64-linux/include"),
                str(SDK / "CUDA_SDK/targets/x86_64-linux/include"),
                str(ROOT / "deep_gemm_mega/include"),
                str(ACTLIZE),
                str(ACTLIZE / "include"),
                str(ACTLIZE / "tools"),
                str(ACTLIZE / "include/cute"),
                str(FMT / "include"),
            ],
            library_dirs=[str(SDK / "lib")],
            libraries=["hggc", "hggcrt1", "hgrtc", "acblasLt"],
            extra_compile_args=[
                "-O3", "-std=c++17", "-fvisibility=hidden",
                "-DUSE_HGGC", "-DDG_MEGA_TORCH_CUDA_ABI=1",
                "-DDG_MEGA_HGGC_SUPPORT_PCH=1",
            ],
        )
    ],
    cmdclass={"build_py": BuildPy, "build_ext": BuildExtension},
    zip_safe=False,
)
