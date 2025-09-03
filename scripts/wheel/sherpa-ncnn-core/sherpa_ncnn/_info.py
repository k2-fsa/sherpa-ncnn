from pathlib import Path
from typing import List

_pkg_dir = Path(__file__).parent
libs_dir = _pkg_dir / "lib"
include_dir = _pkg_dir / "include"

# List of libraries (without "lib" prefix, without extension)
# Adjust to match your actual .so/.dll/.dylib files
ncnn_lib = ["ncnn"]
c_lib = ["sherpa-ncnn-c-api"] + ncnn_lib


def get_include_dir() -> str:
    return str(include_dir)


def get_libs_dir() -> str:
    return str(libs_dir)


def get_c_api_libs() -> List[str]:
    return c_lib
