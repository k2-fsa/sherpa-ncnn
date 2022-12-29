#!/usr/bin/env python3

import os
import re
import sys
from pathlib import Path

import setuptools

from cmake.cmake_extension import (
    BuildExtension,
    bdist_wheel,
    cmake_extension,
    is_windows,
)


def read_long_description():
    with open("README.md", encoding="utf8") as f:
        readme = f.read()
    return readme


def get_package_version():
    with open("CMakeLists.txt") as f:
        content = f.read()

    match = re.search(r"set\(SHERPA_NCNN_VERSION (.*)\)", content)
    latest_version = match.group(1).strip('"')
    return latest_version


def get_binaries_to_install():
    bin_dir = Path("build") / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if is_windows() else ""
    # Remember to also change cmake/cmake_extension.py
    binaries = ["sherpa-ncnn"]
    binaries += ["sherpa-ncnn-microphone"]
    exe = []
    for f in binaries:
        t = bin_dir / (f + suffix)
        exe.append(str(t))
    return exe


package_name = "sherpa-ncnn"

with open("sherpa-ncnn/python/sherpa_ncnn/__init__.py", "a") as f:
    f.write(f"__version__ = '{get_package_version()}'\n")

setuptools.setup(
    name=package_name,
    version=get_package_version(),
    author="The sherpa-ncnn development team",
    author_email="dpovey@gmail.com",
    package_dir={
        "sherpa_ncnn": "sherpa-ncnn/python/sherpa_ncnn",
    },
    data_files=[("bin", get_binaries_to_install())],
    packages=["sherpa_ncnn"],
    url="https://github.com/k2-fsa/sherpa-ncnn",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    ext_modules=[cmake_extension("_sherpa_ncnn")],
    cmdclass={"build_ext": BuildExtension, "bdist_wheel": bdist_wheel},
    zip_safe=False,
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="Apache licensed, as found in the LICENSE file",
)

with open("sherpa-ncnn/python/sherpa_ncnn/__init__.py", "r") as f:
    lines = f.readlines()

with open("sherpa-ncnn/python/sherpa_ncnn/__init__.py", "w") as f:
    for line in lines:
        if "__version__" in line:
            # skip __version__ = "x.x.x"
            continue
        f.write(line)
