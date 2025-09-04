#!/usr/bin/env python3
# Copyright (c)  2023  Xiaomi Corporation

import os
import re
from pathlib import Path

import jinja2

SHERPA_NCNN_DIR = Path(__file__).resolve().parent.parent.parent

src_dir = os.environ.get("src_dir", "/tmp")


def get_version():
    cmake_file = SHERPA_NCNN_DIR / "CMakeLists.txt"
    with open(cmake_file) as f:
        content = f.read()

    version = re.search(r"set\(SHERPA_NCNN_VERSION (.*)\)", content).group(1)
    return version.strip('"')


def read_proj_file(filename):
    with open(filename) as f:
        return f.read()


def get_dict():
    version = get_version()
    return {
        "version": version,
    }


def process_linux(s, rid):
    libs = [
        "libncnn.so",
        "libsherpa-ncnn-c-api.so",
    ]
    prefix = f"{src_dir}/linux-{rid}/"
    libs = [prefix + lib for lib in libs]
    libs = "\n      ;".join(libs)

    d = get_dict()
    d["dotnet_rid"] = f"linux-{rid}"
    d["libs"] = libs

    environment = jinja2.Environment()
    template = environment.from_string(s)
    s = template.render(**d)
    with open(f"./linux-{rid}/sherpa-ncnn.runtime.csproj", "w") as f:
        f.write(s)


def process_macos(s, rid):
    libs = [
        "libncnn.dylib",
        "libsherpa-ncnn-c-api.dylib",
    ]
    prefix = f"{src_dir}/macos-{rid}/"
    libs = [prefix + lib for lib in libs]
    libs = "\n      ;".join(libs)

    d = get_dict()
    d["dotnet_rid"] = f"osx-{rid}"
    d["libs"] = libs

    environment = jinja2.Environment()
    template = environment.from_string(s)
    s = template.render(**d)
    with open(f"./macos-{rid}/sherpa-ncnn.runtime.csproj", "w") as f:
        f.write(s)


def process_windows(s, rid):
    libs = [
        "ncnn.dll",
        "sherpa-ncnn-c-api.dll",
    ]

    prefix = f"{src_dir}/windows-{rid}/"
    libs = [prefix + lib for lib in libs]
    libs = "\n      ;".join(libs)

    d = get_dict()
    d["dotnet_rid"] = f"win-{rid}"
    d["libs"] = libs

    environment = jinja2.Environment()
    template = environment.from_string(s)
    s = template.render(**d)
    with open(f"./windows-{rid}/sherpa-ncnn.runtime.csproj", "w") as f:
        f.write(s)


def main():
    s = read_proj_file("./sherpa-ncnn.csproj.runtime.in")
    process_linux(s, "x64")
    process_linux(s, "arm64")
    process_macos(s, "x64")
    process_macos(s, "arm64")
    process_windows(s, "x64")
    process_windows(s, "x86")

    s = read_proj_file("./sherpa-ncnn.csproj.in")
    d = get_dict()
    d["packages_dir"] = str(SHERPA_NCNN_DIR / "scripts/dotnet/packages")

    environment = jinja2.Environment()
    template = environment.from_string(s)
    s = template.render(**d)
    with open("./all/sherpa-ncnn.csproj", "w") as f:
        f.write(s)


if __name__ == "__main__":
    main()
