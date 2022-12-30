# Copyright (c)  2021-2022  Xiaomi Corporation (author: Fangjun Kuang)
# flake8: noqa

import glob
import os
import platform
import shutil
import sys
from pathlib import Path

import setuptools
from setuptools.command.build_ext import build_ext


def is_for_pypi():
    ans = os.environ.get("SHERPA_NCNN_IS_FOR_PYPI", None)
    return ans is not None


def is_macos():
    return platform.system() == "Darwin"


def is_windows():
    return platform.system() == "Windows"


try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            # In this case, the generated wheel has a name in the form
            # sherpa-xxx-pyxx-none-any.whl
            if is_for_pypi() and not is_macos():
                self.root_is_pure = True
            else:
                # The generated wheel has a name ending with
                # -linux_x86_64.whl
                self.root_is_pure = False


except ImportError:
    bdist_wheel = None


def cmake_extension(name, *args, **kwargs) -> setuptools.Extension:
    kwargs["language"] = "c++"
    sources = []
    return setuptools.Extension(name, sources, *args, **kwargs)


class BuildExtension(build_ext):
    def build_extension(self, ext: setuptools.extension.Extension):
        # build/temp.linux-x86_64-3.8
        os.makedirs(self.build_temp, exist_ok=True)

        # build/lib.linux-x86_64-3.8
        os.makedirs(self.build_lib, exist_ok=True)

        install_dir = Path(self.build_lib).resolve() / "sherpa_ncnn"

        sherpa_ncnn_dir = Path(__file__).parent.parent.resolve()

        cmake_args = os.environ.get("SHERPA_NCNN_CMAKE_ARGS", "")
        make_args = os.environ.get("SHERPA_NCNN_MAKE_ARGS", "")
        system_make_args = os.environ.get("MAKEFLAGS", "")

        if cmake_args == "":
            cmake_args = "-DCMAKE_BUILD_TYPE=Release"

        extra_cmake_args = f" -DCMAKE_INSTALL_PREFIX={install_dir} "
        extra_cmake_args += f" -DBUILD_SHARED_LIBS=ON "
        extra_cmake_args += f" -DSHERPA_NCNN_ENABLE_PYTHON=ON "
        extra_cmake_args += f" -DSHERPA_NCNN_ENABLE_PORTAUDIO=OFF "

        if "PYTHON_EXECUTABLE" not in cmake_args:
            print(f"Setting PYTHON_EXECUTABLE to {sys.executable}")
            cmake_args += f" -DPYTHON_EXECUTABLE={sys.executable}"

        cmake_args += extra_cmake_args

        if is_windows():
            build_cmd = f"""
         cmake {cmake_args} -B {self.build_temp} -S {sherpa_ncnn_dir}
         cmake --build {self.build_temp} --target install --config Release -- -m
            """
            print(f"build command is:\n{build_cmd}")
            ret = os.system(
                f"cmake {cmake_args} -B {self.build_temp} -S {sherpa_ncnn_dir}"
            )
            if ret != 0:
                raise Exception("Failed to configure sherpa")

            ret = os.system(
                f"cmake --build {self.build_temp} --target install --config Release -- -m"  # noqa
            )
            if ret != 0:
                raise Exception("Failed to build and install sherpa")
        else:
            if make_args == "" and system_make_args == "":
                print("for fast compilation, run:")
                print('export SHERPA_NCNN_MAKE_ARGS="-j"; python setup.py install')
                print('Setting make_args to "-j4"')
                make_args = "-j4"

            build_cmd = f"""
                cd {self.build_temp}

                cmake {cmake_args} {sherpa_ncnn_dir}

                make {make_args} install/strip
            """
            print(f"build command is:\n{build_cmd}")

            ret = os.system(build_cmd)
            if ret != 0:
                raise Exception(
                    "\nBuild sherpa-ncnn failed. Please check the error message.\n"
                    "You can ask for help by creating an issue on GitHub.\n"
                    "\nClick:\n\thttps://github.com/k2-fsa/sherpa-ncnn/issues/new\n"  # noqa
                )
        dirs = [
            install_dir / "include",
            install_dir / "lib" / "cmake",
            install_dir / "lib" / "pkgconfig",
            install_dir / "lib64" / "cmake",
            install_dir / "lib64" / "pkgconfig",
        ]

        for d in dirs:
            if not d.is_dir():
                continue

            shutil.rmtree(str(d))

        is_in_github_actions = os.environ.get("SHERPA_NCNN_IS_IN_GITHUB_ACTIONS", None)
        if is_macos() and is_in_github_actions is not None:
            libs = glob.glob(f"{self.install_dir}/lib/lib*.dylib", recursive=False)

            for lib in libs:
                print(f"Copying {lib} to {sherpa_ncnn_dir}/")
                shutil.copy(lib, sherpa_ncnn_dir)
