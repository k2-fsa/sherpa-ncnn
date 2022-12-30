# Copyright (c)  2021-2022  Xiaomi Corporation (author: Fangjun Kuang)
# flake8: noqa

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

        out_bin_dir = os.path.dirname(self.build_lib) + "/bin"
        install_dir = os.path.realpath(self.build_lib)

        sherpa_ncnn_dir = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))

        cmake_args = os.environ.get("SHERPA_NCNN_CMAKE_ARGS", "")
        make_args = os.environ.get("SHERPA_NCNN_MAKE_ARGS", "")
        system_make_args = os.environ.get("MAKEFLAGS", "")

        if cmake_args == "":
            cmake_args = "-DCMAKE_BUILD_TYPE=Release"

        extra_cmake_args = " -DCMAKE_INSTALL_PREFIX=%s " % install_dir
        extra_cmake_args += " -DBUILD_SHARED_LIBS=ON "
        extra_cmake_args += " -DSHERPA_NCNN_ENABLE_PYTHON=ON "
        extra_cmake_args += " -DSHERPA_NCNN_ENABLE_PORTAUDIO=ON "

        if "PYTHON_EXECUTABLE" not in cmake_args:
            print("Setting PYTHON_EXECUTABLE to %s" % sys.executable)
            cmake_args += " -DPYTHON_EXECUTABLE=%s " % sys.executable

        cmake_args += extra_cmake_args

        if is_windows():
            build_cmd = """
         cmake {0} -B {1} -S {2}
         cmake --build {1} --target install --config Release -- -m
            """.format(
                cmake_args, self.build_temp, sherpa_ncnn_dir
            )
            print("build command is:\n{}".format(build_cmd))
            ret = os.system(
                "cmake {} -B {} -S {}".format(
                    cmake_args, self.build_temp, sherpa_ncnn_dir
                )
            )
            if ret != 0:
                raise Exception("Failed to configure sherpa")

            ret = os.system(
                "cmake --build {} --target install --config Release -- -m".format(
                    self.build_temp
                )
            )
            if ret != 0:
                raise Exception("Failed to build and install sherpa")
        else:
            if make_args == "" and system_make_args == "":
                print("for fast compilation, run:")
                print('export SHERPA_NCNN_MAKE_ARGS="-j"; python setup.py install')
                print('Setting make_args to "-j4"')
                make_args = "-j4"

            build_cmd = """
                cd {}

                cmake {} {}

                make {} install/strip
            """.format(
                self.build_temp, cmake_args, sherpa_ncnn_dir, make_args
            )
            print("build command is:\n{}".format(build_cmd))

            ret = os.system(build_cmd)
            if ret != 0:
                raise Exception(
                    "\nBuild sherpa-ncnn failed. Please check the error message.\n"
                    "You can ask for help by creating an issue on GitHub.\n"
                    "\nClick:\n\thttps://github.com/k2-fsa/sherpa-ncnn/issues/new\n"  # noqa
                )

        suffix = ".exe" if is_windows() else ""
        # Remember to also change setup.py
        binaries = ["sherpa-ncnn"]
        binaries += ["sherpa-ncnn-microphone"]

        for f in binaries:
            src_file = install_dir + "/bin/" + (f + suffix)
            print("Copying {} to {}/".format(src_file, out_bin_dir))
            shutil.copy(src_file, "{}/".format(out_bin_dir))
