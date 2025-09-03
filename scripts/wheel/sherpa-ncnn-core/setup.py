import platform

from setuptools import setup


def is_windows():
    return platform.system() == "Windows"


def get_binaries():
    if not is_windows():
        return None
    libs = [
        "ncnn.dll",
        "sherpa-ncnn-c-api.dll",
        "sherpa-ncnn-c-api.lib",
    ]
    prefix = "./sherpa_ncnn/lib"
    return [f"{prefix}/{lib}" for lib in libs]


setup(
    name="sherpa-ncnn-core",
    version="0.0.1",
    description="Core shared libraries for sherpa-ncnn",
    packages=["sherpa_ncnn"],
    include_package_data=True,
    data_files=[("Scripts", get_binaries())] if get_binaries() else None,
    author="The sherpa-ncnn development team",
    url="https://github.com/k2-fsa/sherpa-ncnn",
    author_email="dpovey@gmail.com",
    zip_safe=False,
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
