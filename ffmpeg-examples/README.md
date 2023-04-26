# ffmpeg-examples

Enable sherpa-ncnn to use any url/file input that FFmpeg supports.

## Usage

This example is disabled by default, please enable it by:

```bash
cd sherpa-ncnn
mkdir -p build
cd build
cmake -DSHERPA_NCNN_ENABLE_FFMPEG_EXAMPLES=ON ..
make -j10
```

> Note: You can set `-DSHERPA_NCNN_ENABLE_DEBUG_FOR_RELEASE=ON` to enable debug symbols for release build, 
> see [#147](https://github.com/k2-fsa/sherpa-ncnn/issues/147) for more details.

Please install ffmpeg first:

* macOS: `brew install ffmpeg`

# Fixes for errors

To fix the following error:
```
Package libavdevice was not found in the pkg-config search path.
```
please run

```
sudo apt-get install libavdevice-dev
```

To fix the following error
```
Makefile:28: *** FFmpeg version should be n5.1 or above!.  Stop.
```
please run
```
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:savoury1/ffmpeg4
sudo add-apt-repository ppa:savoury1/ffmpeg5
sudo apt-get update
sudo apt-get install ffmpeg --reinstall
sudo apt-get install libavutil-dev --reinstall
```

To fix the following error:
```
ModuleNotFoundError: No module named 'apt_pkg'
```
please run:
```
sudo apt-get install python-apt
```
