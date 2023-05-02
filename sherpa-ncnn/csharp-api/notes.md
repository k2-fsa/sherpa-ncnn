# Introduction

This directory is for .Net.

We use <https://github.com/Mizux/dotnet-native>
as a reference to build native Nuget packages for various platforms
such as Linux, macOS, and Windows.

# Install .Net on Linux

Please see <https://learn.microsoft.com/en-us/dotnet/core/install/linux-scripted-manual>
for details.

The following is just some notes about the installation:

```bash
wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh
chmod +x dotnet-install.sh
./dotnet-install.sh --help
./dotnet-install.sh  --install-dir /star-fj/fangjun/software/dotnet
export PATH=/star-fj/fangjun/software/dotnet:$PATH

# Check that the installation is successful
which dotnet
dotnet --info

# To install the runtime, use
./dotnet-install.sh --runtime dotnet --install-dir /star-fj/fangjun/software/dotnet/
```

# Build Nuget packages
```bash
cd /tmp
git clone http://github.com/k2-fsa/sherpa-ncnn
cd sherpa-ncnn

mkdir build
cd build
cmake -DBUILD_SHARED_LIBS=ON -DSHERPA_NCNN_ENABLE_DOT_NET_API=ON -DSHERPA_NCNN_ENABLE_PORTAUDIO=OFF ..
make -j 6 sherpa-ncnn-dot-net-package

ls -lh packages

dotnet nuget add source --name my-source $PWD/packages

dotnet nuget list source
```

If you are using Linux, you will see the following output:
```bash
$ ls -lh  packages/
total 2.3M
-rw-r--r-- 1 kuangfangjun root  14K May  2 17:42 org.k2fsa.sherpa.ncnn.1.8.1.nupkg
-rw-r--r-- 1 kuangfangjun root 2.3M May  2 17:42 org.k2fsa.sherpa.ncnn.runtime.linux-x64.1.8.1.nupkg
```

If you are using macOS, you will see the following output:
```bash
$ ls -lh packages/
total 5200
-rw-r--r--  1 fangjun  staff    13K May  2 17:17 org.k2fsa.sherpa.ncnn.1.8.1.nupkg
-rw-r--r--  1 fangjun  staff   2.5M May  2 17:17 org.k2fsa.sherpa.ncnn.runtime.osx-x64.1.8.1.nupkg
```

# Test the generated packages
```bash
cd /tmp
mkdir hello
cd hello
dotnet new sln

dotnet new console -o test-sherpa-ncnn


dotnet sln add ./test-sherpa-ncnn/test-sherpa-ncnn.csproj
cd test-sherpa-ncnn
dotnet add package org.k2fsa.sherpa.ncnn -v 1.8.1
```

# Notes about dotnet

To list available nuget sources:

```bash
dotnet nuget list source
```

To publish a package:

```bash
export MY_API_KEY=xxxxxx
dotnet nuget push ./org.k2fsa.sherpa.ncnn.runtime.osx-x64.1.8.2.nupkg --api-key $MY_API_KEY --source https://api.nuget.org/v3/index.json
dotnet nuget push ./org.k2fsa.sherpa.ncnn.1.8.2.nupkg --api-key $MY_API_KEY --source https://api.nuget.org/v3/index.json
```

To clear all caches:

```bash
dotnet nuget locals all --clear
```
