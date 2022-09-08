# gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz

Go to <https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads/8-3-2019-03> to download the toolchain.

```bash
mkdir /ceph-fj/fangjun/software
cd /ceph-fj/fangjun/software
tar xvf /path/to/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz

export PATH=/ceph-fj/fangjun/software/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin:$PATH
```


# gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu.tar.xz

Go to <https://releases.linaro.org/components/toolchain/binaries/latest-7/aarch64-linux-gnu/>

```bash
wget https://releases.linaro.org/components/toolchain/binaries/latest-7/aarch64-linux-gnu/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu.tar.xz

tar xvf gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu.tar.xz -C /ceph-fj/fangjun/software

export PATH=/ceph-fj/fangjun/software/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin:$PATH
```
