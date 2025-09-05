#!/usr/bin/env bash

set -ex

old_version="2\.1\.13"
new_version="2\.1\.14"

replace_str="s/$old_version/$new_version/g"

sed -i.bak "$replace_str" ./sherpa-ncnn/csrc/version.cc
sha1=$(git describe --match=NeVeRmAtCh --always --abbrev=8)
date=$(git log -1 --format=%ad --date=local)

sed -i.bak "$replace_str" ./CMakeLists.txt

sed -i.bak "s/  static const char \*sha1.*/  static const char \*sha1 = \"$sha1\";/g" ./sherpa-ncnn/csrc/version.cc
sed -i.bak "s/  static const char \*date.*/  static const char \*date = \"$date\";/g" ./sherpa-ncnn/csrc/version.cc

find scripts/wheel -name "setup.py" -type f -exec sed -i.bak "$replace_str" {} \;

find . -name "*.bak" -exec rm {} \;
