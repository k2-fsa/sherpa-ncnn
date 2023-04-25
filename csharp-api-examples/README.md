# Usage

First, let us build `DLLs`:

```bash
git clone https://github.com/k2-fsa/sherpa-ncnn
cd sherpa-ncnn
mkdir build
cd build
cmake -DBUILD_SHARED_LIBS=ON ..
cmake --build . --target sherpa-ncnn-c-api --config Release
```

After building, you will find the following DLLs in the directory `./bin/Release`:

```bash
C:\Users\fangjun\open-source\sherpa-ncnn\build>dir bin\Release\*.dll
 Volume in drive C is System
 Volume Serial Number is 8E17-A21F

 Directory of C:\Users\fangjun\open-source\sherpa-ncnn\build\bin\Release

04/25/2023  04:20 PM            90,624 kaldi-native-fbank-core.dll
04/25/2023  04:20 PM         4,120,064 ncnn.dll
03/06/2023  07:02 PM           175,104 portaudio_x64.dll
04/25/2023  04:20 PM            22,528 sherpa-ncnn-c-api.dll
04/25/2023  04:20 PM           304,128 sherpa-ncnn-core.dll
               5 File(s)      4,712,448 bytes
               0 Dir(s)  95,997,632,512 bytes free
```

For simplicity, we copy all required C# source files to bin\Release
so that you don't need to setup the environment variable `PATH`

```bash
cd bin\Release
copy ..\..\..\csharp-api-examples\DecodeFile.cs .
copy ..\..\..\csharp-api-examples\WaveReader.cs .
copy ..\..\..\sherpa-ncnn\csharp-api\SherpaNcnn.cs
```

Now, let us build an `exe` from the generated DLLs and copied C# source files:

```bash

```

You will see the following output:

```bash
C:\Users\fangjun\open-source\sherpa-ncnn\build\bin\Release>csc .\SherpaNcnn.cs .\WaveReader.cs .\DecodeFile.cs

Microsoft (R) Visual C# Compiler version 3.11.0-4.22108.8 (d9bef045)
Copyright (C) Microsoft Corporation. All rights reserved.


C:\Users\fangjun\open-source\sherpa-ncnn\build\bin\Release>dir DecodeFile.exe
 Volume in drive C is System
 Volume Serial Number is 8E17-A21F

 Directory of C:\Users\fangjun\open-source\sherpa-ncnn\build\bin\Release

04/25/2023  05:06 PM            11,776 DecodeFile.exe
               1 File(s)         11,776 bytes
               0 Dir(s)  95,997,943,808 bytes free
```

```bash
C:\Users\fangjun\open-source\sherpa-ncnn\build\bin\Release>.\DecodeFile.exe
0 args

      ./DecodeFile \
         /path/to/tokens.txt \
         /path/to/encoder.ncnn.param \
         /path/to/encoder.ncnn.bin \
         /path/to/decoder.ncnn.param \
         /path/to/decoder.ncnn.bin \
         /path/to/joiner.ncnn.param \
         /path/to/joiner.ncnn.bin \
         /path/to/foo.wav [<num_threads> [decode_method]]

      num_threads: Default to 2
      decoding_method: greedy_search (default), or modified_beam_search

      Please refer to
      https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html
      for a list of pre-trained models to download.
```

Please refer to the help information for usage.
