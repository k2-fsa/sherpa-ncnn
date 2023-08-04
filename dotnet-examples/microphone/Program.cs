// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
using System;
using System.Threading;
using PortAudioSharp;
using System.Runtime.InteropServices;

class Microphone
{
    public static void Main(String[] args)
    {
        String usage = @"
      ./microphone.exe \
         /path/to/tokens.txt \
         /path/to/encoder.ncnn.param \
         /path/to/encoder.ncnn.bin \
         /path/to/decoder.ncnn.param \
         /path/to/decoder.ncnn.bin \
         /path/to/joiner.ncnn.param \
         /path/to/joiner.ncnn.bin \
         [<num_threads> [decode_method]]

      num_threads: Default to 1
      decoding_method: greedy_search (default), or modified_beam_search

      Please refer to
      https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html
      for a list of pre-trained models to download.
      ";
        if (args.Length < 7 || args.Length > 9)
        {
            Console.WriteLine(usage);
            return;
        }

        SherpaNcnn.OnlineRecognizerConfig config = new SherpaNcnn.OnlineRecognizerConfig();
        config.FeatConfig.SampleRate = 16000;
        config.FeatConfig.FeatureDim = 80;
        config.ModelConfig.Tokens = args[0];
        config.ModelConfig.EncoderParam = args[1];
        config.ModelConfig.EncoderBin = args[2];

        config.ModelConfig.DecoderParam = args[3];
        config.ModelConfig.DecoderBin = args[4];

        config.ModelConfig.JoinerParam = args[5];
        config.ModelConfig.JoinerBin = args[6];

        config.ModelConfig.UseVulkanCompute = 0;
        config.ModelConfig.NumThreads = 1;
        if (args.Length >= 8)
        {
            config.ModelConfig.NumThreads = Int32.Parse(args[7]);
            if (config.ModelConfig.NumThreads > 1)
            {
                Console.WriteLine($"Use num_threads: {config.ModelConfig.NumThreads}");
            }
        }

        config.DecoderConfig.DecodingMethod = "greedy_search";
        if (args.Length == 9 && args[8] != "greedy_search")
        {
            Console.WriteLine($"Use decoding_method {args[8]}");
            config.DecoderConfig.DecodingMethod = args[8];
        }

        config.DecoderConfig.NumActivePaths = 4;
        config.EnableEndpoint = 1;
        config.Rule1MinTrailingSilence = 2.4F;
        config.Rule2MinTrailingSilence = 1.2F;
        config.Rule3MinUtteranceLength = 20.0F;


        SherpaNcnn.OnlineRecognizer recognizer = new SherpaNcnn.OnlineRecognizer(config);

        SherpaNcnn.OnlineStream s = recognizer.CreateStream();

        Console.WriteLine(PortAudio.VersionInfo.versionText);
        PortAudio.Initialize();

        Console.WriteLine($"Number of devices: {PortAudio.DeviceCount}");
        for (int i = 0; i != PortAudio.DeviceCount; ++i)
        {
            Console.WriteLine($" Device {i}");
            DeviceInfo deviceInfo = PortAudio.GetDeviceInfo(i);
            Console.WriteLine($"   Name: {deviceInfo.name}");
            Console.WriteLine($"   Max input channels: {deviceInfo.maxInputChannels}");
            Console.WriteLine($"   Default sample rate: {deviceInfo.defaultSampleRate}");
        }
        int deviceIndex = PortAudio.DefaultInputDevice;
        if (deviceIndex == PortAudio.NoDevice)
        {
            Console.WriteLine("No default input device found");
            Environment.Exit(1);
        }

        DeviceInfo info = PortAudio.GetDeviceInfo(deviceIndex);

        Console.WriteLine();
        Console.WriteLine($"Use default device {deviceIndex} ({info.name})");

        StreamParameters param = new StreamParameters();
        param.device = deviceIndex;
        param.channelCount = 1;
        param.sampleFormat = SampleFormat.Float32;
        param.suggestedLatency = info.defaultLowInputLatency;
        param.hostApiSpecificStreamInfo = IntPtr.Zero;

        PortAudioSharp.Stream.Callback callback = (IntPtr input, IntPtr output,
            UInt32 frameCount,
            ref StreamCallbackTimeInfo timeInfo,
            StreamCallbackFlags statusFlags,
            IntPtr userData
            ) =>
        {
            float[] samples = new float[frameCount];
            Marshal.Copy(input, samples, 0, (Int32)frameCount);

            s.AcceptWaveform(16000, samples);

            return StreamCallbackResult.Continue;
        };

        PortAudioSharp.Stream stream = new PortAudioSharp.Stream(inParams: param, outParams: null, sampleRate: 16000,
            framesPerBuffer: 0,
            streamFlags: StreamFlags.ClipOff,
            callback: callback,
            userData: IntPtr.Zero
            );

        Console.WriteLine(param);
        Console.WriteLine("Started! Please speak\n\n");

        stream.Start();

        String lastText = "";
        int segmentIndex = 0;

        while (true)
        {
            while (recognizer.IsReady(s))
            {
                recognizer.Decode(s);
            }

            var text = recognizer.GetResult(s).Text;
            bool isEndpoint = recognizer.IsEndpoint(s);
            if (!string.IsNullOrWhiteSpace(text) && lastText != text)
            {
                lastText = text;
                Console.Write($"\r{segmentIndex}: {lastText}");
            }

            if (isEndpoint)
            {
                if (!string.IsNullOrWhiteSpace(text))
                {
                    ++segmentIndex;
                    Console.WriteLine();
                }
                recognizer.Reset(s);
            }

            Thread.Sleep(200); // ms
        }

        PortAudio.Terminate();
    }
}
