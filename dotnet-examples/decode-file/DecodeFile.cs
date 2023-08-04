// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
using System;

class DecodeFile
{
    public static void Main(String[] args)
    {
        String usage = @"
      ./DecodeFile.exe \
         /path/to/tokens.txt \
         /path/to/encoder.ncnn.param \
         /path/to/encoder.ncnn.bin \
         /path/to/decoder.ncnn.param \
         /path/to/decoder.ncnn.bin \
         /path/to/joiner.ncnn.param \
         /path/to/joiner.ncnn.bin \
         /path/to/foo.wav [<num_threads> [decode_method]]

      num_threads: Default to 1
      decoding_method: greedy_search (default), or modified_beam_search

      Please refer to
      https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html
      for a list of pre-trained models to download.
      ";
        if (args.Length < 8 || args.Length > 10)
        {
            Console.WriteLine(usage);
            return;
        }

        String waveFilename = args[7];
        SherpaNcnn.WaveReader waveReader = new SherpaNcnn.WaveReader(waveFilename);

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
        if (args.Length >= 9)
        {
            config.ModelConfig.NumThreads = Int32.Parse(args[8]);
            if (config.ModelConfig.NumThreads > 1)
            {
                Console.WriteLine($"Use num_threads: {config.ModelConfig.NumThreads}");
            }
        }

        config.DecoderConfig.DecodingMethod = "greedy_search";
        if (args.Length == 10 && args[9] != "greedy_search")
        {
            Console.WriteLine($"Use decoding_method {args[9]}");
            config.DecoderConfig.DecodingMethod = args[9];
        }

        config.DecoderConfig.NumActivePaths = 4;

        SherpaNcnn.OnlineRecognizer recognizer = new SherpaNcnn.OnlineRecognizer(config);

        SherpaNcnn.OnlineStream stream = recognizer.CreateStream();
        stream.AcceptWaveform(waveReader.SampleRate, waveReader.Samples);

        float[] tailPadding = new float[(int)(waveReader.SampleRate * 0.3)];
        stream.AcceptWaveform(waveReader.SampleRate, tailPadding);

        stream.InputFinished();

        while (recognizer.IsReady(stream))
        {
            recognizer.Decode(stream);
        }

        SherpaNcnn.OnlineRecognizerResult result = recognizer.GetResult(stream);
        Console.WriteLine(result.Text);
    }
}
