using System.Runtime.InteropServices;
using System;

namespace SherpaNcnn {

[StructLayout(LayoutKind.Sequential)]
public struct TransducerModelConfig {
  [MarshalAs(UnmanagedType.LPStr)]
  public string EncoderParam;

  [MarshalAs(UnmanagedType.LPStr)]
  public string EncoderBin;

  [MarshalAs(UnmanagedType.LPStr)]
  public string DecoderParam;

  [MarshalAs(UnmanagedType.LPStr)]
  public string DecoderBin;

  [MarshalAs(UnmanagedType.LPStr)]
  public string JoinerParam;

  [MarshalAs(UnmanagedType.LPStr)]
  public string JoinerBin;

  [MarshalAs(UnmanagedType.LPStr)]
  public string Tokens;

  public int UseVulkanCompute;

  public int NumThreads;
}

[StructLayout(LayoutKind.Sequential)]
public struct TransducerDecoderConfig {
  [MarshalAs(UnmanagedType.LPStr)]
  public string DecodingMethod;

  public int NumActivePaths;
}

[StructLayout(LayoutKind.Sequential)]
public struct FeatureConfig {
  public float SampleRate;
  public int FeatureDim;
}

[StructLayout(LayoutKind.Sequential)]
public struct OnlineRecognizerConfig {
  public FeatureConfig FeatConfig;
  public TransducerModelConfig ModelConfig;
  public TransducerDecoderConfig DecoderConfig;

  public int EnableEndpoit;
  public float Rule1MinTrailingSilence;
  public float Rule2MinTrailingSilence;
  public float Rule3MinUtteranceLength;
}


// please see
// https://learn.microsoft.com/en-us/dotnet/api/system.idisposable.dispose?view=net-7.0
class OnlineRecognizer : IDisposable {
 public OnlineRecognizer(OnlineRecognizerConfig config) {
    handle = CreateOnlineRecognizer(config);
  }

  public OnlineStream CreateStream() {
    IntPtr p = CreateOnlineStream(handle);
    return new OnlineStream(p);
  }

  public void Dispose() {
    Dispose(disposing: true);
    GC.SuppressFinalize(this);
  }

  protected virtual void Dispose(bool disposing) {
    // disposing is not used
    if(!this.disposed) {
      DestroyOnlineRecognizer(handle);
      handle = IntPtr.Zero;
      disposed = true;
    }
  }

  ~OnlineRecognizer() {
    Dispose(disposing: false);
  }

  private IntPtr handle;
  private bool disposed = false;

  private const string dllName = "sherpa-ncnn-c-api.dll";

  [DllImport(dllName, EntryPoint="CreateRecognizer")]
  public static extern IntPtr CreateOnlineRecognizer(OnlineRecognizerConfig config);

  [DllImport(dllName, EntryPoint="DestroyRecognizer")]
  public static extern void DestroyOnlineRecognizer(IntPtr handle);

  [DllImport(dllName, EntryPoint="CreateStream")]
  public static extern IntPtr CreateOnlineStream(IntPtr handle);
}

class OnlineStream : IDisposable {
  public OnlineStream(IntPtr p) {
    handle = p;
  }

  public void Dispose() {
    Dispose(disposing: true);
    GC.SuppressFinalize(this);
  }

  protected virtual void Dispose(bool disposing) {
    // disposing is not used
    if(!this.disposed) {
      DestroyOnlineStream(handle);
      handle = IntPtr.Zero;
      disposed = true;
    }
  }

  ~OnlineStream() {
    Dispose(disposing: false);
  }

  private IntPtr handle;
  private bool disposed = false;

  private const string dllName = "sherpa-ncnn-c-api.dll";

  [DllImport(dllName, EntryPoint="DestroyStream")]
  public static extern void DestroyOnlineStream(IntPtr handle);

}

public class Hello {
  public static void Main(String[] args) {
    OnlineRecognizerConfig config = new OnlineRecognizerConfig();
    config.FeatConfig.SampleRate =  16000;
    config.FeatConfig.FeatureDim =  80;
    config.ModelConfig.EncoderParam = "encoder_jit_trace-pnnx.ncnn.param";
    config.ModelConfig.EncoderBin = "encoder_jit_trace-pnnx.ncnn.bin";

    config.ModelConfig.DecoderParam = "decoder_jit_trace-pnnx.ncnn.param";
    config.ModelConfig.DecoderBin = "decoder_jit_trace-pnnx.ncnn.bin";

    config.ModelConfig.JoinerParam = "joiner_jit_trace-pnnx.ncnn.param";
    config.ModelConfig.JoinerBin = "joiner_jit_trace-pnnx.ncnn.bin";

    config.ModelConfig.Tokens = "tokens.txt";
    config.ModelConfig.UseVulkanCompute = 0;
    config.ModelConfig.NumThreads = 2;

    config.DecoderConfig.DecodingMethod = "greedy_search";
    config.DecoderConfig.NumActivePaths = 4;

    OnlineRecognizer recognizer = new OnlineRecognizer(config);

    Console.WriteLine("hello sherpa-ncnn");
  }
}

}
