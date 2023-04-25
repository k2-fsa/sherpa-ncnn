using System.Runtime.InteropServices;

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

  int UseVulkanCompute;

  int NumThreads;
}

[StructLayout(LayoutKind.Sequential)]
public struct TransducerDecoderConfig {
  [MarshalAs(UnmanagedType.LPStr)]
  public string DecodingMethod;

  int NumActivePaths;
}

[StructLayout(LayoutKind.Sequential)]
public struct FeatureConfig {
  float SampleRate;
  int FeatureDim;
}

[StructLayout(LayoutKind.Sequential)]
public struct OnlineRecognizerConfig {
  FeatureConfig FeatConfig;
  TransducerModelConfig ModelConfig;
  TransducerDecoderConfig DecoderConfig;

  int EnableEndpoit;
  float Rule1MinTrailingSilence;
  float Rule2MinTrailingSilence;
  float Rule3MinUtteranceLength;
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

}
