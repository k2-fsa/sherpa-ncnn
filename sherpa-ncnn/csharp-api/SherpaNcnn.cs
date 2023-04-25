// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)

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

  public bool IsReady(OnlineStream stream) {
    return IsReady(handle, stream.Handle) != 0;
  }

  public void Decode(OnlineStream stream) {
    Decode(handle, stream.Handle);
  }

  public OnlineRecognizerResult GetResult(OnlineStream stream) {
    IntPtr h = GetResult(handle, stream.Handle);
    OnlineRecognizerResult result = new OnlineRecognizerResult(h);
    DestroyResult(h);
    return result;
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

  [DllImport(dllName)]
  public static extern int IsReady(IntPtr handle, IntPtr stream);

  [DllImport(dllName, EntryPoint="Decode")]
  public static extern void Decode(IntPtr handle, IntPtr stream);

  [DllImport(dllName)]
  public static extern IntPtr GetResult(IntPtr handle, IntPtr stream);

  [DllImport(dllName)]
  public static extern void DestroyResult(IntPtr result);
}

class OnlineStream : IDisposable {
  public OnlineStream(IntPtr p) {
    _handle = p;
  }

  public void AcceptWaveform(float sampleRate, float[] samples) {
    AcceptWaveform(_handle, sampleRate, samples, samples.Length);
  }
  public void InputFinished() {
    InputFinished(_handle);
  }

  public void Dispose() {
    Dispose(disposing: true);
    GC.SuppressFinalize(this);
  }

  protected virtual void Dispose(bool disposing) {
    // disposing is not used
    if(!this.disposed) {
      DestroyOnlineStream(_handle);
      _handle = IntPtr.Zero;
      disposed = true;
    }
  }

  ~OnlineStream() {
    Dispose(disposing: false);
  }

  private IntPtr _handle;
  public IntPtr Handle => _handle;

  private bool disposed = false;

  private const string dllName = "sherpa-ncnn-c-api.dll";

  [DllImport(dllName, EntryPoint="DestroyStream")]
  public static extern void DestroyOnlineStream(IntPtr handle);

  [DllImport(dllName)]
  public static extern void AcceptWaveform(IntPtr handle, float sampleRate, float[] samples, int n);

  [DllImport(dllName)]
  public static extern void InputFinished(IntPtr handle);
}

class OnlineRecognizerResult {
  public OnlineRecognizerResult(IntPtr handle) {
    Impl impl = (Impl)Marshal.PtrToStructure(handle, typeof(Impl));
    _text = Marshal.PtrToStringAnsi(impl.Text);
  }

  [StructLayout(LayoutKind.Sequential)]
  struct Impl {
    public IntPtr Text;
    public IntPtr Tokens;
    public IntPtr Timestamps;
    int Count;
  }

  private String _text;
  public String Text => _text;

}

}
