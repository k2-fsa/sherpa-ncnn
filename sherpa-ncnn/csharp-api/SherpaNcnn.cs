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
// https://www.mono-project.com/docs/advanced/pinvoke/#gc-safe-pinvoke-code
// https://www.mono-project.com/docs/advanced/pinvoke/#properly-disposing-of-resources
public class OnlineRecognizer : IDisposable {
 public OnlineRecognizer(OnlineRecognizerConfig config) {
    IntPtr h = CreateOnlineRecognizer(ref config);
    _handle = new HandleRef(this, h);
  }

  public OnlineStream CreateStream() {
    IntPtr p = CreateOnlineStream(_handle.Handle);
    return new OnlineStream(p);
  }

  public bool IsReady(OnlineStream stream) {
    return IsReady(_handle.Handle, stream.Handle) != 0;
  }

  public void Decode(OnlineStream stream) {
    Decode(_handle.Handle, stream.Handle);
  }

  public OnlineRecognizerResult GetResult(OnlineStream stream) {
    IntPtr h = GetResult(_handle.Handle, stream.Handle);
    OnlineRecognizerResult result = new OnlineRecognizerResult(h);
    DestroyResult(h);
    return result;
  }

  public void Dispose() {
    Cleanup();
    // Prevent the object from being placed on the
    // finalization queue
    System.GC.SuppressFinalize(this);
  }

  ~OnlineRecognizer() {
    Cleanup();
  }

  private void Cleanup() {
    DestroyOnlineRecognizer(_handle.Handle);

    // Don't permit the handle to be used again.
    _handle = new HandleRef(this, IntPtr.Zero);
  }

  private HandleRef _handle;

  // private const string dllName = "sherpa-ncnn-c-api.dll";
  private const string dllName = "sherpa-ncnn-c-api";

  [DllImport(dllName, EntryPoint="CreateRecognizer")]
  public static extern IntPtr CreateOnlineRecognizer(ref OnlineRecognizerConfig config);

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


public class OnlineStream : IDisposable {
  public OnlineStream(IntPtr p) {
    _handle = new HandleRef(this, p);
  }

  public void AcceptWaveform(float sampleRate, float[] samples) {
    AcceptWaveform(Handle, sampleRate, samples, samples.Length);
  }
  public void InputFinished() {
    InputFinished(Handle);
  }

  ~OnlineStream() {
    Cleanup();
  }

  public void Dispose() {
    Cleanup();
    // Prevent the object from being placed on the
    // finalization queue
    System.GC.SuppressFinalize(this);
  }

  private void Cleanup() {
    DestroyOnlineStream(Handle);

    // Don't permit the handle to be used again.
    _handle = new HandleRef(this, IntPtr.Zero);
  }

  private HandleRef _handle;
  public IntPtr Handle => _handle.Handle;

  // private const string dllName = "sherpa-ncnn-c-api.dll";
  private const string dllName = "sherpa-ncnn-c-api";

  [DllImport(dllName, EntryPoint="DestroyStream")]
  public static extern void DestroyOnlineStream(IntPtr handle);

  [DllImport(dllName)]
  public static extern void AcceptWaveform(IntPtr handle, float sampleRate, float[] samples, int n);

  [DllImport(dllName)]
  public static extern void InputFinished(IntPtr handle);
}

public class OnlineRecognizerResult {
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
