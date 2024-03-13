using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

public class SpeechToText : MonoBehaviour
{

    private SherpaNcnn.OnlineRecognizer recognizer;
    private SherpaNcnn.OnlineStream onlineStream;
    private int segmentIndex = 0;
    private string lastText = "";
    [SerializeField]
    public Text Text;

    public string tokensPath;
    public string encoderParamPath;
    public string encoderBinPath;
    public string decoderParamPath;
    public string decoderBinPath;
    public string joinerParamPath;
    public string joinerBinPath;
    public int numThreads = 1;
    public string decodingMethod = "greedy_search";

    void Start()
    {

        SherpaNcnn.OnlineRecognizerConfig config = new SherpaNcnn.OnlineRecognizerConfig
        {
            FeatConfig = { SampleRate = 16000, FeatureDim = 80 },
            ModelConfig = {
                Tokens = Path.Combine(Application.streamingAssetsPath,tokensPath),
                EncoderParam =  Path.Combine(Application.streamingAssetsPath,encoderParamPath),
                EncoderBin =Path.Combine(Application.streamingAssetsPath, encoderBinPath),
                DecoderParam =Path.Combine(Application.streamingAssetsPath, decoderParamPath),
                DecoderBin = Path.Combine(Application.streamingAssetsPath, decoderBinPath),
                JoinerParam = Path.Combine(Application.streamingAssetsPath,joinerParamPath),
                JoinerBin =Path.Combine(Application.streamingAssetsPath,joinerBinPath),
                UseVulkanCompute = 0,
                NumThreads = numThreads
            },
            DecoderConfig = {
                DecodingMethod = decodingMethod,
                NumActivePaths = 4
            },
            EnableEndpoint = 1,
            Rule1MinTrailingSilence = 2.4F,
            Rule2MinTrailingSilence = 1.2F,
            Rule3MinUtteranceLength = 20.0F
        };

        recognizer = new SherpaNcnn.OnlineRecognizer(config);
        onlineStream = recognizer.CreateStream();


        StartMicrophoneCapture();
    }

    void Update()
    {
        if (!Microphone.IsRecording(null)) return;
        int currentPosition = Microphone.GetPosition(null);
        int sampleCount = currentPosition - lastSamplePosition;
        if (sampleCount < 0)
        {
            sampleCount += micClip.samples * micClip.channels;
        }

        if (sampleCount > 0)
        {
            float[] samples = new float[sampleCount];
            micClip.GetData(samples, lastSamplePosition);


            onlineStream.AcceptWaveform(micClip.frequency, samples);


            lastSamplePosition = currentPosition;
        }

        if (recognizer.IsReady(onlineStream))
        {
            recognizer.Decode(onlineStream);
        }

        var text = recognizer.GetResult(onlineStream).Text;
        bool isEndpoint = recognizer.IsEndpoint(onlineStream);
        if (!string.IsNullOrWhiteSpace(text) && lastText != text)
        {
            lastText = text;
            Debug.Log($"{segmentIndex}: {lastText}");
            Text.text = lastText;
        }

        if (isEndpoint)
        {
            if (!string.IsNullOrWhiteSpace(text))
            {
                ++segmentIndex;
            }
            recognizer.Reset(onlineStream);
        }
    }
    private AudioClip micClip;
    private int lastSamplePosition = 0;
    private void StartMicrophoneCapture()
    {

        string device = Microphone.devices.Length > 0 ? Microphone.devices[0] : null;
        micClip = Microphone.Start(device, true, 10, 16000);
        while (!(Microphone.GetPosition(device) > 0)) { }
        lastSamplePosition = Microphone.GetPosition(device);
    }

    private void OnDestroy()
    {
        Microphone.End(null);
    }
}
