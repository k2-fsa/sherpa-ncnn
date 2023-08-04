package main

import (
	"errors"
	"fmt"
	"github.com/gordonklaus/portaudio"
	sherpa "github.com/k2-fsa/sherpa-ncnn-go/sherpa_ncnn"
	flag "github.com/spf13/pflag"
	"log"
	"os"
	"strings"
)

func main() {
	err := portaudio.Initialize()
	if err != nil {
		log.Fatalf("Unable to initialize portaudio: %v\n", err)
	}
	defer portaudio.Terminate()

	default_device, err := portaudio.DefaultInputDevice()
	if err != nil {
		log.Fatal("Failed to get default input device: %v\n", err)
	}
	fmt.Printf("Select default input device: %s\n", default_device.Name)
	param := portaudio.StreamParameters{}
	param.Input.Device = default_device
	param.Input.Channels = 1
	param.Input.Latency = default_device.DefaultLowInputLatency

	param.SampleRate = 16000
	param.FramesPerBuffer = 0
	param.Flags = portaudio.ClipOff

	config := sherpa.RecognizerConfig{}
	config.Feat = sherpa.FeatureConfig{SampleRate: 16000, FeatureDim: 80}

	flag.StringVar(&config.Model.EncoderParam, "encoder-param", "", "Path to the encoder.ncnn.param")
	flag.StringVar(&config.Model.EncoderBin, "encoder-bin", "", "Path to the encoder.ncnn.bin")
	flag.StringVar(&config.Model.DecoderParam, "decoder-param", "", "Path to the decoder.ncnn.param")
	flag.StringVar(&config.Model.DecoderBin, "decoder-bin", "", "Path to the decoder.ncnn.bin")
	flag.StringVar(&config.Model.JoinerParam, "joiner-param", "", "Path to the joiner.ncnn.param")
	flag.StringVar(&config.Model.JoinerBin, "joiner-bin", "", "Path to the joiner.ncnn.bin")

	flag.StringVar(&config.Model.Tokens, "tokens", "", "Path to the tokens file")
	flag.IntVar(&config.Model.NumThreads, "num-threads", 1, "Number of threads for computing")
	flag.StringVar(&config.Decoder.DecodingMethod, "decoding-method", "greedy_search", "Decoding method. Possible values: greedy_search, modified_beam_search")
	flag.IntVar(&config.Decoder.NumActivePaths, "num-active-paths", 4, "Used only when --decoding-method is modified_beam_search")

	flag.IntVar(&config.EnableEndpoint, "enable-endpoint", 1, "Whether to enable endpoint")
	flag.Float32Var(&config.Rule1MinTrailingSilence, "rule1-min-trailing-silence", 2.4, "Threshold for rule1")
	flag.Float32Var(&config.Rule2MinTrailingSilence, "rule2-min-trailing-silence", 1.2, "Threshold for rule2")
	flag.Float32Var(&config.Rule3MinUtteranceLength, "rule3-min-utterance-length", 20, "Threshold for rule3")

	flag.Parse()

	checkConfig(&config)

	log.Println("Initializing recognizer")
	recognizer := sherpa.NewRecognizer(&config)
	log.Println("Recognizer created!")
	defer sherpa.DeleteRecognizer(recognizer)

	stream := sherpa.NewStream(recognizer)
	defer sherpa.DeleteStream(stream)

	// you can choose another value for 0.1 if you want
	samplesPerCall := int32(param.SampleRate * 0.1) // 0.1 second

	samples := make([]float32, samplesPerCall)

	s, err := portaudio.OpenStream(param, samples)
	if err != nil {
		log.Fatalf("Failed to open the stream")
	}
	defer s.Close()
	chk(s.Start())

	var last_text string

	segment_idx := 0

	fmt.Println("Started! Please speak")

	for {
		chk(s.Read())
		stream.AcceptWaveform(int(param.SampleRate), samples)

		for recognizer.IsReady(stream) {
			recognizer.Decode(stream)
		}

		text := recognizer.GetResult(stream).Text
		if len(text) != 0 && last_text != text {
			last_text = strings.ToLower(text)
			fmt.Printf("\r%d: %s", segment_idx, last_text)
		}

		if recognizer.IsEndpoint(stream) {
			if len(text) != 0 {
				segment_idx++
				fmt.Println()
			}
			recognizer.Reset(stream)
		}
	}

	chk(s.Stop())
}

func chk(err error) {
	if err != nil {
		panic(err)
	}
}

func checkConfig(config *sherpa.RecognizerConfig) {
	// --encoder-param
	if config.Model.EncoderParam == "" {
		log.Fatal("Please provide --encoder-param")
	}

	if _, err := os.Stat(config.Model.EncoderParam); errors.Is(err, os.ErrNotExist) {
		log.Fatalf("--encoder-param %v does not exist", config.Model.EncoderParam)
	}

	// --encoder-bin
	if config.Model.EncoderBin == "" {
		log.Fatal("Please provide --encoder-bin")
	}

	if _, err := os.Stat(config.Model.EncoderBin); errors.Is(err, os.ErrNotExist) {
		log.Fatalf("--encoder-bin %v does not exist", config.Model.EncoderBin)
	}

	// --decoder-param
	if config.Model.DecoderParam == "" {
		log.Fatal("Please provide --decoder-param")
	}

	if _, err := os.Stat(config.Model.DecoderParam); errors.Is(err, os.ErrNotExist) {
		log.Fatalf("--decoder-param %v does not exist", config.Model.DecoderParam)
	}

	// --decoder-bin
	if config.Model.DecoderBin == "" {
		log.Fatal("Please provide --decoder-bin")
	}

	if _, err := os.Stat(config.Model.DecoderBin); errors.Is(err, os.ErrNotExist) {
		log.Fatalf("--decoder-bin %v does not exist", config.Model.DecoderBin)
	}

	// --joiner-param
	if config.Model.JoinerParam == "" {
		log.Fatal("Please provide --joiner-param")
	}

	if _, err := os.Stat(config.Model.JoinerParam); errors.Is(err, os.ErrNotExist) {
		log.Fatalf("--joiner-param %v does not exist", config.Model.JoinerParam)
	}

	// --joiner-bin
	if config.Model.JoinerBin == "" {
		log.Fatal("Please provide --joiner-bin")
	}

	if _, err := os.Stat(config.Model.JoinerBin); errors.Is(err, os.ErrNotExist) {
		log.Fatalf("--joiner-bin %v does not exist", config.Model.JoinerBin)
	}

	// --tokens
	if config.Model.Tokens == "" {
		log.Fatal("Please provide --tokens")
	}

	if _, err := os.Stat(config.Model.Tokens); errors.Is(err, os.ErrNotExist) {
		log.Fatalf("--tokens %v does not exist", config.Model.Tokens)
	}
}
