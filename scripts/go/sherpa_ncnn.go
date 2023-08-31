/*
Speech recognition with [Next-gen Kaldi].

[sherpa-ncnn] is an open-source speech recognition framework for [Next-gen Kaldi].
It depends only on [ncnn], supporting both streaming and non-streaming
speech recognition.

It does not need to access the network during recognition and everything
runs locally.

It supports a variety of platforms, such as Linux (x86_64, aarch64, arm),
Windows (x86_64, x86), macOS (x86_64, arm64), RISC-V, etc.

Usage examples:

 1. Real-time speech recognition from a microphone

    Please see
    https://github.com/k2-fsa/sherpa-ncnn/tree/master/go-api-examples/real-time-speech-recognition-from-microphone

 2. Decode a file

    Please see
    https://github.com/k2-fsa/sherpa-ncnn/tree/master/go-api-examples/decode-file

[sherpa-ncnn]: https://github.com/k2-fsa/sherpa-ncnn
[ncnn]: https://github.com/tencent/ncnn
[Next-gen Kaldi]: https://github.com/k2-fsa/
*/
package sherpa_ncnn

// #include <stdlib.h>
// #include "c-api.h"
import "C"
import "unsafe"

// Please refer to
// https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/
// to download pre-trained models
type ModelConfig struct {
	EncoderParam string // Path to the encoder.ncnn.param
	EncoderBin   string // Path to the encoder.ncnn.bin
	DecoderParam string // Path to the decoder.ncnn.param
	DecoderBin   string // Path to the decoder.ncnn.bin
	JoinerParam  string // Path to the joiner.ncnn.param
	JoinerBin    string // Path to the joiner.ncnn.bin
	Tokens       string // Path to tokens.txt
	NumThreads   int    // Number of threads to use for neural network computation
}

// Configuration for the feature extractor
type FeatureConfig struct {
	// Sample rate expected by the model. It is 16000 for all
	// pre-trained models provided by us
	SampleRate int
	// Feature dimension expected by the model. It is 80 for all
	// pre-trained models provided by us
	FeatureDim int
}

// Configuration for the beam search decoder
type DecoderConfig struct {
	// Decoding method. Supported values are:
	// greedy_search, modified_beam_search
	DecodingMethod string

	// Number of active paths for modified_beam_search.
	// It is ignored when decoding_method is greedy_search.
	NumActivePaths int
}

// Configuration for the online/streaming recognizer.
type RecognizerConfig struct {
	Feat    FeatureConfig
	Model   ModelConfig
	Decoder DecoderConfig

	EnableEndpoint int // 1 to enable endpoint detection.

	// Please see
	// https://k2-fsa.github.io/sherpa/ncnn/endpoint.html
	// for the meaning of Rule1MinTrailingSilence, Rule2MinTrailingSilence
	// and Rule3MinUtteranceLength.
	Rule1MinTrailingSilence float32
	Rule2MinTrailingSilence float32
	Rule3MinUtteranceLength float32

	HotwordsFile  string
	HotwordsScore float32
}

// It contains the recognition result for a online stream.
type RecognizerResult struct {
	Text string
}

// The online recognizer class. It wraps a pointer from C.
type Recognizer struct {
	impl *C.struct_SherpaNcnnRecognizer
}

// The online stream class. It wraps a pointer from C.
type Stream struct {
	impl *C.struct_SherpaNcnnStream
}

// Free the internal pointer inside the recognizer to avoid memory leak.
func DeleteRecognizer(recognizer *Recognizer) {
	C.DestroyRecognizer(recognizer.impl)
	recognizer.impl = nil
}

// The user is responsible to invoke [DeleteRecognizer]() to free
// the returned recognizer to avoid memory leak
func NewRecognizer(config *RecognizerConfig) *Recognizer {
	c := C.struct_SherpaNcnnRecognizerConfig{}
	c.feat_config.sampling_rate = C.float(config.Feat.SampleRate)
	c.feat_config.feature_dim = C.int(config.Feat.FeatureDim)

	c.model_config.encoder_param = C.CString(config.Model.EncoderParam)
	defer C.free(unsafe.Pointer(c.model_config.encoder_param))

	c.model_config.encoder_bin = C.CString(config.Model.EncoderBin)
	defer C.free(unsafe.Pointer(c.model_config.encoder_bin))

	c.model_config.decoder_param = C.CString(config.Model.DecoderParam)
	defer C.free(unsafe.Pointer(c.model_config.decoder_param))

	c.model_config.decoder_bin = C.CString(config.Model.DecoderBin)
	defer C.free(unsafe.Pointer(c.model_config.decoder_bin))

	c.model_config.joiner_param = C.CString(config.Model.JoinerParam)
	defer C.free(unsafe.Pointer(c.model_config.joiner_param))

	c.model_config.joiner_bin = C.CString(config.Model.JoinerBin)
	defer C.free(unsafe.Pointer(c.model_config.joiner_bin))

	c.model_config.tokens = C.CString(config.Model.Tokens)
	defer C.free(unsafe.Pointer(c.model_config.tokens))

	c.model_config.use_vulkan_compute = C.int(0)
	c.model_config.num_threads = C.int(config.Model.NumThreads)

	c.decoder_config.decoding_method = C.CString(config.Decoder.DecodingMethod)
	defer C.free(unsafe.Pointer(c.decoder_config.decoding_method))

	c.decoder_config.num_active_paths = C.int(config.Decoder.NumActivePaths)

	c.enable_endpoint = C.int(config.EnableEndpoint)
	c.rule1_min_trailing_silence = C.float(config.Rule1MinTrailingSilence)
	c.rule2_min_trailing_silence = C.float(config.Rule2MinTrailingSilence)
	c.rule3_min_utterance_length = C.float(config.Rule3MinUtteranceLength)

	c.hotwords_file = C.CString(config.HotwordsFile)
	defer C.free(unsafe.Pointer(c.hotwords_file))

	c.hotwords_score = C.float(config.HotwordsScore)

	recognizer := &Recognizer{}
	recognizer.impl = C.CreateRecognizer(&c)

	return recognizer
}

// Delete the internal pointer inside the stream to avoid memory leak.
func DeleteStream(stream *Stream) {
	C.DestroyStream(stream.impl)
	stream.impl = nil
}

// The user is responsible to invoke [DeleteStream]() to free
// the returned stream to avoid memory leak
func NewStream(recognizer *Recognizer) *Stream {
	stream := &Stream{}
	stream.impl = C.CreateStream(recognizer.impl)
	return stream
}

// Input audio samples for the stream.
//
// sampleRate is the actual sample rate of the input audio samples. If it
// is different from the sample rate expected by the feature extractor, we will
// do resampling inside.
//
// samples contains audio samples. Each sample is in the range [-1, 1]
func (s *Stream) AcceptWaveform(sampleRate int, samples []float32) {
	C.AcceptWaveform(s.impl, C.float(sampleRate), (*C.float)(&samples[0]), C.int(len(samples)))
}

// Signal that there will be no incoming audio samples.
// After calling this function, you cannot call [Stream.AcceptWaveform] any longer.
//
// The main purpose of this function is to flush the remaining audio samples
// buffered inside for feature extraction.
func (s *Stream) InputFinished() {
	C.InputFinished(s.impl)
}

// Check whether the stream has enough feature frames for decoding.
// Return true if this stream is ready for decoding. Return false otherwise.
//
// You will usually use it like below:
//
//	for recognizer.IsReady(s) {
//	   recognizer.Decode(s)
//	}
func (recognizer *Recognizer) IsReady(s *Stream) bool {
	return C.IsReady(recognizer.impl, s.impl) == 1
}

// Return true if an endpoint is detected.
//
// You usually use it like below:
//
//	if recognizer.IsEndpoint(s) {
//	   // do your own stuff after detecting an endpoint
//
//	   recognizer.Reset(s)
//	}
func (recognizer *Recognizer) IsEndpoint(s *Stream) bool {
	return C.IsEndpoint(recognizer.impl, s.impl) == 1
}

// After calling this function, the internal neural network model states
// are reset and IsEndpoint(s) would return false. GetResult(s) would also
// return an empty string.
func (recognizer *Recognizer) Reset(s *Stream) {
	C.Reset(recognizer.impl, s.impl)
}

// Decode the stream. Before calling this function, you have to ensure
// that recognizer.IsReady(s) returns true. Otherwise, you will be SAD.
//
// You usually use it like below:
//
//	for recognizer.IsReady(s) {
//	  recognizer.Decode(s)
//	}
func (recognizer *Recognizer) Decode(s *Stream) {
	C.Decode(recognizer.impl, s.impl)
}

// Get the current result of stream since the last invoke of Reset()
func (recognizer *Recognizer) GetResult(s *Stream) *RecognizerResult {
	p := C.GetResult(recognizer.impl, s.impl)
	defer C.DestroyResult(p)
	result := &RecognizerResult{}
	result.Text = C.GoString(p.text)

	return result
}
