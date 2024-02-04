#include <memory>

#include "sherpa-ncnn/c-api/c-api.h"
#include "sherpa-ncnn/csrc/recognizer.h"
#include "sherpa-ncnn/csrc/wave-reader.h"

// see also
// https://emscripten.org/docs/porting/connecting_cpp_and_javascript/Interacting-with-code.html

extern "C" {

sherpa_ncnn::Recognizer *g_recognizer = nullptr;
std::unique_ptr<sherpa_ncnn::Stream> g_stream;

static_assert(sizeof(SherpaNcnnModelConfig) == 4 * 9, "");

void MyTest2(SherpaNcnnModelConfig *model_config) {
  fprintf(stdout, "encoder_param: %s\n", model_config->encoder_param);
  fprintf(stdout, "encoder_bin: %s\n", model_config->encoder_bin);

  fprintf(stdout, "decoder_param: %s\n", model_config->decoder_param);
  fprintf(stdout, "decoder_bin: %s\n", model_config->decoder_bin);

  fprintf(stdout, "joiner_param: %s\n", model_config->joiner_param);
  fprintf(stdout, "joiner_bin: %s\n", model_config->joiner_bin);
  fprintf(stdout, "tokens: %s\n", model_config->tokens);
  fprintf(stdout, "use_vulkan_compute: %d\n", model_config->use_vulkan_compute);
  fprintf(stdout, "num_threads: %d\n", model_config->num_threads);
}

void MyTest(const float *samples, int32_t n) {
  fprintf(stdout, "n: %d\n", n);
  for (int32_t i = 0; i != n; ++i) {
    fprintf(stdout, "%d: %.2f\n", i, samples[i]);
  }
}

float *ReadData(int32_t *n) {
  bool is_ok = false;
  std::vector<float> samples = sherpa_ncnn::ReadWave("./1.wav", 16000, &is_ok);
  float *p = new float[samples.size()];
  std::copy(samples.begin(), samples.end(), p);
  *n = samples.size();
  fprintf(stdout, "n: %d\n", *n);
  return p;
}

static void FreeRecognizer() {
  if (g_recognizer) {
    delete g_recognizer;
    g_recognizer = nullptr;
  }
}

static void CreateRecognizer() {
  if (g_recognizer) {
    return;
  }

  sherpa_ncnn::RecognizerConfig config;
  std::string base = "./";
  config.model_config.tokens = base + "tokens.txt";
  // clang-format off
  config.model_config.encoder_param = base + "encoder_jit_trace-pnnx.ncnn.param";
  config.model_config.encoder_bin = base + "encoder_jit_trace-pnnx.ncnn.bin";
  config.model_config.decoder_param = base + "decoder_jit_trace-pnnx.ncnn.param";
  config.model_config.decoder_bin = base + "decoder_jit_trace-pnnx.ncnn.bin";
  config.model_config.joiner_param = base + "joiner_jit_trace-pnnx.ncnn.param";
  config.model_config.joiner_bin = base + "joiner_jit_trace-pnnx.ncnn.bin";
  // clang-format on

  int32_t num_threads = 2;
  config.model_config.encoder_opt.num_threads = num_threads;
  config.model_config.decoder_opt.num_threads = num_threads;
  config.model_config.joiner_opt.num_threads = num_threads;
  config.decoder_config.method = "greedy_search";

  float expected_sampling_rate = 16000;
  config.feat_config.sampling_rate = expected_sampling_rate;
  config.feat_config.feature_dim = 80;
  fprintf(stdout, "%s\n", config.ToString().c_str());
  g_recognizer = new sherpa_ncnn::Recognizer(config);
  fprintf(stdout, "created recognizer\n");
}

static void CreateStream() {
  if (!g_recognizer) {
    CreateRecognizer();
  }

  g_stream = g_recognizer->CreateStream();
}

void RunSherpaNcnn() {
  if (!g_recognizer) {
    CreateRecognizer();
  }

  fprintf(stdout, "decoding\n");
  float expected_sampling_rate = 16000;
  std::string wav_filename = "./1.wav";
  bool is_ok = false;
  std::vector<float> samples =
      sherpa_ncnn::ReadWave(wav_filename, expected_sampling_rate, &is_ok);
  if (!is_ok) {
    fprintf(stderr, "Failed to read %s\n", wav_filename.c_str());
    return;
  }
  const float duration = samples.size() / expected_sampling_rate;
  fprintf(stdout, "wav filename: %s\n", wav_filename.c_str());
  fprintf(stdout, "wav duration (s): %.2f\n", duration);

  auto begin = std::chrono::steady_clock::now();
  fprintf(stdout, "Started!\n");
  auto stream = g_recognizer->CreateStream();
  stream->AcceptWaveform(expected_sampling_rate, samples.data(),
                         samples.size());
  std::vector<float> tail_paddings(
      static_cast<int>(0.3 * expected_sampling_rate));
  stream->AcceptWaveform(expected_sampling_rate, tail_paddings.data(),
                         tail_paddings.size());

  while (g_recognizer->IsReady(stream.get())) {
    g_recognizer->DecodeStream(stream.get());
  }
  auto result = g_recognizer->GetResult(stream.get());
  fprintf(stdout, "Done!\n");

  fprintf(stdout, "Recognition result for %s\n%s\n", wav_filename.c_str(),
          result.ToString().c_str());

  auto end = std::chrono::steady_clock::now();
  float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;

  fprintf(stdout, "Elapsed seconds: %.3f s\n", elapsed_seconds);
  float rtf = elapsed_seconds / duration;
  fprintf(stdout, "Real time factor (RTF): %.3f / %.3f = %.3f\n",
          elapsed_seconds, duration, rtf);
}
}
