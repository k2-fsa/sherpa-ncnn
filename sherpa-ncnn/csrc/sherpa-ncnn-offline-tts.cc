// sherpa-ncnn/csrc/sherpa-ncnn-offline-tts.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include <chrono>  // NOLINT
#include <fstream>

#include "sherpa-ncnn/csrc/offline-tts.h"
#include "sherpa-ncnn/csrc/parse-options.h"
#include "sherpa-ncnn/csrc/wave-writer.h"

static int32_t AudioCallback(const float * /*samples*/, int32_t num_samples,
                             int32_t processed, int32_t total, void *arg) {
  float progress = static_cast<float>(processed) / total;
  printf("Progress=%.3f%%\n", progress * 100);

  return 1;
}

int main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
Offline/Non-streaming text-to-speech with sherpa-ncnn

Usage example:

wget https://github.com/k2-fsa/sherpa-ncnn/releases/download/tts-models/ncnn-vits-piper-en_US-amy-low.tar.bz2
tar xf ncnn-vits-piper-en_US-amy-low.tar.bz2

./bin/sherpa-ncnn-offline-tts \
 --vits-model-dir=./vits-piper-en_US-amy-low \
 --output-filename=./generated.wav \
 "Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar."

It will generate a file ./generated.wav as specified by --output-filename.

You can find more models at
https://github.com/k2-fsa/sherpa-ncnn/releases/tag/tts-models

Please see
https://k2-fsa.github.io/sherpa/ncnn/tts/index.html
or details.
)usage";

  sherpa_ncnn::ParseOptions po(kUsageMessage);
  std::string output_filename = "./generated.wav";
  int32_t sid = 0;

  po.Register("output-filename", &output_filename,
              "Path to save the generated audio");

  po.Register("sid", &sid,
              "Speaker ID. Used only for multi-speaker models, e.g., models "
              "trained using the VCTK dataset. Not used for single-speaker "
              "models, e.g., models trained using the LJSpeech dataset");

  sherpa_ncnn::OfflineTtsConfig config;

  config.Register(&po);
  po.Read(argc, argv);

  if (po.NumArgs() == 0) {
    fprintf(stderr, "Error: Please provide the text to generate audio.\n\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  if (po.NumArgs() > 1) {
    fprintf(stderr,
            "Error: Accept only one positional argument. Please use single "
            "quotes to wrap your text\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  if (config.model.debug) {
    fprintf(stderr, "%s\n", config.model.ToString().c_str());
  }

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    exit(EXIT_FAILURE);
  }

  sherpa_ncnn::OfflineTts tts(config);

  const auto begin = std::chrono::steady_clock::now();
  sherpa_ncnn::TtsArgs args;
  args.text = po.GetArg(1);
  args.tokens = std::vector<std::vector<int32_t>>{
      {1,  0, 20,  0, 121, 0, 14,  0, 100, 0, 3,  0, 51,  0, 122, 0,
       88, 0, 3,   0, 22,  0, 33,  0, 122, 0, 3,  0, 17,  0, 120, 0,
       33, 0, 122, 0, 74,  0, 44,  0, 13,  0, 19, 0, 39,  0, 26,  0,
       32, 0, 120, 0, 39,  0, 31,  0, 32,  0, 74, 0, 23,  0, 4,   0,
       20, 0, 121, 0, 14,  0, 100, 0, 3,   0, 50, 0, 15,  0, 120, 0,
       14, 0, 100, 0, 32,  0, 3,   0, 22,  0, 33, 0, 122, 0, 13,  2}};
  args.sid = sid;
  args.speed = 1.0;
  auto audio = tts.Generate(args, AudioCallback);
  const auto end = std::chrono::steady_clock::now();

  if (audio.samples.empty()) {
    fprintf(
        stderr,
        "Error in generating audio. Please read previous error messages.\n");
    exit(EXIT_FAILURE);
  }

  float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;
  float duration = audio.samples.size() / static_cast<float>(audio.sample_rate);

  float rtf = elapsed_seconds / duration;
  fprintf(stderr, "Number of threads: %d\n", config.model.num_threads);
  fprintf(stderr, "Elapsed seconds: %.3f s\n", elapsed_seconds);
  fprintf(stderr, "Audio duration: %.3f s\n", duration);
  fprintf(stderr, "Real-time factor (RTF): %.3f/%.3f = %.3f\n", elapsed_seconds,
          duration, rtf);

  bool ok = sherpa_ncnn::WriteWave(output_filename, audio.sample_rate,
                                   audio.samples.data(), audio.samples.size());
  if (!ok) {
    fprintf(stderr, "Failed to write wave to %s\n", output_filename.c_str());
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "The text is: %s. Speaker ID: %d\n", po.GetArg(1).c_str(),
          sid);
  fprintf(stderr, "Saved to %s successfully!\n", output_filename.c_str());

  return 0;
}
