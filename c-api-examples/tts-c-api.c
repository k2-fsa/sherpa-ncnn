/**
 * Copyright (c) 2026 Yantai Xiaoyingtao Technology. (authors: Seven Du)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <stdio.h>

#include "sherpa-ncnn/c-api/c-api.h"

const char* kUsage =
    "\n"
    "Usage:\n"
    "    wget "
    "https://github.com/k2-fsa/sherpa-ncnn/releases/download/tts-models/"
    "ncnn-vits-piper-en_US-amy-low.tar.bz2\n"
    "    tar xf ncnn-vits-piper-en_US-amy-low.tar.bz2\n"
    "\n\n"
    "  ./bin/tts-c-api \\\n"
    "    generated.pcm \\\n"
    "    ./ncnn-vits-piper-en_US-amy-low \\\n"
    "    \"hello, how do you do?\"\n"
    "\n\n"
    ""
    "You can find more models at:\n"
    "https://github.com/k2-fsa/sherpa-ncnn/releases/tag/tts-models\n"
    "\n\n";

int main(int32_t argc, char* argv[]) {
  if (argc != 4) {
    fprintf(stderr, "%s\n", kUsage);
    return -1;
  }

  const char* text = argv[3];
  SherpaNcnnTtsConfig config = {0};

  config.model_dir = argv[2];
  config.sid = 0;
  config.speed = 1.0;
  config.debug = 0;
  // config.n_threads = 4;

  SherpaNcnnTts* tts = SherpaNcnnCreateTts(&config);
  if (!tts) {
    fprintf(stderr, "Failed to create TTS\n");
    return -1;
  }

  SherpaNcnnTtsAudio* audio = SherpaNcnnTtsGenerate(tts, text, &config);

  if (!audio) {
    fprintf(stderr, "Failed to generate audio\n");
    goto err;
  }

  const char* pcm_filename = argv[1];
  FILE* fp = fopen(pcm_filename, "wb");
  if (!fp) {
    fprintf(stderr, "Failed to open %s\n", pcm_filename);
    goto err;
  }

  size_t expected = SherpaNcnnTtsAudioGetSampleCount(audio);
  size_t written = fwrite(SherpaNcnnTtsAudioGetSamplePtr(audio), sizeof(float),
                            expected, fp);
  if (written != expected) {
    fprintf(stderr, "Failed to write all PCM samples\n");
    fclose(fp);
    goto err;
  }
  fclose(fp);

  fprintf(stderr,
          "File name: %s\n"
          "Sample rate: %d\n"
          "Sample Count: %u\n"
          "Duration: %.3fs\n"
          "Elapsed Seconds: %.3fs\n"
          "Real-Time factor: %.3f\n",
          pcm_filename, SherpaNcnnTtsAudioGetSampleRate(audio),
          SherpaNcnnTtsAudioGetSampleCount(audio),
          SherpaNcnnTtsAudioGetDuration(audio),
          SherpaNcnnTtsAudioGetElapsedSeconds(audio),
          SherpaNcnnTtsAudioGetRtf(audio));

  fprintf(stderr, "\n\n");
  fprintf(stderr, "apt install sox\n");
  fprintf(stderr, "sox -t raw -r %d -e float -b 32 -c 1 %s %s.wav\n",
          SherpaNcnnTtsAudioGetSampleRate(audio), pcm_filename, pcm_filename);
  fprintf(stderr, "\n");

  SherpaNcnnTtsAudioDestroy(&audio);
  SherpaNcnnTtsDestroy(&tts);

  return 0;

err:
  SherpaNcnnTtsAudioDestroy(&audio);
  SherpaNcnnTtsDestroy(&tts);
  return -1;
}
