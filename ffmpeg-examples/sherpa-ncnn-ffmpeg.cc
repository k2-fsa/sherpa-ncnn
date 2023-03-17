/**
 * Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
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
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cctype>  // std::tolower
#include <string>

#include "sherpa-ncnn/csrc/display.h"
#include "sherpa-ncnn/csrc/recognizer.h"

/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2010 Nicolas George
 * Copyright (c) 2011 Stefano Sabatini
 * Copyright (c) 2012 Clément Bœsch
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * @file audio decoding and filtering usage example
 * @example sherpa-ncnn-ffmpeg.c
 *
 * Demux, decode and filter audio input file, generate a raw audio
 * file to be played with ffplay.
 */

#include <unistd.h>
#ifdef __cplusplus
extern "C" {
#endif
#include <libavcodec/avcodec.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libavutil/samplefmt.h>
#ifdef __cplusplus
}
#endif

static const char *filter_descr =
    "aresample=16000,aformat=sample_fmts=s16:channel_layouts=mono";

static AVFormatContext *fmt_ctx;
static AVCodecContext *dec_ctx;
AVFilterContext *buffersink_ctx;
AVFilterContext *buffersrc_ctx;
AVFilterGraph *filter_graph;
static int32_t audio_stream_index = -1;

static int32_t FFmpegOpenInputFile(const char *filename) {
  const AVCodec *dec;
  int32_t ret;

  if ((ret = avformat_open_input(&fmt_ctx, filename, NULL, NULL)) < 0) {
    av_log(NULL, AV_LOG_ERROR, "Cannot open input file %s\n", filename);
    return ret;
  }

  if ((ret = avformat_find_stream_info(fmt_ctx, NULL)) < 0) {
    av_log(NULL, AV_LOG_ERROR, "Cannot find stream information\n");
    return ret;
  }

  /* select the audio stream */
  ret = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, &dec, 0);
  if (ret < 0) {
    av_log(NULL, AV_LOG_ERROR,
           "Cannot find an audio stream in the input file\n");
    return ret;
  }
  audio_stream_index = ret;

  /* create decoding context */
  dec_ctx = avcodec_alloc_context3(dec);
  if (!dec_ctx) return AVERROR(ENOMEM);
  avcodec_parameters_to_context(dec_ctx,
                                fmt_ctx->streams[audio_stream_index]->codecpar);

  /* init the audio decoder */
  if ((ret = avcodec_open2(dec_ctx, dec, NULL)) < 0) {
    av_log(NULL, AV_LOG_ERROR, "Cannot open audio decoder\n");
    return ret;
  }

  return 0;
}

static int32_t FFmpegInitFilters(const char *filters_descr) {
  char args[512];
  int32_t ret = 0;
  const AVFilter *abuffersrc = avfilter_get_by_name("abuffer");
  const AVFilter *abuffersink = avfilter_get_by_name("abuffersink");
  AVFilterInOut *outputs = avfilter_inout_alloc();
  AVFilterInOut *inputs = avfilter_inout_alloc();
  static const enum AVSampleFormat out_sample_fmts[] = {AV_SAMPLE_FMT_S16,
                                                        AV_SAMPLE_FMT_NONE};
  static const int32_t out_sample_rates[] = {16000, -1};
  const AVFilterLink *outlink;
  AVRational time_base = fmt_ctx->streams[audio_stream_index]->time_base;

  filter_graph = avfilter_graph_alloc();
  if (!outputs || !inputs || !filter_graph) {
    ret = AVERROR(ENOMEM);
    goto end;
  }

  /* buffer audio source: the decoded frames from the decoder will be inserted
   * here. */
  if (dec_ctx->ch_layout.order == AV_CHANNEL_ORDER_UNSPEC)
    av_channel_layout_default(&dec_ctx->ch_layout,
                              dec_ctx->ch_layout.nb_channels);
  ret = snprintf(args, sizeof(args),
                 "time_base=%d/%d:sample_rate=%d:sample_fmt=%s:channel_layout=",
                 time_base.num, time_base.den, dec_ctx->sample_rate,
                 av_get_sample_fmt_name(dec_ctx->sample_fmt));
  av_channel_layout_describe(&dec_ctx->ch_layout, args + ret,
                             sizeof(args) - ret);
  ret = avfilter_graph_create_filter(&buffersrc_ctx, abuffersrc, "in", args,
                                     NULL, filter_graph);
  if (ret < 0) {
    av_log(NULL, AV_LOG_ERROR, "Cannot create audio buffer source\n");
    goto end;
  }

  /* buffer audio sink: to terminate the filter chain. */
  ret = avfilter_graph_create_filter(&buffersink_ctx, abuffersink, "out", NULL,
                                     NULL, filter_graph);
  if (ret < 0) {
    av_log(NULL, AV_LOG_ERROR, "Cannot create audio buffer sink\n");
    goto end;
  }

  ret = av_opt_set_int_list(buffersink_ctx, "sample_fmts", out_sample_fmts, -1,
                            AV_OPT_SEARCH_CHILDREN);
  if (ret < 0) {
    av_log(NULL, AV_LOG_ERROR, "Cannot set output sample format\n");
    goto end;
  }

  ret =
      av_opt_set(buffersink_ctx, "ch_layouts", "mono", AV_OPT_SEARCH_CHILDREN);
  if (ret < 0) {
    av_log(NULL, AV_LOG_ERROR, "Cannot set output channel layout\n");
    goto end;
  }

  ret = av_opt_set_int_list(buffersink_ctx, "sample_rates", out_sample_rates,
                            -1, AV_OPT_SEARCH_CHILDREN);
  if (ret < 0) {
    av_log(NULL, AV_LOG_ERROR, "Cannot set output sample rate\n");
    goto end;
  }

  /*
   * Set the endpoints for the filter graph. The filter_graph will
   * be linked to the graph described by filters_descr.
   */

  /*
   * The buffer source output must be connected to the input pad of
   * the first filter described by filters_descr; since the first
   * filter input label is not specified, it is set to "in" by
   * default.
   */
  outputs->name = av_strdup("in");
  outputs->filter_ctx = buffersrc_ctx;
  outputs->pad_idx = 0;
  outputs->next = NULL;

  /*
   * The buffer sink input must be connected to the output pad of
   * the last filter described by filters_descr; since the last
   * filter output label is not specified, it is set to "out" by
   * default.
   */
  inputs->name = av_strdup("out");
  inputs->filter_ctx = buffersink_ctx;
  inputs->pad_idx = 0;
  inputs->next = NULL;

  if ((ret = avfilter_graph_parse_ptr(filter_graph, filters_descr, &inputs,
                                      &outputs, NULL)) < 0)
    goto end;

  if ((ret = avfilter_graph_config(filter_graph, NULL)) < 0) goto end;

  /* Print summary of the sink buffer
   * Note: args buffer is reused to store channel layout string */
  outlink = buffersink_ctx->inputs[0];
  av_channel_layout_describe(&outlink->ch_layout, args, sizeof(args));
  fprintf(
      stdout,
      "Event:FFmpeg: Detect audio stream ok, srate:%dHz fmt:%s chlayout:%s\n",
      (int)outlink->sample_rate,
      (char *)av_x_if_null(
          av_get_sample_fmt_name((AVSampleFormat)outlink->format), "?"),
      args);
  fflush(stdout);

end:
  avfilter_inout_free(&inputs);
  avfilter_inout_free(&outputs);

  return ret;
}

static void FFmpegDecodeFrame(const AVFrame *frame,
                              const sherpa_ncnn::Recognizer &recognizer,
                              sherpa_ncnn::Stream *s,
                              sherpa_ncnn::Display *display,
                              std::string *last_text, int32_t *segment_index,
                              int32_t *zero_samples) {
  // TODO: FIXME: Can we directly consume frame by s without buffer?
#define N 3200  // 0.2 s. Sample rate is fixed to 16 kHz
  static float samples[N];
  static int32_t nb_samples = 0;
  const int16_t *p = (int16_t *)frame->data[0];

  if (frame->nb_samples + nb_samples >= N) {
    s->AcceptWaveform(16000, samples, nb_samples);

    while (recognizer.IsReady(s)) {
      recognizer.DecodeStream(s);
    }

    bool is_endpoint = recognizer.IsEndpoint(s);
    auto text = recognizer.GetResult(s).text;

    if (!text.empty() && *last_text != text) {
      *last_text = text;

      std::transform(text.begin(), text.end(), text.begin(),
                     [](auto c) { return std::tolower(c); });

      display->Print(*segment_index, text);
    }

    if (is_endpoint) {
      if (!text.empty()) {
        (*segment_index)++;
      }

      recognizer.Reset(s);
    }

    nb_samples = 0;
  }

  for (int32_t i = 0; i < frame->nb_samples; i++) {
    if (p[i] == 0) {
      (*zero_samples)++;
    }
    samples[nb_samples++] = p[i] / 32768.;
  }
}

static inline char *FFmpegAvError2String(int32_t errnum) {
  static char str[AV_ERROR_MAX_STRING_SIZE];
  memset(str, 0, sizeof(str));
  return av_make_error_string(str, AV_ERROR_MAX_STRING_SIZE, errnum);
}

// When stream unpublish, use this signal to notify application.
static int32_t signal_unpublish_sigusr1 = 0;

static void Handler(int32_t sig) {
  if (sig == SIGUSR1) {
    fprintf(stdout, "\nEvent:Signal: Got signal %d\n", sig);
    fflush(stdout);
    signal_unpublish_sigusr1 = 1;
    return;
  }

  fprintf(stdout, "\nEvent:Signal: Caught Ctrl + C. Exiting...\n");
  fflush(stdout);

  signal(sig, SIG_DFL);
  raise(sig);
};

#define SET_STRING_BY_ENV(config, key) \
  if (getenv(key)) {                   \
    config = getenv(key);              \
  }

#define SET_CONFIG_BY_ENV(config, key, required) \
  config = "";                                   \
  SET_STRING_BY_ENV(config, key);                \
  if (!(config).empty() && required) {           \
    parsed_required_envs++;                      \
  }

#define SET_INTEGER_BY_ENV(config, key)                  \
  {                                                      \
    std::string val;                                     \
    SET_STRING_BY_ENV(val, "SHERPA_NCNN_ASD_ENDPOINTS"); \
    if (!val.empty() && ::atoi(val.c_str()) > 0) {       \
      config = ::atoi(val.c_str());                      \
    }                                                    \
  }

static int32_t ParseConfigFromENV(sherpa_ncnn::RecognizerConfig *config,
                                  std::string *input_url) {
  int32_t parsed_required_envs = 0;

  sherpa_ncnn::ModelConfig &mc = config->model_config;
  SET_CONFIG_BY_ENV(mc.tokens, "SHERPA_NCNN_TOKENS", true);
  SET_CONFIG_BY_ENV(mc.encoder_param, "SHERPA_NCNN_ENCODER_PARAM", true);
  SET_CONFIG_BY_ENV(mc.encoder_bin, "SHERPA_NCNN_ENCODER_BIN", true);
  SET_CONFIG_BY_ENV(mc.decoder_param, "SHERPA_NCNN_DECODER_PARAM", true);
  SET_CONFIG_BY_ENV(mc.decoder_bin, "SHERPA_NCNN_DECODER_BIN", true);
  SET_CONFIG_BY_ENV(mc.joiner_param, "SHERPA_NCNN_JOINER_PARAM", true);
  SET_CONFIG_BY_ENV(mc.joiner_bin, "SHERPA_NCNN_JOINER_BIN", true);
  SET_CONFIG_BY_ENV(*input_url, "SHERPA_NCNN_INPUT_URL", true);

  std::string val;
  SET_CONFIG_BY_ENV(val, "SHERPA_NCNN_NUM_THREADS", false);
  if (!val.empty()) {
    if (atoi(val.c_str()) <= 0) {
      fprintf(stderr, "Invalid SHERPA_NCNN_NUM_THREADS=%s\n", val.c_str());
      return -1;
    }
    mc.encoder_opt.num_threads = atoi(val.c_str());
    mc.decoder_opt.num_threads = atoi(val.c_str());
    mc.joiner_opt.num_threads = atoi(val.c_str());
  }

  SET_CONFIG_BY_ENV(val, "SHERPA_NCNN_METHOD", false);
  if (!val.empty()) {
    if (val != "greedy_search" && val != "modified_beam_search") {
      fprintf(stderr, "Invalid SHERPA_NCNN_METHOD=%s\n", val.c_str());
      return -1;
    }
    config->decoder_config.method = val;
  }

  SET_CONFIG_BY_ENV(val, "SHERPA_NCNN_ENABLE_ENDPOINT", false);
  if (!val.empty()) {
    std::transform(val.begin(), val.end(), val.begin(),
                   [](auto c) { return std::tolower(c); });
    config->enable_endpoint = val == "true" || val == "on";
  }

  SET_CONFIG_BY_ENV(val, "SHERPA_NCNN_RULE1_MIN_TRAILING_SILENCE", false);
  if (!val.empty()) {
    if (::atof(val.c_str()) <= 0) {
      fprintf(stderr, "Invalid SHERPA_NCNN_RULE1_MIN_TRAILING_SILENCE=%s\n",
              val.c_str());
      return -1;
    }
    config->endpoint_config.rule1.min_trailing_silence = ::atof(val.c_str());
  }

  SET_CONFIG_BY_ENV(val, "SHERPA_NCNN_RULE2_MIN_TRAILING_SILENCE", false);
  if (!val.empty()) {
    if (::atof(val.c_str()) <= 0) {
      fprintf(stderr, "Invalid SHERPA_NCNN_RULE2_MIN_TRAILING_SILENCE=%s\n",
              val.c_str());
      return -1;
    }
    config->endpoint_config.rule2.min_trailing_silence = ::atof(val.c_str());
  }

  SET_CONFIG_BY_ENV(val, "SHERPA_NCNN_RULE3_MIN_UTTERANCE_LENGTH", false);
  if (!val.empty()) {
    if (::atof(val.c_str()) <= 0) {
      fprintf(stderr, "Invalid SHERPA_NCNN_RULE3_MIN_UTTERANCE_LENGTH=%s\n",
              val.c_str());
      return -1;
    }
    config->endpoint_config.rule3.min_utterance_length = ::atof(val.c_str());
  }

  return parsed_required_envs;
}

static void SetDefaultConfigurations(sherpa_ncnn::RecognizerConfig *config) {
  int32_t num_threads = 4;
  config->model_config.encoder_opt.num_threads = num_threads;
  config->model_config.decoder_opt.num_threads = num_threads;
  config->model_config.joiner_opt.num_threads = num_threads;

  config->enable_endpoint = true;
  config->endpoint_config.rule1.min_trailing_silence = 2.4;
  config->endpoint_config.rule2.min_trailing_silence = 1.2;
  config->endpoint_config.rule3.min_utterance_length = 300;

  const float expected_sampling_rate = 16000;
  config->feat_config.sampling_rate = expected_sampling_rate;
  config->feat_config.feature_dim = 80;
}

static int32_t OverwriteConfigByCLI(int32_t argc, char **argv,
                                    sherpa_ncnn::RecognizerConfig *config,
                                    std::string *input_url) {
  if (argc > 1) config->model_config.tokens = argv[1];
  if (argc > 2) config->model_config.encoder_param = argv[2];
  if (argc > 3) config->model_config.encoder_bin = argv[3];
  if (argc > 4) config->model_config.decoder_param = argv[4];
  if (argc > 5) config->model_config.decoder_bin = argv[5];
  if (argc > 6) config->model_config.joiner_param = argv[6];
  if (argc > 7) config->model_config.joiner_bin = argv[7];
  if (argc > 8) *input_url = argv[8];
  if (argc >= 10 && atoi(argv[9]) > 0) {
    int32_t num_threads = atoi(argv[9]);
    config->model_config.encoder_opt.num_threads = num_threads;
    config->model_config.decoder_opt.num_threads = num_threads;
    config->model_config.joiner_opt.num_threads = num_threads;
  }

  if (argc == 11) {
    std::string val = argv[10];
    if (val != "greedy_search" && val != "modified_beam_search") {
      fprintf(stderr, "Invalid SHERPA_NCNN_METHOD=%s\n", val.c_str());
      return -1;
    }
    config->decoder_config.method = val;
  }

  return 0;
}

// A simple display, without window support, doesn't rewrite current line.
// It only output the new text, which only works in greedy_search mode.
// It doesn't support modified_beam_search mode, which might change the
// generated text.
class SimpleDisplay : public sherpa_ncnn::Display {
 public:
  SimpleDisplay(std::string label) {
    label_ = label.empty() ? "" : label + ":";
  }
  void Print(int32_t segment_id, const std::string &s) {
    if (last_segment_ != segment_id) {
      last_segment_ = segment_id;
      last_text_ = "";
      if (segment_id) {
        fprintf(stderr, "\n");
      }
      fprintf(stderr, "%s%d:", label_.c_str(), segment_id);
      if (!s.empty() && s.at(0) != ' ') {
        fprintf(stderr, " ");
      }
    }

    if (s.length() > last_text_.length()) {
      std::string tmp(s.begin() + last_text_.length(), s.end());
      fprintf(stderr, "%s", tmp.c_str());
    } else {
      fprintf(stderr, "%s", s.c_str());
    }
    last_text_ = s;
  }

 private:
  std::string label_;
  std::string last_text_;
  int32_t last_segment_ = -1;
};

std::unique_ptr<sherpa_ncnn::Display> CreateDisplay() {
  std::string val;
  SET_STRING_BY_ENV(val, "SHERPA_NCNN_SIMPLE_DISLAY");

  std::transform(val.begin(), val.end(), val.begin(),
                 [](auto c) { return std::tolower(c); });

  if (val == "on" || val == "true") {
    std::string label;
    SET_STRING_BY_ENV(label, "SHERPA_NCNN_DISPLAY_LABEL");
    return std::make_unique<SimpleDisplay>(label);
  } else {
    return std::make_unique<sherpa_ncnn::Display>();
  }
}

int32_t main(int32_t argc, char **argv) {
  // Set the default values for config.
  sherpa_ncnn::RecognizerConfig config;
  SetDefaultConfigurations(&config);

  // Load and overwrite config from environment variables.
  std::string input_url;
  int32_t parsed_required_envs = ParseConfigFromENV(&config, &input_url);
  if (parsed_required_envs < 0) {
    exit(-1);
  }

  // Error if not set by neither environment variables nor CLI.
  if (parsed_required_envs < 8 && (argc < 9 || argc > 11)) {
    const char *usage = R"usage(
Usage:
  ./bin/sherpa-ncnn-ffmpeg \
    /path/to/tokens.txt \
    /path/to/encoder.ncnn.param \
    /path/to/encoder.ncnn.bin \
    /path/to/decoder.ncnn.param \
    /path/to/decoder.ncnn.bin \
    /path/to/joiner.ncnn.param \
    /path/to/joiner.ncnn.bin \
    ffmpeg-input-url \
    [num_threads] [decode_method, can be greedy_search/modified_beam_search]

Or configure by environment variables:
  SHERPA_NCNN_TOKENS=/path/to/tokens.txt \
  SHERPA_NCNN_ENCODER_PARAM=/path/to/encoder_jit_trace-pnnx.ncnn.param  \
  SHERPA_NCNN_ENCODER_BIN=/path/to/encoder_jit_trace-pnnx.ncnn.bin \
  SHERPA_NCNN_DECODER_PARAM=/path/to/decoder_jit_trace-pnnx.ncnn.param \
  SHERPA_NCNN_DECODER_BIN=/path/to/decoder_jit_trace-pnnx.ncnn.bin \
  SHERPA_NCNN_JOINER_PARAM=/path/to/joiner_jit_trace-pnnx.ncnn.param  \
  SHERPA_NCNN_JOINER_BIN=/path/to/joiner_jit_trace-pnnx.ncnn.bin \
  SHERPA_NCNN_INPUT_URL=ffmpeg-input-url \
  SHERPA_NCNN_NUM_THREADS=4 \
  SHERPA_NCNN_METHOD=greedy_search|modified_beam_search \
  SHERPA_NCNN_ENABLE_ENDPOINT=on|off \
  SHERPA_NCNN_RULE1_MIN_TRAILING_SILENCE=2.4 \
  SHERPA_NCNN_RULE2_MIN_TRAILING_SILENCE=1.2 \
  SHERPA_NCNN_RULE3_MIN_UTTERANCE_LENGTH=300 \
  SHERPA_NCNN_SIMPLE_DISLAY=on|off \
  SHERPA_NCNN_DISPLAY_LABEL=Data \
  SHERPA_NCNN_ASD_ENDPOINTS=3 \
  SHERPA_NCNN_ASD_SAMPLES=10 \
  ./bin/sherpa-ncnn-ffmpeg

Please refer to
https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html
for a list of pre-trained models to download.
)usage";
    fprintf(stderr, "%s\n", usage);
    fprintf(stderr, "argc, %d\n", argc);

    return -1;
  }
  signal(SIGINT, Handler);
  signal(SIGUSR1, Handler);

  // Overwrite the config by CLI.
  if (OverwriteConfigByCLI(argc, argv, &config, &input_url)) {
    exit(-1);
  }

  fprintf(stdout, "Event:K2: Config is %s\n", config.ToString().c_str());
  fflush(stdout);

  sherpa_ncnn::Recognizer recognizer(config);
  auto s = recognizer.CreateStream();
  fprintf(stdout, "Event:K2: Create recognizer ok\n");
  fflush(stdout);

  // Initialize FFmpeg framework.
  AVPacket *packet = av_packet_alloc();
  AVFrame *frame = av_frame_alloc();
  AVFrame *filt_frame = av_frame_alloc();
  if (!packet || !frame || !filt_frame) {
    fprintf(stderr, "Could not allocate frame or packet\n");
    exit(1);
  }

  int32_t ret;
  fprintf(stdout, "Event:FFmpeg: Open input %s\n", input_url.c_str());
  fflush(stdout);
  if ((ret = FFmpegOpenInputFile(input_url.c_str())) < 0) {
    fprintf(stderr, "Open input file %s failed, r0=%d\n", input_url.c_str(),
            ret);
    exit(1);
  }
  fprintf(stdout, "Event:FFmpeg: Open input ok, %s\n", input_url.c_str());
  fflush(stdout);

  if ((ret = FFmpegInitFilters(filter_descr)) < 0) {
    fprintf(stderr, "Init filters %s failed, r0=%d\n", filter_descr, ret);
    exit(1);
  }

  int32_t asd_endpoints = 0, asd_samples = 0;
  SET_INTEGER_BY_ENV(asd_endpoints, "SHERPA_NCNN_ASD_ENDPOINTS");
  SET_INTEGER_BY_ENV(asd_samples, "SHERPA_NCNN_ASD_SAMPLES");

  std::string last_text;
  int32_t segment_index = 0, zero_samples = 0, asd_segment = 0;
  std::unique_ptr<sherpa_ncnn::Display> display = CreateDisplay();
  while (1) {
    if ((ret = av_read_frame(fmt_ctx, packet)) < 0) {
      break;
    }

    // Reset the ASD segment when stream unpublish.
    if (signal_unpublish_sigusr1) {
      signal_unpublish_sigusr1 = 0;
      if (asd_segment != segment_index) {
        asd_segment = segment_index;
      }
    }

    // ASD(Active speaker detection), note that 16000 samples is 1s.
    if (asd_samples && zero_samples > asd_samples * 16000) {
      // When unpublish, there might be some left samples in buffer.
      if (asd_endpoints && segment_index - asd_segment < asd_endpoints) {
        fprintf(stdout,
                "\nEvent:FFmpeg: All silence samples, incorrect microphone?\n");
        fflush(stdout);
      }
      zero_samples = 0;
    }

    if (packet->stream_index == audio_stream_index) {
      ret = avcodec_send_packet(dec_ctx, packet);
      if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR,
               "Error while sending a packet to the decoder\n");
        break;
      }

      while (ret >= 0) {
        ret = avcodec_receive_frame(dec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
          break;
        } else if (ret < 0) {
          av_log(NULL, AV_LOG_ERROR,
                 "Error while receiving a frame from the decoder\n");
          exit(1);
        }

        if (ret >= 0) {
          /* push the audio data from decoded frame into the filtergraph */
          if (av_buffersrc_add_frame_flags(buffersrc_ctx, frame,
                                           AV_BUFFERSRC_FLAG_KEEP_REF) < 0) {
            av_log(NULL, AV_LOG_ERROR,
                   "Error while feeding the audio filtergraph\n");
            break;
          }

          /* pull filtered audio from the filtergraph */
          while (1) {
            ret = av_buffersink_get_frame(buffersink_ctx, filt_frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
              break;
            }
            if (ret < 0) {
              fprintf(stderr, "Error get frame, ret=%d\n", ret);
              exit(1);
            }
            FFmpegDecodeFrame(filt_frame, recognizer, s.get(), display.get(),
                              &last_text, &segment_index, &zero_samples);
            av_frame_unref(filt_frame);
          }
          av_frame_unref(frame);
        }
      }
    }
    av_packet_unref(packet);
  }

  // Add some tail padding
  float tail_paddings[4800] = {0};  // 0.3 seconds at 16 kHz sample rate
  s->AcceptWaveform(16000, tail_paddings, 4800);

  s->InputFinished();

  while (recognizer.IsReady(s.get())) {
    recognizer.DecodeStream(s.get());
  }

  auto text = recognizer.GetResult(s.get()).text;
  if (!text.empty() && last_text != text) {
    last_text = text;
    std::transform(text.begin(), text.end(), text.begin(),
                   [](auto c) { return std::tolower(c); });
    display->Print(segment_index, text);
  }

  avfilter_graph_free(&filter_graph);
  avcodec_free_context(&dec_ctx);
  avformat_close_input(&fmt_ctx);
  av_packet_free(&packet);
  av_frame_free(&frame);
  av_frame_free(&filt_frame);

  if (ret < 0 && ret != AVERROR_EOF) {
    fprintf(stderr, "Error occurred: %s\n", FFmpegAvError2String(ret));
    exit(1);
  }

  return 0;
}
