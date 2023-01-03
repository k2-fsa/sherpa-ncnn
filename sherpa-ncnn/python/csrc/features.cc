/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include "sherpa-ncnn/csrc/features.h"

#include "sherpa-ncnn/python/csrc/mat-util.h"
#include "sherpa-ncnn/python/csrc/sherpa-ncnn.h"

namespace sherpa_ncnn {

void PybindFeatures(py::module *m) {
  using PyClass = FeatureExtractor;

  py::class_<PyClass>(*m, "FeatureExtractor")
      .def(py::init([](int32_t feature_dim,
                       float sample_rate) -> std::unique_ptr<PyClass> {
             knf::FbankOptions fbank_opts;
             fbank_opts.frame_opts.dither = 0;
             fbank_opts.frame_opts.snip_edges = false;
             fbank_opts.frame_opts.samp_freq = sample_rate;
             fbank_opts.mel_opts.num_bins = feature_dim;

             return std::make_unique<PyClass>(fbank_opts);
           }),
           py::arg("feature_dim"), py::arg("sample_rate"))
      .def("accept_waveform",
           [](PyClass &self, float sample_rate, py::array_t<float> waveform) {
             self.AcceptWaveform(sample_rate, waveform.data(), waveform.size());
           })
      .def("input_finished", &PyClass::InputFinished)
      .def_property_readonly("num_frames_ready", &PyClass::NumFramesReady)
      .def("is_last_frame", &PyClass::IsLastFrame, py::arg("frame"))
      .def("get_frames",
           [](PyClass &self, int32_t frame_index, int32_t n) -> py::array {
             ncnn::Mat frames = self.GetFrames(frame_index, n);
             return MatToArray(frames);
           })
      .def("reset", &PyClass::Reset);
}

}  // namespace sherpa_ncnn
