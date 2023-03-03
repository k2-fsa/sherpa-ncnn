#!/usr/bin/env python3

"""
This file demonstrates how to use sherpa-ncnn Python API to recognize
a single file.

Please refer to
https://k2-fsa.github.io/sherpa/ncnn/index.html
to install sherpa-ncnn and to download the pre-trained models
used in this file.
"""

import time
import wave

import numpy as np
import sherpa_ncnn


def main():
    # Please refer to https://k2-fsa.github.io/sherpa/ncnn/index.html
    # to download the model files
    recognizer = sherpa_ncnn.Recognizer(
        tokens="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/tokens.txt",
        encoder_param="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.param",
        encoder_bin="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.bin",
        decoder_param="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.param",
        decoder_bin="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.bin",
        joiner_param="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.param",
        joiner_bin="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.bin",
        num_threads=4,
    )

    filename = "./sherpa-ncnn-conv-emformer-transducer-2022-12-06/test_wavs/1.wav"
    with wave.open(filename) as f:
        # Note: If wave_file_sample_rate is different from
        # recognizer.sample_rate, we will do resampling inside sherpa-ncnn
        wave_file_sample_rate = f.getframerate()
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)

        samples_float32 = samples_float32 / 32768

    # simulate streaming
    chunk_size = int(0.1 * wave_file_sample_rate)  # 0.1 seconds
    start = 0
    while start < samples_float32.shape[0]:
        end = start + chunk_size
        end = min(end, samples_float32.shape[0])
        recognizer.accept_waveform(wave_file_sample_rate, samples_float32[start:end])
        start = end
        text = recognizer.text
        if text:
            print(text)

        # simulate streaming by sleeping
        time.sleep(0.1)

    tail_paddings = np.zeros(int(wave_file_sample_rate * 0.5), dtype=np.float32)
    recognizer.accept_waveform(wave_file_sample_rate, tail_paddings)
    recognizer.input_finished()
    text = recognizer.text
    if text:
        print(text)


if __name__ == "__main__":
    main()
