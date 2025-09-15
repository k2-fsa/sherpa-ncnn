#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation

"""
This file shows how to use the sherpa-ncnn Python API to decode a file
with sense-voice ASR models.

Plese download model files from
https://github.com/k2-fsa/sherpa-ncnn/releases/tag/asr-models

Example:

wget https://github.com/k2-fsa/sherpa-ncnn/releases/download/asr-models/sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xvf sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
"""

import time

import sherpa_ncnn
import soundfile as sf


def create_offline_recognizer():
    config = sherpa_ncnn.OfflineRecognizerConfig(
        model_config=sherpa_ncnn.OfflineModelConfig(
            sense_voice=sherpa_ncnn.OfflineSenseVoiceModelConfig(
                model_dir="./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17",
                use_itn=True,
            ),
            tokens="./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt",
            num_threads=2,
            debug=True,
        )
    )
    if not config.validate():
        raise ValueError("Please check you config")

    print(config)

    return sherpa_ncnn.OfflineRecognizer(config)


def main():
    recognizer = create_offline_recognizer()

    wave_filename = (
        "./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/en.wav"
    )

    audio, sample_rate = sf.read(wave_filename, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # only use the first channel

    print("Started")
    start_time = time.time()

    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio)
    recognizer.decode_stream(stream)

    end_time = time.time()
    print("Done!")

    print(wave_filename)
    print(stream.result)
    print("----")
    print("text:", stream.result.text)
    print("====")

    duration = len(audio) / sample_rate
    elapsed_seconds = end_time - start_time
    rtf = elapsed_seconds / duration
    print(f"num_threads: {recognizer.config.model_config.num_threads}")
    print(f"decoding_method: {recognizer.config.decoding_method}")
    print(f"Wave duration: {duration:.3f} s")
    print(f"Elapsed time: {elapsed_seconds:.3f} s")
    print(f"Real time factor (RTF): {elapsed_seconds:.3f}/{duration:.3f} = {rtf:.3f}")


if __name__ == "__main__":
    main()
