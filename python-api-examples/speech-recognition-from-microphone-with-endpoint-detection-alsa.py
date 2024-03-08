#!/usr/bin/env python3

# Real-time speech recognition from a microphone with sherpa-ncnn Python API
# with endpoint detection.
#
# Note: This script uses ALSA and works only on Linux systems, especially
# for embedding Linux systems and for running Linux on Windows using WSL.
#
# Please refer to
# https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html
# to download pre-trained models

import argparse
import sys

import sherpa_ncnn


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--device-name",
        type=str,
        required=True,
        help="""
The device name specifies which microphone to use in case there are several
on your system. You can use

  arecord -l

to find all available microphones on your computer. For instance, if it outputs

**** List of CAPTURE Hardware Devices ****
card 3: UACDemoV10 [UACDemoV1.0], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0

and if you want to select card 3 and the device 0 on that card, please use:

  plughw:3,0

as the device_name.
        """,
    )

    return parser.parse_args()


def create_recognizer():
    # Please replace the model files if needed.
    # See https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html
    # for download links.
    recognizer = sherpa_ncnn.Recognizer(
        tokens="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/tokens.txt",
        encoder_param="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.param",
        encoder_bin="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.bin",
        decoder_param="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.param",
        decoder_bin="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.bin",
        joiner_param="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.param",
        joiner_bin="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.bin",
        num_threads=4,
        decoding_method="modified_beam_search",
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=300,
        hotwords_file="",
        hotwords_score=1.5,
    )
    return recognizer


def main():
    args = get_args()
    device_name = args.device_name
    print(f"device_name: {device_name}")
    alsa = sherpa_ncnn.Alsa(device_name)

    recognizer = create_recognizer()
    print("Started! Please speak")
    sample_rate = recognizer.sample_rate
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms
    last_result = ""
    segment_id = 0

    while True:
        samples = alsa.read(samples_per_read)  # a blocking read
        recognizer.accept_waveform(sample_rate, samples)

        is_endpoint = recognizer.is_endpoint

        result = recognizer.text
        if result and (last_result != result):
            last_result = result
            print("\r{}:{}".format(segment_id, result), end="", flush=True)

        if is_endpoint:
            if result:
                print("\r{}:{}".format(segment_id, result), flush=True)
                segment_id += 1
            recognizer.reset()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
