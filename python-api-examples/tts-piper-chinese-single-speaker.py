#!/usr/bin/env python3

"""
You can download model files used in this script from
https://github.com/k2-fsa/sherpa-ncnn/releases/tag/tts-models

The following is an example:

wget https://github.com/k2-fsa/sherpa-ncnn/releases/download/tts-models/ncnn-vits-piper-zh_CN-huayan-medium-fp16.tar.bz2
tar xvf ncnn-vits-piper-zh_CN-huayan-medium-fp16.tar.bz2
rm ncnn-vits-piper-zh_CN-huayan-medium-fp16.tar.bz2
"""
import time

import sherpa_ncnn
import soundfile as sf


def create_tts():
    model_dir = "ncnn-vits-piper-zh_CN-huayan-medium-fp16"

    config = sherpa_ncnn.OfflineTtsConfig(
        model=sherpa_ncnn.OfflineTtsModelConfig(
            vits=sherpa_ncnn.OfflineTtsVitsModelConfig(model_dir=model_dir),
            num_threads=2,
            debug=False,
        ),
    )

    print(config)
    print()

    if not config.validate():
        raise ValueError("Please check your config")

    tts = sherpa_ncnn.OfflineTts(config=config)

    return tts


def main():
    tts = create_tts()

    print("number of speakers", tts.num_speakers)

    text = "他于长沙出生, 在长白山长大。几年前他成为了某银行的领导，主管行政, 能力很强，绝不勉强, 样样在行。 当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔."

    # If you use a multi-speaker model, you can set sid (speaker ID)
    args = sherpa_ncnn.TtsArgs(text=text, sid=0, speed=1.0)

    print(args)
    print()

    start = time.time()
    audio = tts.generate(args)
    end = time.time()

    if len(audio.samples) == 0:
        print("Error in generating audios. Please read previous error messages.")
        return

    # the sample rate of a model is fixed.
    # You can use librosa or some other python packages to resample the generated audio
    print("sample rate", audio.sample_rate)
    assert audio.sample_rate == tts.sample_rate

    elapsed_seconds = end - start
    audio_duration = len(audio.samples) / audio.sample_rate
    real_time_factor = elapsed_seconds / audio_duration

    filename = "test-zh.wav"
    sf.write(
        filename,
        audio.samples,
        samplerate=audio.sample_rate,
        subtype="PCM_16",
    )
    print(f"Saved to {filename}")
    print(f"The text is '{args.text}'")
    print(f"Elapsed seconds: {elapsed_seconds:.3f}")
    print(f"Audio duration in seconds: {audio_duration:.3f}")
    print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}")


if __name__ == "__main__":
    main()
