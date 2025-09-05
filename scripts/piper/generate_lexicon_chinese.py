#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

from pypinyin import load_phrases_dict, phrases_dict, pinyin_dict

try:
    from piper_phonemize import phonemize_espeak
except Exception as ex:
    raise RuntimeError(
        f"{ex}\nPlease run\n"
        "pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html"
    )


def main():
    filename = "lexicon-zh_CN.txt"

    word_dict = pinyin_dict.pinyin_dict
    phrases = phrases_dict.phrases_dict

    i = 0
    with open(filename, "w", encoding="utf-8") as f:
        for key in word_dict:
            if not (0x4E00 <= key <= 0x9FFF):
                continue

            w = chr(key)
            tokens = " ".join(phonemize_espeak(w, "cmn")[0])

            f.write(f"{w} {tokens}\n")

        for key in phrases:
            tokens = " ".join(phonemize_espeak(key, "cmn")[0])

            f.write(f"{key} {tokens}\n")


if __name__ == "__main__":
    main()
