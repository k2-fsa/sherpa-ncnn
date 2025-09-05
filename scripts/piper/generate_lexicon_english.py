#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

try:
    from piper_phonemize import phonemize_espeak
except Exception as ex:
    raise RuntimeError(
        f"{ex}\nPlease run\n"
        "pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html"
    )
import re
import sys

if len(sys.argv) != 2:
    raise ValueError("python3 ./generate_lexicon_english.py <en-us|en-gb-x-rp>")


def read_lexicon():
    in_file = "./CMU.in.IPA.txt"
    words = set()
    pattern = re.compile("^[a-zA-Z'-\.]+$")
    with open(in_file) as f:
        for line in f:
            try:
                line = line.strip()
                word, _ = line.split(",")
                word = word.strip()
                if not pattern.match(word):
                    #  print(line, "word is", word)
                    continue
            except:
                #  print(line)
                continue

            assert word not in words, word
            words.add(word)
    return list(words)


def main():
    words = read_lexicon()
    words.sort()

    num_words = len(words)
    word2ipa = dict()
    for w in words:
        tokens = " ".join(phonemize_espeak(w, sys.argv[1])[0])
        word2ipa[w] = tokens

    with open("lexicon.txt", "w", encoding="utf-8") as f:
        for w, p in word2ipa.items():
            f.write(f"{w} {p}\n")


if __name__ == "__main__":
    main()
