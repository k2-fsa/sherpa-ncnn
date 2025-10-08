#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation

import argparse
from typing import List, Tuple

import pnnx
import sentencepiece as spm
import torch

from torch_model import SenseVoiceSmall


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--fp16",
        type=int,
        default=1,
        help="1 to use fp16. 0 to use float32",
    )

    return parser.parse_args()


def load_cmvn(filename) -> Tuple[List[float], List[float]]:
    neg_mean = None
    inv_stddev = None

    with open(filename) as f:
        for line in f:
            if not line.startswith("<LearnRateCoef>"):
                continue
            t = line.split()[3:-1]

            if neg_mean is None:
                neg_mean = list(map(lambda x: float(x), t))
            else:
                inv_stddev = list(map(lambda x: float(x), t))

    return neg_mean, inv_stddev


def generate_tokens(sp):
    with open("tokens.txt", "w", encoding="utf-8") as f:
        for i in range(sp.vocab_size()):
            f.write(f"{sp.id_to_piece(i)} {i}\n")
    print("saved to tokens.txt")


@torch.no_grad()
def main():
    args = get_args()
    fp16 = bool(args.fp16)

    sp = spm.SentencePieceProcessor()
    sp.load("./chn_jpn_yue_eng_ko_spectok.bpe.model")
    generate_tokens(sp)

    state_dict = torch.load("./model.pt", map_location="cpu")

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    neg_mean, inv_stddev = load_cmvn("./am.mvn")

    neg_mean = torch.tensor(neg_mean, dtype=torch.float32)
    inv_stddev = torch.tensor(inv_stddev, dtype=torch.float32)

    model = SenseVoiceSmall(neg_mean=neg_mean, inv_stddev=inv_stddev)
    model.load_state_dict(state_dict)
    model.eval()
    del state_dict
    x1 = torch.rand(1, 100, 560, dtype=torch.float32)
    x2 = torch.rand(1, 200, 560, dtype=torch.float32)

    language = 3
    text_norm = 15
    prompt = torch.tensor([language, 1, 2, text_norm], dtype=torch.int32)

    pos_emb1 = torch.rand(1, x1.shape[1] + 4, 560, dtype=torch.float32)
    pos_emb2 = torch.rand(1, x2.shape[1] + 4, 560, dtype=torch.float32)

    pnnx.export(
        model,
        "model.torchscript",
        (x1, prompt, pos_emb1),
        (x2, prompt, pos_emb2),
        fp16=fp16,
    )


if __name__ == "__main__":
    torch.manual_seed(20250912)
    main()
