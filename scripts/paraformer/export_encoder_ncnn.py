#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation

from typing import List, Tuple

import pnnx
import torch
import yaml

from torch_model import Paraformer, SANMEncoder


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


if __name__ == "__main__":

    def modified_sanm_encoder_forward(
        self: SANMEncoder, xs_pad: torch.Tensor, pos: torch.Tensor
    ):
        xs_pad = (xs_pad + self.neg_mean) * self.inv_stddev

        xs_pad = xs_pad * self.output_size() ** 0.5

        xs_pad = xs_pad + pos

        xs_pad = self.encoders0(xs_pad)[0]
        xs_pad = self.encoders(xs_pad)[0]

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        return xs_pad

    SANMEncoder.forward = modified_sanm_encoder_forward


def load_model():
    with open("./config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("creating model")

    neg_mean, inv_stddev = load_cmvn("./am.mvn")

    neg_mean = torch.tensor(neg_mean, dtype=torch.float32)
    inv_stddev = torch.tensor(inv_stddev, dtype=torch.float32)

    m = Paraformer(
        neg_mean=neg_mean,
        inv_stddev=inv_stddev,
        input_size=560,
        vocab_size=8404,
        encoder_conf=config["encoder_conf"],
        decoder_conf=config["decoder_conf"],
        predictor_conf=config["predictor_conf"],
    )
    m.eval()

    print("loading state dict")
    state_dict = torch.load("./model_state_dict.pt", map_location="cpu")["state_dict"]
    m.load_state_dict(state_dict)
    del state_dict

    return m


@torch.no_grad()
def main():
    print("loading model")
    model = load_model()

    fp16 = True
    x1 = torch.rand(1, 100, 560, dtype=torch.float32)
    x2 = torch.rand(1, 200, 560, dtype=torch.float32)

    pos_emb1 = torch.rand(1, x1.shape[1], 560, dtype=torch.float32)
    pos_emb2 = torch.rand(1, x2.shape[1], 560, dtype=torch.float32)

    print(x1.shape, pos_emb1.shape)
    print(x2.shape, pos_emb2.shape)

    print("exporting")
    pnnx.export(
        model.encoder,
        "encoder.torchscript",
        (x1, pos_emb1),
        (x2, pos_emb2),
        fp16=fp16,
    )


if __name__ == "__main__":
    torch.manual_seed(20251008)
    main()
