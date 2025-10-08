#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation

import pnnx
import torch
import yaml

from export_encoder_ncnn import load_model


@torch.no_grad()
def main():
    print("loading model")
    model = load_model()

    fp16 = False

    encoder_out1 = torch.rand(1, 100, 512, dtype=torch.float32)
    acoustic_embeds1 = torch.rand(1, 10, 512, dtype=torch.float32)

    encoder_out2 = torch.rand(1, 50, 512, dtype=torch.float32)
    acoustic_embeds2 = torch.rand(1, 8, 512, dtype=torch.float32)

    print("exporting")
    pnnx.export(
        model.decoder,
        "decoder.torchscript",
        (encoder_out1, acoustic_embeds1),
        (encoder_out2, acoustic_embeds2),
        fp16=fp16,
    )


if __name__ == "__main__":
    torch.manual_seed(20251008)
    main()
