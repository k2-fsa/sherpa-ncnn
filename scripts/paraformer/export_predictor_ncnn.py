#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation

import pnnx
import torch

from export_encoder_ncnn import load_model, get_args
from torch_model import CifPredictorV2

if __name__ == "__main__":

    def modified_predictor_forward(self: CifPredictorV2, hidden: torch.Tensor):
        h = hidden
        context = h.transpose(1, 2)
        queries = self.pad(context)
        output = torch.relu(self.cif_conv1d(queries))
        output = output.transpose(1, 2)

        output = self.cif_output(output)
        alphas = torch.sigmoid(output)
        alphas = torch.nn.functional.relu(
            alphas * self.smooth_factor - self.noise_threshold
        )

        alphas = alphas.squeeze(-1)

        return alphas

    CifPredictorV2.forward = modified_predictor_forward


@torch.no_grad()
def main():
    print("loading model")
    model = load_model()

    args = get_args()
    fp16 = bool(args.fp16)

    x1 = torch.rand(1, 100, 512, dtype=torch.float32)
    x2 = torch.rand(1, 200, 512, dtype=torch.float32)

    print("exporting")
    pnnx.export(
        model.predictor,
        "predictor.torchscript",
        (x1,),
        (x2,),
        fp16=fp16,
    )


if __name__ == "__main__":
    torch.manual_seed(20251008)
    main()
