#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation

import ncnn
import numpy as np
import torch

from export_encoder_ncnn import load_model


@torch.no_grad()
def run_torch(hidden):
    model = load_model()
    return model.predictor(hidden)[2][0, :-1]


def run_ncnn(hidden):
    print("run ncnn")
    with ncnn.Net() as net:
        print("load ncnn model")
        net.opt.num_threads = 1
        net.load_param("./predictor.ncnn.param")
        net.load_model("./predictor.ncnn.bin")

        print("run ncnn model")
        with net.create_extractor() as ex:
            hidden = hidden.squeeze(0).numpy()

            ex.input("in0", ncnn.Mat(hidden).clone())

            _, out0 = ex.extract("out0")
            out0 = np.array(out0)

            return torch.from_numpy(out0).clone()


def main():
    for i in range(5):
        hidden = torch.rand(1, 50, 512)
        y0 = run_torch(hidden)
        y1 = run_ncnn(hidden)
        print(y0.shape, y0.sum(), y0.mean())
        print(y1.shape, y1.sum(), y1.mean())
        print((y0 - y1).abs().max())


if __name__ == "__main__":
    torch.manual_seed(20251008)
    main()
