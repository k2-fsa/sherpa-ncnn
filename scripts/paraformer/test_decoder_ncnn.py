#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation

import ncnn
import numpy as np
import torch

from export_encoder_ncnn import load_model


@torch.no_grad()
def run_torch(encoder_out, acoustic_embedding):
    model = load_model()
    return model.decoder(encoder_out, acoustic_embedding)


def run_ncnn(encoder_out, acoustic_embedding):
    print("run ncnn")
    with ncnn.Net() as net:
        print("load ncnn model")
        net.opt.num_threads = 1
        net.load_param("./decoder.ncnn.param")
        net.load_model("./decoder.ncnn.bin")

        print("run ncnn model")
        with net.create_extractor() as ex:
            encoder_out = encoder_out.squeeze(0).numpy()
            acoustic_embedding = acoustic_embedding.squeeze(0).numpy()

            ex.input("in0", ncnn.Mat(encoder_out).clone())
            ex.input("in1", ncnn.Mat(acoustic_embedding).clone())

            _, out0 = ex.extract("out0")
            out0 = np.array(out0)

            return torch.from_numpy(out0).clone()


def main():
    for i in range(5):
        encoder_out = torch.rand(1, 50, 512)
        acoustic_embedding = torch.rand(1, 10, 512)

        y0 = run_torch(encoder_out, acoustic_embedding)
        y1 = run_ncnn(encoder_out, acoustic_embedding)
        print(y0.shape, y0.sum(), y0.mean())
        print(y1.shape, y1.sum(), y1.mean())
        print((y0 - y1).abs().max())


if __name__ == "__main__":
    torch.manual_seed(20251008)
    main()
