#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation

import ncnn
import numpy as np
import torch

from export_encoder_ncnn import load_model


class SinusoidalPositionEncoder(torch.nn.Module):
    def encode(
        self,
        positions: torch.Tensor = None,
        depth: int = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
          positions: (batch_size, )
        """
        batch_size = positions.size(0)
        positions = positions.type(dtype)
        device = positions.device
        log_timescale_increment = torch.log(
            torch.tensor([10000], dtype=dtype, device=device)
        ) / (depth / 2 - 1)
        inv_timescales = torch.exp(
            torch.arange(depth / 2, device=device).type(dtype)
            * (-log_timescale_increment)
        )
        inv_timescales = torch.reshape(inv_timescales, [batch_size, -1])
        scaled_time = torch.reshape(positions, [1, -1, 1]) * torch.reshape(
            inv_timescales, [1, 1, -1]
        )
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        return encoding.type(dtype)

    def forward(self, batch_size, timesteps, input_dim):
        positions = torch.arange(1, timesteps + 1)[None, :]
        position_encoding = self.encode(positions, input_dim, torch.float32)

        return position_encoding


@torch.no_grad()
def run_torch(x):
    model = load_model()
    return model.encoder(x)


def run_ncnn(x):
    print("run ncnn", x.shape)
    pos_emb = SinusoidalPositionEncoder()(*x.shape)

    with ncnn.Net() as net:
        print("load ncnn model")
        net.opt.num_threads = 1
        net.load_param("./encoder.ncnn.param")
        net.load_model("./encoder.ncnn.bin")

        print("run ncnn model")
        with net.create_extractor() as ex:
            x = x.squeeze(0).numpy()
            pos = pos_emb.squeeze(0).numpy()
            ex.input("in0", ncnn.Mat(x).clone())
            ex.input("in1", ncnn.Mat(pos).clone())

            _, out0 = ex.extract("out0")
            out0 = np.array(out0)

            return torch.from_numpy(out0).clone()


def main():
    for i in range(5):
        x = torch.rand(1, 50, 560)
        y0 = run_torch(x)
        y1 = run_ncnn(x)
        print(y0.shape, y0.sum(), y0.mean())
        print(y1.shape, y1.sum(), y1.mean())
        print((y0 - y1).abs().max())


if __name__ == "__main__":
    torch.manual_seed(20251008)
    main()
