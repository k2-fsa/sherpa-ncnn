#!/usr/bin/env python3
import argparse
from typing import Tuple

import kaldi_native_fbank as knf
import ncnn
import numpy as np
import soundfile as sf
import torch
from torch import nn


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-ncnn-param",
        type=str,
        required=True,
        help="path to model.ncnn.param",
    )

    parser.add_argument(
        "--model-ncnn-bin",
        type=str,
        required=True,
        help="path to model.ncnn.bin",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="path to tokens.txt",
    )

    parser.add_argument(
        "--wave",
        type=str,
        required=True,
        help="The input wave to be recognized",
    )

    return parser.parse_args()


class SinusoidalPositionEncoder(nn.Module):
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


def load_tokens(filename):
    ans = dict()
    i = 0
    with open(filename, encoding="utf-8") as f:
        for line in f:
            ans[i] = line.strip().split()[0]
            i += 1
    return ans


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def compute_feat(
    samples,
    sample_rate,
    window_size: int = 7,  # lfr_m
    window_shift: int = 6,  # lfr_n
):
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = 80

    online_fbank = knf.OnlineFbank(opts)
    online_fbank.accept_waveform(sample_rate, (samples * 32768).tolist())
    online_fbank.input_finished()

    features = np.stack(
        [online_fbank.get_frame(i) for i in range(online_fbank.num_frames_ready)]
    )
    assert features.data.contiguous is True
    assert features.dtype == np.float32, features.dtype

    T = (features.shape[0] - window_size) // window_shift + 1
    features = np.lib.stride_tricks.as_strided(
        features,
        shape=(T, features.shape[1] * window_size),
        strides=((window_shift * features.shape[1]) * 4, 4),
    )

    return features.copy()


def main():
    args = get_args()
    samples, sample_rate = load_audio(args.wave)
    if sample_rate != 16000:
        import librosa

        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    features = compute_feat(
        samples=samples,
        sample_rate=sample_rate,
        window_size=7,
        window_shift=6,
    )

    pos_emb = SinusoidalPositionEncoder()(1, features.shape[0] + 4, features.shape[-1])[
        0
    ]

    in1 = torch.tensor([3, 1, 2, 15], dtype=torch.int32)

    out = []

    with ncnn.Net() as net:
        net.opt.num_threads = 1
        net.load_param(args.model_ncnn_param)
        net.load_model(args.model_ncnn_bin)

        with net.create_extractor() as ex:
            x = features
            ex.input("in0", ncnn.Mat(x).clone())
            ex.input("in1", ncnn.Mat(in1.numpy()).clone())
            ex.input("in2", ncnn.Mat(pos_emb.numpy()).clone())

            _, out0 = ex.extract("out0")
            logits = np.array(out0)

    print(logits.shape)
    ids = logits.argmax(axis=-1).tolist()

    ans = []
    prev = -1
    blank = 0

    for i in ids:
        if i != blank and i != prev:
            ans.append(i)
        prev = i

    print(ids)
    print(ans)

    tokens = load_tokens("./tokens.txt")
    text = "".join([tokens[i] for i in ans])

    text = text.replace("▁", " ")
    print(text)


if __name__ == "__main__":
    main()
