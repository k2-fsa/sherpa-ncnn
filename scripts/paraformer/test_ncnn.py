#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation

import kaldi_native_fbank as knf
import librosa
import ncnn
import numpy as np

from test_encoder_ncnn import SinusoidalPositionEncoder


def load_cmvn():
    neg_mean = None
    inv_std = None

    with open("am.mvn") as f:
        for line in f:
            if not line.startswith("<LearnRateCoef>"):
                continue
            t = line.split()[3:-1]
            t = list(map(lambda x: float(x), t))

            if neg_mean is None:
                neg_mean = np.array(t, dtype=np.float32)
            else:
                inv_std = np.array(t, dtype=np.float32)

    return neg_mean, inv_std


def compute_feat(filename):
    sample_rate = 16000
    samples, _ = librosa.load(filename, sr=sample_rate)
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
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
    print("features sum", features.sum(), features.shape)

    window_size = 7  # lfr_m
    window_shift = 6  # lfr_n

    T = (features.shape[0] - window_size) // window_shift + 1
    features = np.lib.stride_tricks.as_strided(
        features,
        shape=(T, features.shape[1] * window_size),
        strides=((window_shift * features.shape[1]) * 4, 4),
    )
    neg_mean, inv_std = load_cmvn()
    features = (features + neg_mean) * inv_std
    return features


def load_tokens():
    ans = dict()
    i = 0
    with open("tokens.txt", encoding="utf-8") as f:
        for line in f:
            ans[i] = line.strip().split()[0]
            i += 1
    return ans


def run_encoder(features, pos_emb):
    with ncnn.Net() as net:
        print("load encoder model")
        net.opt.num_threads = 1
        net.load_param("./encoder.ncnn.param")
        net.load_model("./encoder.ncnn.bin")

        print("run encoder model")
        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(features).clone())
            ex.input("in1", ncnn.Mat(pos_emb).clone())

            _, out0 = ex.extract("out0")
            out0 = np.array(out0)

            return np.copy(out0)


def run_predictor(hidden):
    with ncnn.Net() as net:
        print("load predictor model")
        net.opt.num_threads = 1
        net.load_param("./predictor.ncnn.param")
        net.load_model("./predictor.ncnn.bin")

        print("run predictor model")
        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(hidden).clone())

            _, out0 = ex.extract("out0")
            out0 = np.array(out0)

            return np.copy(out0)


def get_acoustic_embedding(alpha: np.array, hidden: np.array):
    """
    Args:
      alpha: (T,)
      hidden: (T, C)
    Returns:
      acoustic_embeds: (num_tokens, C)
    """
    alpha = alpha.tolist()
    acc = 0
    num_tokens = 0

    embeddings = []
    cur_embedding = np.zeros((hidden.shape[1],), dtype=np.float32)
    fire_idx = []

    for i, w in enumerate(alpha):
        acc += w
        if acc >= 1:
            fire_idx.append(i)
            overflow = acc - 1
            remain = w - overflow
            cur_embedding += remain * hidden[i]
            embeddings.append(cur_embedding)

            cur_embedding = overflow * hidden[i]
            acc = overflow
        else:
            cur_embedding += w * hidden[i]

    if len(embeddings) == 0:
        raise ValueError("No speech in the audio file")

    embeddings = np.array(embeddings)
    return embeddings, fire_idx


def run_decoder(encoder_out, acoustic_embedding):
    with ncnn.Net() as net:
        print("load decoder model")
        net.opt.num_threads = 1
        net.load_param("./decoder.ncnn.param")
        net.load_model("./decoder.ncnn.bin")

        print("run decoder model")
        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(encoder_out).clone())
            ex.input("in1", ncnn.Mat(acoustic_embedding).clone())

            _, out0 = ex.extract("out0")
            out0 = np.array(out0)

            return np.copy(out0)


def main():
    features = compute_feat("./16.wav")
    pos_emb = (
        SinusoidalPositionEncoder()(1, features.shape[0], features.shape[1])
        .squeeze(0)
        .numpy()
    )
    print("features.shape", features.shape, pos_emb.shape)
    encoder_out = run_encoder(features, pos_emb)
    print("encoder_out.shape", encoder_out.shape)
    alpha = run_predictor(encoder_out)
    print("alpha.shape", alpha.shape)

    acoustic_embedding, fire_idx = get_acoustic_embedding(alpha, encoder_out)
    print("acoustic_embedding.shape", acoustic_embedding.shape)
    print("fire_idx", fire_idx)
    print([i * 0.06 for i in fire_idx], len(fire_idx))

    decoder_out = run_decoder(encoder_out, acoustic_embedding)
    yseq = decoder_out.argmax(axis=-1).tolist()
    print(yseq, "-->", len(yseq))

    tokens = load_tokens()
    words = [tokens[i] for i in yseq if i not in (1, 2)]
    print(words)
    text = "".join(words)
    print(text)


if __name__ == "__main__":
    main()
