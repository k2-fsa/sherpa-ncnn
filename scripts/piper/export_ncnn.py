#!/usr/bin/env python3

'''
This file is modified from
https://github.com/nihui/ncnn-android-piper/blob/master/export_ncnn.py
'''

import torch
import os
import sys
sys.path.insert(0, './src')
import pnnx

if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} en.ckpt")
    sys.exit(0)

ckptpath = sys.argv[1]

from piper.train.vits.lightning import VitsModel
import piper
print('piper', piper.__file__)

model = VitsModel.load_from_checkpoint(ckptpath, map_location="cpu")

model_g = model.model_g

model_g.eval()

num_symbols = model_g.n_vocab
num_speakers = model_g.n_speakers

print(num_symbols)
print(num_speakers)

dummy_input_length = 50
dummy_input_length2 = 200
sequences = torch.randint(low=0, high=num_symbols, size=(1, dummy_input_length), dtype=torch.long)
sequences2 = torch.randint(low=0, high=num_symbols, size=(1, dummy_input_length2), dtype=torch.long)

if num_speakers > 1:
    sid = torch.LongTensor([0])

    # export emb_g
    def infer_forward_emb_g(sid):
        g = model_g.emb_g(sid)
        return g

    model_g.forward = infer_forward_emb_g
    pnnx.export(model_g, 'embedding.torchscript', (sid,))

    # export enc_p
    def infer_forward_enc_p(text):
        x, m_p, logs_p = model_g.enc_p(text)
        return x, m_p, logs_p

    model_g.forward = infer_forward_enc_p
    pnnx.export(model_g, 'encoder.torchscript', (sequences,), (sequences2,), moduleop='piper.train.vits.attentions.relative_embeddings_k_module,piper.train.vits.attentions.relative_embeddings_v_module')

    # export dp
    def infer_forward_dp(x, noise, g):
        x_mask = torch.ones(1, 1, x.size(2))
        logw = model_g.dp(x, noise, x_mask, g=g, reverse=True)
        return logw

    noise_scale_w = 0.8
    x = torch.rand(1, 192, dummy_input_length)
    noise = torch.rand(1, 2, dummy_input_length) * noise_scale_w
    x2 = torch.rand(1, 192, dummy_input_length2)
    noise2 = torch.rand(1, 2, dummy_input_length2) * noise_scale_w
    g = torch.rand(1, 512, 1)

    model_g.forward = infer_forward_dp
    pnnx.export(model_g, 'dp.torchscript', (x, noise, g), (x2, noise2, g), moduleop='piper.train.vits.modules.piecewise_rational_quadratic_transform_module')

    # export flow
    def infer_forward_flow(z_p, g):
        y_mask = torch.ones(1, 1, z_p.size(2))
        z = model_g.flow(z_p, y_mask, g=g, reverse=True)
        return z

    z_p = torch.rand(1, 192, 224)

    model_g.forward = infer_forward_flow
    pnnx.export(model_g, 'flow.torchscript', (z_p, g))

    # export dec
    def infer_forward_dec(z, g):
        o = model_g.dec(z, g=g)
        return o

    z = torch.rand(1, 192, 164)

    model_g.forward = infer_forward_dec
    pnnx.export(model_g, 'decoder.torchscript', (z, g))

else:
    # export enc_p
    def infer_forward_enc_p(text):
        x, m_p, logs_p = model_g.enc_p(text)
        return x, m_p, logs_p

    model_g.forward = infer_forward_enc_p

    pnnx.export(model_g, 'encoder.torchscript', (sequences,), (sequences2,), moduleop='piper.train.vits.attentions.relative_embeddings_k_module,piper.train.vits.attentions.relative_embeddings_v_module')

    # export dp
    def infer_forward_dp(x, noise):
        x_mask = torch.ones(1, 1, x.size(2))
        g = None
        logw = model_g.dp(x, noise, x_mask, g=g, reverse=True)
        return logw

    noise_scale_w = 0.8
    x = torch.rand(1, 192, dummy_input_length)
    noise = torch.rand(1, 2, dummy_input_length) * noise_scale_w
    x2 = torch.rand(1, 192, dummy_input_length2)
    noise2 = torch.rand(1, 2, dummy_input_length2) * noise_scale_w

    model_g.forward = infer_forward_dp
    pnnx.export(model_g, 'dp.torchscript', (x, noise), (x2, noise2), moduleop='piper.train.vits.modules.piecewise_rational_quadratic_transform_module')

    # export flow
    def infer_forward_flow(z_p):
        y_mask = torch.ones(1, 1, z_p.size(2))
        g = None
        z = model_g.flow(z_p, y_mask, g=g, reverse=True)
        return z

    z_p = torch.rand(1, 192, 224)

    model_g.forward = infer_forward_flow
    pnnx.export(model_g, 'flow.torchscript', (z_p, ))

    # export dec
    def infer_forward_dec(z):
        g = None
        o = model_g.dec(z, g=g)
        return o

    z = torch.rand(1, 192, 164)

    model_g.forward = infer_forward_dec
    pnnx.export(model_g, 'decoder.torchscript', (z, ))
