// Copyright (c)  2023-2024  Xiaomi Corporation (authors: Fangjun Kuang)
//
'use strict'

const wasmModule = require('./sherpa-ncnn-wasm-main.js')();
const sherpa_ncnn = require('./sherpa-ncnn.js');

function createRecognizer(config) {
  sherpa_ncnn.createRecognizer(wasmModule, config);
}

module.exports = {
  createRecognizer,
};
