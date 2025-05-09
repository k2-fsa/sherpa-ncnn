// Copyright (c)  2024-2025  Xiaomi Corporation (authors: Fangjun Kuang)
//
// See also https://github.com/k2-fsa/sherpa-onnx/pull/1909
//
// It uses `mic` for better compatibility, do check its
// [npm](https://www.npmjs.com/package/mic) before running it.
const mic = require('mic');

const sherpa_ncnn = require('sherpa-ncnn');

function createRecognizer() {
  let modelConfig = {
    encoderParam:
        './sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/encoder_jit_trace-pnnx.ncnn.param',
    encoderBin:
        './sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/encoder_jit_trace-pnnx.ncnn.bin',
    decoderParam:
        './sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/decoder_jit_trace-pnnx.ncnn.param',
    decoderBin:
        './sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/decoder_jit_trace-pnnx.ncnn.bin',
    joinerParam:
        './sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/joiner_jit_trace-pnnx.ncnn.param',
    joinerBin:
        './sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/joiner_jit_trace-pnnx.ncnn.bin',
    tokens:
        './sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/tokens.txt',
    useVulkanCompute: 0,
    numThreads: 1,
  };

  let decoderConfig = {
    decodingMethod: 'greedy_search',
    numActivePaths: 4,
  };

  let featConfig = {
    samplingRate: 16000,
    featureDim: 80,
  };

  let config = {
    featConfig: featConfig,
    modelConfig: modelConfig,
    decoderConfig: decoderConfig,
    enableEndpoint: 1,
    rule1MinTrailingSilence: 1.2,
    rule2MinTrailingSilence: 2.4,
    rule3MinUtternceLength: 20,
  };

  return sherpa_ncnn.createRecognizer(config);
}

const recognizer = createRecognizer();
const stream = recognizer.createStream();


let lastText = '';
let segmentIndex = 0;

const micInstance = mic({
  rate: recognizer.config.featConfig.samplingRate,
  channels: 1,
  debug: false,
  device: 'default',
  bitwidth: 16,
  encoding: 'signed-integer',
  exitOnSilence: 6,
  fileType: 'raw'
});

const micInputStream = micInstance.getAudioStream();

function clearConsole() {
  process.stdout.write('\x1B[2J\x1B[0f');
}

/**
 * SpeechSession class, work as a session manager with the formatOutput function
 * Sample output:
=== Automated Speech Recognition ===
Current Session #1
Time: 8:44:46 PM
------------------------
Recognized Sentences:
[8:44:43 PM] 1. it's so great three result is great great 她还支持中文
[8:44:46 PM] 2. 很厉
------------------------
Recognizing: 真的很厉害太厉害

*/
class SpeechSession {
  constructor() {
    this.startTime = Date.now();
    this.sentences = [];
    this.currentText = '';
    this.lastUpdateTime = Date.now();
  }

  addOrUpdateText(text) {
    this.currentText = text;
    this.lastUpdateTime = Date.now();
  }

  finalizeSentence() {
    if (this.currentText.trim()) {
      this.sentences.push({
        text: this.currentText.trim(),
        timestamp: new Date().toLocaleTimeString()
      });
    }
    this.currentText = '';
  }

  shouldStartNewSession() {
    return Date.now() - this.lastUpdateTime > 10000;  // 10 seconds of silence
  }
}


let currentSession = new SpeechSession();
let sessionCount = 1;

function formatOutput() {
  clearConsole();
  console.log('\n=== Automated Speech Recognition ===');
  console.log(`Current Session #${sessionCount}`);
  console.log('Time:', new Date().toLocaleTimeString());
  console.log('------------------------');

  // display history sentences
  if (currentSession.sentences.length > 0) {
    console.log('Recognized Sentences:');
    currentSession.sentences.forEach((sentence, index) => {
      console.log(`[${sentence.timestamp}] ${index + 1}. ${sentence.text}`);
    });
    console.log('------------------------');
  }

  // display the current sentence
  if (currentSession.currentText) {
    console.log('Recognizing:', currentSession.currentText);
  }
}


function exitHandler(options, exitCode) {
  if (options.cleanup) {
    console.log('\nCleaned up resources...');
    micInstance.stop();
    stream.free();
    recognizer.free();
  }
  if (exitCode || exitCode === 0) console.log('Exit code:', exitCode);
  if (options.exit) process.exit();
}

function startMic() {
  return new Promise((resolve, reject) => {
    micInputStream.once('startComplete', () => {
      console.log('Mic phone started.');
      resolve();
    });

    micInputStream.once('error', (err) => {
      console.error('Mic phone start error:', err);
      reject(err);
    });

    micInstance.start();
  });
}

micInputStream.on('data', buffer => {
  const int16Array = new Int16Array(buffer.buffer);
  const samples = new Float32Array(int16Array.length);

  for (let i = 0; i < int16Array.length; i++) {
    samples[i] = int16Array[i] / 32768.0;
  }

  stream.acceptWaveform(recognizer.config.featConfig.samplingRate, samples);

  while (recognizer.isReady(stream)) {
    recognizer.decode(stream);
  }

  const isEndpoint = recognizer.isEndpoint(stream);
  const text = recognizer.getResult(stream);

  if (text.length > 0) {
    // 检查是否需要开始新会话
    if (currentSession.shouldStartNewSession()) {
      currentSession.finalizeSentence();
      sessionCount++;
      currentSession = new SpeechSession();
    }

    currentSession.addOrUpdateText(text);
    formatOutput();
  }

  if (isEndpoint) {
    if (text.length > 0) {
      currentSession.finalizeSentence();
      formatOutput();
    }
    recognizer.reset(stream);
  }
});

micInputStream.on('error', err => {
  console.error('Audio stream error:', err);
});

micInputStream.on('close', () => {
  console.log('Mic phone closed.');
});

process.on('exit', exitHandler.bind(null, {cleanup: true}));
process.on('SIGINT', exitHandler.bind(null, {exit: true}));
process.on('SIGUSR1', exitHandler.bind(null, {exit: true}));
process.on('SIGUSR2', exitHandler.bind(null, {exit: true}));
process.on('uncaughtException', exitHandler.bind(null, {exit: true}));

async function main() {
  try {
    console.log('Starting ...');
    await startMic();
    console.log('Initialized, waiting for speech ...');
    formatOutput();
  } catch (err) {
    console.error('Failed to initialize:', err);
    process.exit(1);
  }
}

main();
