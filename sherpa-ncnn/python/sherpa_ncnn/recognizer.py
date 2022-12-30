from pathlib import Path

import numpy as np
from _sherpa_ncnn import FeatureExtractor, Model, ModelConfig, greedy_search


def _assert_file_exists(f: str):
    assert Path(f).is_file(), f"{f} does not exist"


def _read_tokens(tokens):
    sym_table = {}
    with open(tokens) as f:
        for line in f:
            sym, i = line.split()
            sym = sym.replace("▁", " ")
            sym_table[int(i)] = sym

    return sym_table


class Recognizer(object):
    """A class for streaming speech recognition.

    Please refer to
    `<https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html>`_
    to download pre-trained models for different languages, e.g., Chinese,
    English, etc.

    **Usage example**

    .. code-block:: python3

        import wave

        import numpy as np
        import sherpa_ncnn


        def main():
            recognizer = sherpa_ncnn.Recognizer(
                tokens="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/tokens.txt",
                encoder_param="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.param",
                encoder_bin="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.bin",
                decoder_param="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.param",
                decoder_bin="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.bin",
                joiner_param="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.param",
                joiner_bin="./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.bin",
                num_threads=4,
            )

            filename = "./sherpa-ncnn-conv-emformer-transducer-2022-12-06/test_wavs/1.wav"
            with wave.open(filename) as f:
                assert f.getframerate() == recognizer.sample_rate, (
                    f.getframerate(),
                    recognizer.sample_rate,
                )
                assert f.getnchannels() == 1, f.getnchannels()
                assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
                num_samples = f.getnframes()
                samples = f.readframes(num_samples)
                samples_int16 = np.frombuffer(samples, dtype=np.int16)
                samples_float32 = samples_int16.astype(np.float32)

                samples_float32 = samples_float32 / 32768

            recognizer.accept_waveform(recognizer.sample_rate, samples_float32)

            tail_paddings = np.zeros(int(recognizer.sample_rate * 0.5), dtype=np.float32)
            recognizer.accept_waveform(recognizer.sample_rate, tail_paddings)

            recognizer.input_finished()

            print(recognizer.text)


        if __name__ == "__main__":
            main()
    """

    def __init__(
        self,
        tokens: str,
        encoder_param: str,
        encoder_bin: str,
        decoder_param: str,
        decoder_bin: str,
        joiner_param: str,
        joiner_bin: str,
        num_threads: int = 4,
    ):
        """
        Please refer to
        `<https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html>`_
        to download pre-trained models for different languages, e.g., Chinese,
        English, etc.

        Args:
          tokens:
            Path to ``tokens.txt``. Each line in ``tokens.txt`` contains two
            columns::

                symbol integer_id
          encoder_param:
            Path to ``encoder.ncnn.param``.
          encoder_bin:
            Path to ``encoder.ncnn.bin``.
          decoder_param:
            Path to ``decoder.ncnn.param``.
          decoder_bin:
            Path to ``decoder.ncnn.bin``.
          joiner_param:
            Path to ``joiner.ncnn.param``.
          joiner_bin:
            Path to ``joiner.ncnn.bin``.
          num_threads:
            Number of threads for neural network computation.
        """
        _assert_file_exists(tokens)
        _assert_file_exists(encoder_param)
        _assert_file_exists(encoder_bin)
        _assert_file_exists(decoder_param)
        _assert_file_exists(decoder_bin)
        _assert_file_exists(joiner_param)
        _assert_file_exists(joiner_bin)

        assert num_threads > 0, num_threads

        self.sym_table = _read_tokens(tokens)

        model_config = ModelConfig(
            encoder_param=encoder_param,
            encoder_bin=encoder_bin,
            decoder_param=decoder_param,
            decoder_bin=decoder_bin,
            joiner_param=joiner_param,
            joiner_bin=joiner_bin,
            num_threads=num_threads,
        )

        self.model = Model.create(model_config)
        self.sample_rate = 16000

        self.feature_extractor = FeatureExtractor(
            feature_dim=80,
            sample_rate=self.sample_rate,
        )

        self.num_processed = 0  # number of processed feature frames so far
        self.states = []  # model state

        self.hyp = [0] * self.model.context_size  # initial hypothesis

        decoder_input = np.array(self.hyp, dtype=np.int32)
        self.decoder_out = self.model.run_decoder(decoder_input)

    def accept_waveform(self, sample_rate: float, waveform: np.array):
        """Decode audio samples.

        Args:
          sample_rate:
            Sample rate of the input audio samples. It should be 16000.
          waveform:
            A 1-D float32 array containing audio samples in the
            range ``[-1, 1]``.
        """
        assert sample_rate == self.sample_rate, (sample_rate, self.sample_rate)
        self.feature_extractor.accept_waveform(sample_rate, waveform)

        self._decode()

    def input_finished(self):
        """Signal that no more audio samples are available."""
        self.feature_extractor.input_finished()
        self._decode()

    @property
    def text(self):
        context_size = self.model.context_size
        text = [self.sym_table[token] for token in self.hyp[context_size:]]
        return "".join(text)

    def _decode(self):
        segment = self.model.segment
        offset = self.model.offset

        while self.feature_extractor.num_frames_ready - self.num_processed >= segment:
            features = self.feature_extractor.get_frames(self.num_processed, segment)
            self.num_processed += offset

            encoder_out, self.states = self.model.run_encoder(
                features=features,
                states=self.states,
            )

            self.decoder_out, self.hyp = greedy_search(
                model=self.model,
                encoder_out=encoder_out,
                decoder_out=self.decoder_out,
                hyp=self.hyp,
            )
