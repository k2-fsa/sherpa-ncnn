from pathlib import Path

import numpy as np
from _sherpa_ncnn import (
    DecoderConfig,
    EndpointConfig,
    FeatureExtractorConfig,
    ModelConfig,
)
from _sherpa_ncnn import Recognizer as _Recognizer
from _sherpa_ncnn import RecognizerConfig


def _assert_file_exists(f: str):
    assert Path(f).is_file(), f"{f} does not exist"


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
        decoding_method: str = "greedy_search",
        num_active_paths: int = 4,
        enable_endpoint_detection: bool = False,
        rule1_min_trailing_silence: int = 2.4,
        rule2_min_trailing_silence: int = 1.2,
        rule3_min_utterance_length: int = 20,
        model_sample_rate: int = 16000,
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
          decoding_method:
            Valid decoding methods are: greedy_search, modified_beam_search.
          num_active_paths:
            Used only when decoding_method is modified_beam_search. Its value
            is ignored when decoding_method is greedy_search. It specifies
            the maximum number of paths to use in beam search.
          enable_endpoint_detection:
            True to enable endpoint detection. False to disable endpoint
            detection.
          rule1_min_trailing_silence:
            Used only when enable_endpoint_detection is True. If the duration
            of trailing silence in seconds is larger than this value, we assume
            an endpoint is detected.
          rule2_min_trailing_silence:
            Used only when enable_endpoint_detection is True. If we have decoded
            something that is nonsilence and if the duration of trailing silence
            in seconds is larger than this value, we assume an endpoint is
            detected.
          rule3_min_utterance_length:
            Used only when enable_endpoint_detection is True. If the utterance
            length in seconds is larger than this value, we assume an endpoint
            is detected.
          model_sample_rate:
            Sample rate expected by the model
        """
        _assert_file_exists(tokens)
        _assert_file_exists(encoder_param)
        _assert_file_exists(encoder_bin)
        _assert_file_exists(decoder_param)
        _assert_file_exists(decoder_bin)
        _assert_file_exists(joiner_param)
        _assert_file_exists(joiner_bin)

        assert num_threads > 0, num_threads
        assert decoding_method in (
            "greedy_search",
            "modified_beam_search",
        ), decoding_method
        feat_config = FeatureExtractorConfig(
            sampling_rate=model_sample_rate,
            feature_dim=80,
        )

        model_config = ModelConfig(
            encoder_param=encoder_param,
            encoder_bin=encoder_bin,
            decoder_param=decoder_param,
            decoder_bin=decoder_bin,
            joiner_param=joiner_param,
            joiner_bin=joiner_bin,
            num_threads=num_threads,
            tokens=tokens,
        )

        endpoint_config = EndpointConfig(
            rule1_min_trailing_silence=rule1_min_trailing_silence,
            rule2_min_trailing_silence=rule2_min_trailing_silence,
            rule3_min_utterance_length=rule3_min_utterance_length,
        )

        decoder_config = DecoderConfig(
            method=decoding_method,
            num_active_paths=num_active_paths,
        )

        self.config = RecognizerConfig(
            feat_config=feat_config,
            model_config=model_config,
            decoder_config=decoder_config,
            endpoint_config=endpoint_config,
            enable_endpoint=enable_endpoint_detection,
        )

        self.sample_rate = self.config.feat_config.sampling_rate

        self.recognizer = _Recognizer(self.config)
        self.stream = self.recognizer.create_stream()

    def accept_waveform(self, sample_rate: float, waveform: np.array):
        """Decode audio samples.

        Args:
          sample_rate:
            Sample rate of the input audio samples. You must use the same
            value across different calls to `accept_waveform`! If it
            is different from self.sample_rate, we will do resampling inside.
          waveform:
            A 1-D float32 array containing audio samples in the
            range ``[-1, 1]``.
        """
        self.stream.accept_waveform(sample_rate, waveform)
        self._decode()

    def input_finished(self):
        """Signal that no more audio samples are available."""
        self.stream.input_finished()
        self._decode()

    def _decode(self):
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

    @property
    def text(self):
        return self.recognizer.get_result(self.stream).text

    @property
    def is_endpoint(self):
        return self.recognizer.is_endpoint(self.stream)

    def reset(self):
        self.recognizer.reset(self.stream)
