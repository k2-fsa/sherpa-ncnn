import logging
import mimetypes
import os
import shlex
import shutil
import subprocess
import uuid
import wave

import numpy as np
import sherpa_ncnn
from flask import Flask, jsonify, render_template, request
from flask_caching import Cache

from config import *

Server = Flask(__name__)

Server.config['VO_UPLOAD_FOLDER'] = VO_UPLOAD_FOLDER
Server.config['CACHE_TYPE'] = 'simple'
Server.config['CACHE_DEFAULT_TIMEOUT'] = 1
cache = Cache(Server)

recognizer = sherpa_ncnn.Recognizer(
    tokens=TOKENS, encoder_param=ENCODER_PARMA,
    encoder_bin=ENCODER_BIN,decoder_param=DECODER_PARAM,
    decoder_bin=DECODER_BIN, joiner_param=JOINER_PARAM,
    joiner_bin=JOINER_BIN, num_threads=NUM_THREADS
)

def rewrite(input_file, output_file):
    shutil.copy(input_file, output_file)


@cache.memoize()
def Voice_recognition(filename):
    with wave.open(filename, 'rb') as f:
        if f.getsampwidth() != 2:
            raise ValueError(
                f"Invalid sample width: {f.getsampwidth()}, expected 2. File: {filename}")

        sample_rate = f.getframerate()
        num_channels = f.getnchannels()
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_int16 = samples_int16.reshape(-1, num_channels)[:, 0]
        samples_float32 = samples_int16.astype(np.float32)
        samples_float32 /= 32768

    recognizer.accept_waveform(sample_rate, samples_float32)
    tail_paddings = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
    recognizer.accept_waveform(sample_rate, tail_paddings)
    res1 = recognizer.text.lower()
    recognizer.stream = recognizer.recognizer.create_stream()
    return res1


def configure_app():
    if not os.path.exists(VO_UPLOAD_FOLDER):
        os.makedirs(VO_UPLOAD_FOLDER)
    cache.init_app(Server)


def configure_log():
    logging.basicConfig(level=logging.INFO, filename='./cache/log/server.log',
    format='%(levelname)s:%(asctime)s %(message)s')


def allowed_file(filename):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type is None or mime_type not in ['audio/wav', 'audio/x-wav']:
        return False
    return True


def check_type(mode):
    if 'file' not in request.files:
        raise ValueError('No file part.')

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        raise ValueError('Please upload a .wav file.')

    filename = str(uuid.uuid4()) + '.wav'
    filepath = os.path.join(Server.config[mode], filename)
    file.save(filepath)
    output_filepath = os.path.join(Server.config[mode], 'output_' + filename)
    return filepath, output_filepath


@Server.route('/voice', methods=['POST'])
def upload_file():
    try:
        if request.method == 'POST':
            filepath, output_filepath = check_type('VO_UPLOAD_FOLDER')
            rewrite(filepath, output_filepath)

            result = Voice_recognition(output_filepath)
            if os.path.exists(filepath):
                os.remove(filepath)
            if os.path.exists(output_filepath):
                os.remove(output_filepath)

            return jsonify({
                'status': 200,
                'message': result
            })

    except ValueError as e:
        return jsonify({
            'status': 400,
            'message': str(e)
        })

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        if os.path.exists(output_filepath):
            os.remove(output_filepath)
        logging.error(f"Recognition error: {e}")
        return jsonify({
            'status': 500,
            'message': 'Error, Please try again later.'
        })


@Server.route('/', methods=['GET'])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    configure_app()
    configure_log()
    print(f" * Running on http://{HOST[0]}:{HOST[1]}")
    Server.run(host=HOST[0], port=HOST[1], debug=False)
