import wave, os, logging
import sherpa_ncnn, shlex
import subprocess, mimetypes
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_caching import Cache

# @Author> Actdream
# @Date  > 2023/4/1
# @Project> Performance Speech Recognition API

Server = Flask(__name__)  # Create a Flask application object
UPLOAD_FOLDER = './cache/'  # Upload file save directory
ALLOWED_EXTENSIONS = {'wav'}  # Allowed file extension types
Server.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # Configure the Flask object
Server.config['CACHE_TYPE'] = 'simple' # Set the cache type to 'simple'
Server.config['CACHE_DEFAULT_TIMEOUT'] = 1 # Set the default cache timeout to 1 second
cache = Cache(Server)  # Create Flask-Caching extension as caching tool

recognizer = sherpa_ncnn.Recognizer(
    tokens="./Aumodel/tokens.txt",
    encoder_param="./Aumodel/encoder_jit_trace-pnnx.ncnn.param",
    encoder_bin="./Aumodel/encoder_jit_trace-pnnx.ncnn.bin",
    decoder_param="./Aumodel/decoder_jit_trace-pnnx.ncnn.param",
    decoder_bin="./Aumodel/decoder_jit_trace-pnnx.ncnn.bin",
    joiner_param="./Aumodel/joiner_jit_trace-pnnx.ncnn.param",
    joiner_bin="./Aumodel/joiner_jit_trace-pnnx.ncnn.bin",
    num_threads=4,
)  # Initialize speech recognition engine

# ------------------------------------------------------------------------------------ config

def rewrite(input_file, output_file):
    '''
    Rewrite the audio file's sample rate as 16000
    input_file: The file path to be converted
    output_file: The converted file path
    '''
    command = ["./ffmpeg/run", "-i", shlex.quote(input_file), 
              "-ar", "16000", shlex.quote(output_file), "-y"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #if result.returncode == 0: # Print the converted and saved file
    #    print(f"File successfully saved {output_file}")
    #else:
    #    print("Error occurred:", result.stderr.decode('utf-8'))
    
def allowed_file(filename):
    '''
    Check if the file is in the allowed format
    filename: The name of the file
    '''
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type is None or mime_type not in ['audio/wav', 'audio/x-wav']:
        return False
    return True


@cache.memoize()  # Use caching to avoid duplicate processing of the same file
def recognition(filename):
    '''
    Perform speech recognition on an audio file.
    filename: path of the file to be recognized
    '''
    with wave.open(filename, 'rb') as f:
        if f.getframerate() != recognizer.sample_rate:
            raise ValueError(
                f"Invalid sample rate: {f.getframerate()}, expected {recognizer.sample_rate}. File: {filename}")
        if f.getnchannels() != 1:
            raise ValueError(
                f"Invalid number of channels: {f.getnchannels()}, expected 1. File: {filename}")
        if f.getsampwidth() != 2:
            raise ValueError(
                f"Invalid sample width: {f.getsampwidth()}, expected 2. File: {filename}")

        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)
        samples_float32 /= 32768

    recognizer.accept_waveform(recognizer.sample_rate, samples_float32)  # Accept audio data
    tail_paddings = np.zeros(
    int(recognizer.sample_rate * 0.5), dtype=np.float32)  # Add ending data
    recognizer.accept_waveform(recognizer.sample_rate, tail_paddings)
    # recognizer.input_finished()  # Notify the engine that input has finished and it can start recognition
    res1 = recognizer.text.lower()
    recognizer.reset() # Clear temporary results
    return res1 # Return recognition result


def configure_app():
    '''
    Configure the Flask application object.
    '''
    if not os.path.exists(UPLOAD_FOLDER):  
        os.makedirs(UPLOAD_FOLDER) # Create directory for uploaded files
    cache.init_app(Server) # Initialize caching


def configure_log():
    '''
    Configure logging output.
    '''
    logging.basicConfig(level=logging.INFO, filename='./log/server.log',
    format='%(levelname)s:%(asctime)s %(message)s')

# ------------------------------------------------------------------------------------ function

@Server.route('/voice', methods=['POST'])
def upload_file():
    '''
    Handle POST requests, receive uploaded audio files, and return speech recognition results.
    '''
    if 'file' not in request.files:
        return jsonify({
            'status': 400,
            'message': 'No file part.'
        })
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({
            'status': 400,
            'message': 'Please upload a .wav file.'
        })
    filename = secure_filename(file.filename)
    filepath = os.path.join(Server.config['UPLOAD_FOLDER'], filename) 
    file.save(filepath)

    output_filepath = os.path.join(
    Server.config['UPLOAD_FOLDER'], 'source.wav')
    rewrite(filepath, output_filepath) # Convert sample rate

    try:
        result = recognition(output_filepath)  # Perform speech recognition
        # print(result)
        os.remove(filepath)  # Delete uploaded file
        return jsonify({
            'status': 200,
            'message': result
        })
    except Exception as e:
        os.remove(filepath)
        logging.error(f"Recognition error: {e}")
        return jsonify({
            'status': 500,
            'message': 'Internal server error, Please try again later.'
        })


if __name__ == '__main__':
    configure_app()
    configure_log()
    Server.run(host='127.0.0.1', port=5620, debug=False) # Start the service