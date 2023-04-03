HOST = ['127.0.0.1', '5620']
VO_UPLOAD_FOLDER = './cache/voice/' 
ALLOWED_EXTENSIONS = {'wav'}
ALLOWED_MIME_TYPES = {'audio/wav'} 

ENCODER_PARMA = './model/encoder_jit_trace-pnnx.ncnn.param'
ENCODER_BIN   = './model/encoder_jit_trace-pnnx.ncnn.bin'
DECODER_PARAM = './model/decoder_jit_trace-pnnx.ncnn.param'
DECODER_BIN   = './model/decoder_jit_trace-pnnx.ncnn.bin'
JOINER_PARAM  = './model/joiner_jit_trace-pnnx.ncnn.param'
JOINER_BIN    = './model/joiner_jit_trace-pnnx.ncnn.bin'
TOKENS        = './model/tokens.txt'
NUM_THREADS   = 4