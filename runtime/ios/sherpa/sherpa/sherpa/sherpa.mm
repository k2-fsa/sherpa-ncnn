//
//  sherpa.m
//  sherpa
//
//  Created by 马丹 on 2022/11/27.
//

#include "sherpa.h"

#undef DEBUG

#include "sherpa-ncnn/csrc/decode.h"
#include "sherpa-ncnn/csrc/features.h"
#include "sherpa-ncnn/csrc/lstm-model.h"
#include "sherpa-ncnn/csrc/symbol-table.h"

@implementation Sherpa {
@protected
    std::shared_ptr<sherpa_ncnn::SymbolTable> sym;
    std::shared_ptr<sherpa_ncnn::LstmModel> model;
    std::shared_ptr<sherpa_ncnn::FeatureExtractor> feature_extractor;
    ncnn::Mat decoder_input;
    ncnn::Mat decoder_out;
    std::vector<int32_t> hyp;
    
    int32_t segment;
    int32_t offset;
    
    int32_t context_size;
    int32_t blank_id;
    
    int32_t num_tokens;
    int32_t num_processed;
    
    std::string result;
}

- (nullable instancetype)initWithEncoderParamPath:
(NSString*)encoderParamPath EncoderBinPath:(NSString*)encoderBinPath DecoderParamPath:
(NSString*)decoderParamPath DecoderBinPath:(NSString*)decoderBinPath JoinerParamPath:
(NSString*)joinerParamPath JoinerBinPath:(NSString*)joinerBinPath TokensPath:(NSString*)tokensPath {
    self = [super init];
    if (self) {
        try {
            sym = std::make_shared<sherpa_ncnn::SymbolTable>(tokensPath.UTF8String);
            
            int32_t num_threads = 4;
            
            model = std::make_shared<sherpa_ncnn::LstmModel>(encoderParamPath.UTF8String,
                                                             encoderBinPath.UTF8String,
                                                             decoderParamPath.UTF8String,
                                                             decoderBinPath.UTF8String,
                                                             joinerParamPath.UTF8String,
                                                             joinerBinPath.UTF8String,
                                                             num_threads);
            
            feature_extractor = std::make_shared<sherpa_ncnn::FeatureExtractor>();
            
            segment = 9;
            offset = 4;
            
            context_size = model->ContextSize();
            blank_id = model->BlankId();
            
            hyp = std::vector<int32_t>(context_size, blank_id);
            
            decoder_input = ncnn::Mat(context_size);
            for (int32_t i = 0; i != context_size; ++i) {
                static_cast<int32_t *>(decoder_input)[i] = blank_id;
            }
            
            decoder_out = model->RunDecoder(decoder_input);
            
            num_tokens = hyp.size();
            num_processed = 0;
        } catch (const std::exception& exception) {
            NSLog(@"%s", exception.what());
            return nil;
        }
    }
    return self;
}

- (void)acceptWaveForm: (float*)pcm: (int)size {
    feature_extractor->AcceptWaveform(16000, pcm, size);
}

- (void)decode {
    ncnn::Mat hx;
    ncnn::Mat cx;
    
    while (feature_extractor->NumFramesReady() - num_processed >= segment) {
        ncnn::Mat features = feature_extractor->GetFrames(num_processed, segment);
        num_processed += offset;
        
        ncnn::Mat encoder_out = model->RunEncoder(features, &hx, &cx);
        
        sherpa_ncnn::GreedySearch(*model, encoder_out, &decoder_out, &hyp);
    }
    
    if (hyp.size() != num_tokens) {
        num_tokens = hyp.size();
        std::string text;
        for (int32_t i = context_size; i != hyp.size(); ++i) {
            text += (*sym)[hyp[i]];
        }
        fprintf(stderr, "%s\n", text.c_str());
        result = text;
    }
}

- (NSString*)get_result {
    return [NSString stringWithUTF8String:result.c_str()];
}

@end
