//
//  sherpa.h
//  sherpa
//
//  Created by 马丹 on 2022/11/27.
//

#ifndef sherpa_h
#define sherpa_h

#include <stdio.h>

#import <Foundation/Foundation.h>

@interface Sherpa : NSObject

- (nullable instancetype)initWithEncoderParamPath:
(NSString*)encoderParamPath EncoderBinPath:(NSString*)encoderBinPath DecoderParamPath:
(NSString*)decoderParamPath DecoderBinPath:(NSString*)decoderBinPath JoinerParamPath:
(NSString*)joinerParamPath JoinerBinPath:(NSString*)joinerBinPath TokensPath:(NSString*)tokensPath;

- (void)acceptWaveForm: (float*)pcm: (int)size;

- (void)decode;

- (NSString*)get_result;

@end

#endif /* sherpa_h */
