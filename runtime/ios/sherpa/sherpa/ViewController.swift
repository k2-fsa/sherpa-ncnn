//
//  ViewController.swift
//  sherpa
//
//  Created by 马丹 on 2022/11/21.
//

import UIKit
import AVFoundation

class ViewController: UIViewController {
    @IBOutlet weak var label: UILabel!
    @IBOutlet weak var button: UIButton!
    
    var sherpaModel: Sherpa?
    var audioEngine: AVAudioEngine?
    var startRecord: Bool?
    private var workItem: DispatchWorkItem?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        initModel()
        
        initRecorder()
    }
    
    func initModel() {
        let encoderParamPath = Bundle.main.path(forResource: "encoder_jit_trace-epoch-11-avg-2-pnnx.ncnn", ofType: "param")
        let encoderBinPath = Bundle.main.path(forResource: "encoder_jit_trace-epoch-11-avg-2-pnnx.ncnn", ofType: "bin")
        let decoderParamPath = Bundle.main.path(forResource: "decoder_jit_trace-epoch-11-avg-2-pnnx.ncnn", ofType: "param")
        let decoderBinPath = Bundle.main.path(forResource: "decoder_jit_trace-epoch-11-avg-2-pnnx.ncnn", ofType: "bin")
        let joinerParamPath = Bundle.main.path(forResource: "joiner_jit_trace-epoch-11-avg-2-pnnx.ncnn", ofType: "param")
        let joinerBinPath = Bundle.main.path(forResource: "joiner_jit_trace-epoch-11-avg-2-pnnx.ncnn", ofType: "bin")
        let tokensPath = Bundle.main.path(forResource: "tokens", ofType: "txt")
        sherpaModel = Sherpa(encoderParamPath:encoderParamPath, encoderBinPath:encoderBinPath, decoderParamPath:decoderParamPath, decoderBinPath:decoderBinPath, joinerParamPath:joinerParamPath, joinerBinPath:joinerBinPath, tokensPath:tokensPath)!
    }
    
    func initRecorder() {
        startRecord = false
        
        audioEngine = AVAudioEngine()
        let inputNode = self.audioEngine?.inputNode
        let bus = 0
        let inputFormat = inputNode?.outputFormat(forBus: bus)
        let outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                         sampleRate: 16000, channels: 1,
                                         interleaved: false)!
        let converter = AVAudioConverter(from: inputFormat!, to: outputFormat)!
        inputNode!.installTap(onBus: bus,
                              bufferSize: 1024,
                              format: inputFormat) {
            (buffer: AVAudioPCMBuffer, when: AVAudioTime) in
            var newBufferAvailable = true
            
            let inputCallback: AVAudioConverterInputBlock = {
                inNumPackets, outStatus in
                if newBufferAvailable {
                    outStatus.pointee = .haveData
                    newBufferAvailable = false
                    
                    return buffer
                } else {
                    outStatus.pointee = .noDataNow
                    return nil
                }
            }
            
            let convertedBuffer = AVAudioPCMBuffer(
                pcmFormat: outputFormat,
                frameCapacity:
                    AVAudioFrameCount(outputFormat.sampleRate)
                * buffer.frameLength
                / AVAudioFrameCount(buffer.format.sampleRate))!
            
            var error: NSError?
            let status = converter.convert(
                to: convertedBuffer,
                error: &error, withInputFrom: inputCallback)
            
            // 16000 Hz buffer
            let actualSampleCount = Int(convertedBuffer.frameLength)
            guard let floatChannelData = convertedBuffer.floatChannelData
            else { return }
            
            self.sherpaModel?.acceptWaveForm(floatChannelData[0],
                                            Int32(actualSampleCount))
        }
    }
    
    @IBAction func btnClicked(_ sender: Any) {
        if(!startRecord!) {
            //Clear result
            self.setResult(text: "")
            
            //Reset model
            
            //Start record
            do {
                try self.audioEngine?.start()
            } catch let error as NSError {
                print("Got an error starting audioEngine: \(error.domain), \(error)")
            }
            
            //Start decode thread
            workItem = DispatchWorkItem {
                while(!self.workItem!.isCancelled) {
                    self.sherpaModel?.decode()
                    DispatchQueue.main.sync {
                        self.setResult(text: (self.sherpaModel?.get_result())!)
                    }
                    Thread.sleep(forTimeInterval: 1)
                }
            }
            DispatchQueue.global().async(execute: workItem!)
            
            startRecord = true
            button.setTitle("Stop Record", for: UIControl.State.normal)
        } else {
            //Stop record
            self.audioEngine?.stop()
            
            //Stop decode thread
            workItem!.cancel()
            
            startRecord = false
            button.setTitle("Start Record", for: UIControl.State.normal)
        }
    }
    
    @objc func setResult(text: String) {
        label.text = text
    }
    
}

