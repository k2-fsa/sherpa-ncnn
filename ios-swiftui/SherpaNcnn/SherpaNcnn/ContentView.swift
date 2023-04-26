//
//  ContentView.swift
//  SherpaNcnn
//
//  Created by knight on 2023/4/5.
//

import SwiftUI

struct ContentView: View {
    @StateObject var sherpaNcnnVM = SherpaNcnnViewModel()
    
    var body: some View {
        VStack {
            Text("ASR with Next-gen Kaldi")
                .font(.title)
            if sherpaNcnnVM.status == .stop {
                Text("See https://github.com/k2-fsa/sherpa-ncnn")
                Text("Press the Start button to run!")
            }
            ScrollView(.vertical, showsIndicators: true) {
                HStack {
                    Text(sherpaNcnnVM.subtitles)
                    Spacer()
                }
            }
            Spacer()
            Button {
                toggleRecorder()
            } label: {
                Text(sherpaNcnnVM.status == .stop ? "Start" : "Stop")
            }
        }
        .padding()
    }
    
    private func toggleRecorder() {
        sherpaNcnnVM.toggleRecorder()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
