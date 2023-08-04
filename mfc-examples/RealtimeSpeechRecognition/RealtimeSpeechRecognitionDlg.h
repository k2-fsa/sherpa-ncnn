
// RealtimeSpeechRecognitionDlg.h : header file
//

#pragma once

#include <string>
#include <vector>

#include "portaudio.h"
#include "sherpa-ncnn/c-api/c-api.h"

class Microphone {
 public:
  Microphone();
  ~Microphone();
};

class RecognizerThread;

// CRealtimeSpeechRecognitionDlg dialog
class CRealtimeSpeechRecognitionDlg : public CDialogEx {
  // Construction
 public:
  CRealtimeSpeechRecognitionDlg(
      CWnd *pParent = nullptr);  // standard constructor
  ~CRealtimeSpeechRecognitionDlg();

// Dialog Data
#ifdef AFX_DESIGN_TIME
  enum { IDD = IDD_REALTIMESPEECHRECOGNITION_DIALOG };
#endif

 protected:
  virtual void DoDataExchange(CDataExchange *pDX);  // DDX/DDV support

  // Implementation
 protected:
  HICON m_hIcon;

  // Generated message map functions
  virtual BOOL OnInitDialog();
  afx_msg void OnPaint();
  afx_msg HCURSOR OnQueryDragIcon();
  DECLARE_MESSAGE_MAP()
 public:
  afx_msg void OnBnClickedOk();
  int RunThread();

 private:
  Microphone mic_;

  SherpaNcnnRecognizer *recognizer_ = nullptr;

  PaStream *pa_stream_ = nullptr;
  RecognizerThread *thread_ = nullptr;

 public:
  bool started_ = false;
  SherpaNcnnStream *stream_ = nullptr;

 public:
  CButton my_btn_;
  CEdit my_text_;

 private:
  void AppendTextToEditCtrl(const std::string &s);
  void AppendLineToMultilineEditCtrl(const std::string &s);
  void InitMicrophone();

  bool Exists(const std::string &filename);
  void InitRecognizer();
};

class RecognizerThread : public CWinThread {
 public:
  RecognizerThread(CRealtimeSpeechRecognitionDlg *dlg) : dlg_(dlg) {}
  virtual BOOL InitInstance() { return TRUE; }
  virtual int Run() { return dlg_->RunThread(); }

 private:
  CRealtimeSpeechRecognitionDlg *dlg_;
};
