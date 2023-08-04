
// RealtimeSpeechRecognitionDlg.h : header file
//

#pragma once


// CRealtimeSpeechRecognitionDlg dialog
class CRealtimeSpeechRecognitionDlg : public CDialogEx
{
// Construction
public:
	CRealtimeSpeechRecognitionDlg(CWnd* pParent = nullptr);	// standard constructor

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_REALTIMESPEECHRECOGNITION_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
};
