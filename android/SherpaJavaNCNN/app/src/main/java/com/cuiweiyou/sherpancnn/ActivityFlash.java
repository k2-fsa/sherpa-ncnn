package com.cuiweiyou.sherpancnn;

import android.Manifest;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.View;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

/**
 * www.gaohaiyan.com
 */
public class ActivityFlash extends AppCompatActivity implements View.OnClickListener {
    
    private String[] permissions = {Manifest.permission.RECORD_AUDIO};
    private int CODE_RECORD_AUDIO_PERMISSION = 200;
    private View kotlinBtn;
    private View javaBtn;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_flash);
        
        kotlinBtn = findViewById(R.id.kotlin);
        kotlinBtn.setOnClickListener(this);
        kotlinBtn.setEnabled(false);
        
        javaBtn = findViewById(R.id.java);
        javaBtn.setOnClickListener(this);
        javaBtn.setEnabled(false);
        
        ActivityCompat.requestPermissions(this, permissions, CODE_RECORD_AUDIO_PERMISSION);
    }
    
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        boolean isPermissionGrant;
        if (requestCode == CODE_RECORD_AUDIO_PERMISSION) {
            isPermissionGrant = grantResults[0] == PackageManager.PERMISSION_GRANTED;
        } else {
            isPermissionGrant = false;
        }
        
        if (isPermissionGrant) { // 有权限了
            ToolRecorder.getInstance().init();
            ToolSherpaNcnn.getInstance().init(getAssets());
            kotlinBtn.setEnabled(true);
            javaBtn.setEnabled(true);
        } else {
            alertPremission();
        }
    }
    
    private void alertPremission() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setMessage("须要获得录音权限，本应用才可以有效运行(The permission must be grant.)");
        builder.setCancelable(false);
        builder.setNegativeButton("好的(OK)", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                dialog.dismiss();
                finish();
            }
        });
        builder.create().show();
    }
    
    @Override
    public void onClick(View v) {
        Intent i = new Intent(this, ActivityMain.class);
        
        if (v.getId() == R.id.kotlin) {
            i = new Intent(this, ActivityMain2.class);
        }
        
        startActivity(i);
        finish();
    }
}

