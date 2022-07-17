package com.kexizheng3.EchoFAS;
/**
 * Activity for check and request permissions
 * Written by Kexin ZHENG
 * Date: 11/06/2022
 */

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity {

    private final String[] requiredPermissions = {
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.MODIFY_AUDIO_SETTINGS
    };
    private static final int PERMISSION_CODE = 123;
    private boolean granted;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        granted = false;

        //  Click button to grant permission and proceed to data collection
        //  If all required permissions are granted, proceed to the data collection page
        //  else request for the permissions
        Button startBtn = findViewById(R.id.start_btn);
        startBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                checkPermissions();
                if (granted) {
                    enterDataCollection();
                } else {
                    requestPermissions();
                }
            }
        });
    }

    //  Check whether all required permissions are granted
    public void checkPermissions(){
        granted = true;
        for(String perm : requiredPermissions) {
            if (ContextCompat.checkSelfPermission(getApplicationContext(), perm)!=PackageManager.PERMISSION_GRANTED) {
                granted = false;
            }
        }
    }

    //  Request for permissions to use camera, record audio, save file, and control audio settings
    public void requestPermissions(){
        ActivityCompat.requestPermissions(
                this,
                requiredPermissions,
                PERMISSION_CODE);
    }

    //Enter the data collection page
    public void enterDataCollection(){
        Intent intent = new Intent(this, DataCollectionActivity.class);
        startActivity(intent);
    }
}