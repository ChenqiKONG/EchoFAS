package com.kexizheng3.EchoFAS;

/**
 * Activity for capture face image and record signal
 * Written by Kexin ZHENG
 * Date: 15/06/2022
 */

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import android.content.Context;
import android.media.AudioAttributes;
import android.media.AudioManager;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import com.google.common.util.concurrent.ListenableFuture;

import org.w3c.dom.Text;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;

public class DataCollectionActivity extends AppCompatActivity {
    private static String TAG = "ECHOFAS";

    private PreviewView previewView;
    private ListenableFuture<ProcessCameraProvider> mCameraProvider;
    private ImageCapture imageCapture;

    private AudioManager mAudioManager;
    private MediaPlayer signalFile;

    private SignalRecorder mSignalRecorder;

    private Button recordBtn;
    private EditText repeatTxt;
    private TextView cdTxt;

    private int recordCount = 0;
    private int repeatTimes = 0;

    private File saveFolder;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_data_collection);
        getWindow().addFlags (WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        previewView = findViewById(R.id.previewView);
        recordBtn = findViewById(R.id.record_btn);
        repeatTxt = findViewById(R.id.repeat_txt);
        cdTxt = findViewById(R.id.cd_txt);

        cameraSetup();

        //  Audio set up
        //  Ensure only top speaker can emit the signal
        mAudioManager = (AudioManager)getSystemService(Context.AUDIO_SERVICE);
        mAudioManager.setSpeakerphoneOn(false);//close speakerphone
        mAudioManager.setMode(AudioManager.MODE_IN_CALL);//mode in call
        setVolumeControlStream(AudioManager.STREAM_VOICE_CALL);//control top microphone volume

        //  Signal recorder setup
        mSignalRecorder = new SignalRecorder(DataCollectionActivity.this);

        //  Set on click listener for record button
        recordBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(repeatTimes==0) {
                    recordCount = 0;
                    try {
                        repeatTimes = Integer.valueOf(repeatTxt.getText().toString());
                    } catch (NumberFormatException e) {
                        Toast.makeText(DataCollectionActivity.this, "Please set repeat time." , Toast.LENGTH_SHORT).show();
                    }

                    if(repeatTimes == 0) {
                        Toast.makeText(DataCollectionActivity.this, "Repeat time should be larger than 0" , Toast.LENGTH_SHORT).show();
                    }
                    else {
                            //  Create save directory
                            SimpleDateFormat formatter = new SimpleDateFormat("MM_dd_HH_mm_ss", Locale.US);
                            Date now = new Date();
                            /** OPTION 1: SAVE AT APPLICATION DIRECTORY (Can be viewed with device explore in Android Studio) **/
                            saveFolder = new File(getApplicationContext().getFilesDir().getAbsolutePath(), formatter.format(now));
                            /** OPTION 2: SAVE AT DCIM FOLDER (required for Samsung note edge) **/
                            //saveFolder = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM),"EchoFAS");
                            //if (!saveFolder.exists())
                            //    saveFolder.mkdirs();
                            //saveFolder = new File(saveFolder,formatter.format(now));

                            if (!saveFolder.exists())
                                saveFolder.mkdirs();

                            //  prepare signal and start recording
                            prepareSignalFile("fassig");
                            startTimer();
                            playAndRecordSignal();

                            recordBtn.setEnabled(false);
                        }
                    }
                }
        });


    }

    /******************     Timer      ********************/
    //  2-second recording count down
    CountDownTimer cTimer = null;
    void startTimer() {
        cTimer = new CountDownTimer(2000, 100) {
            public void onTick(long millisUntilFinished) {
                cdTxt.setText(String.valueOf((int)millisUntilFinished/1000));
            }
            public void onFinish() {
                takePhoto();
                mSignalRecorder.stopRecording();

                recordCount++;
                cdTxt.setText("");
                cancelTimer();
            }
        };
        cTimer.start();
    }

    //  Cancel timer when count down ends
    void cancelTimer() {
        if(cTimer!=null)
            cTimer.cancel();
        if(recordCount<repeatTimes) {
            playAndRecordSignal();
            startTimer();
        }else{
            repeatTimes = 0;
            recordCount = 0;
            recordBtn.setEnabled(true);
            Toast.makeText(DataCollectionActivity.this, "Saved at" + saveFolder , Toast.LENGTH_SHORT).show();
        }
    }

    /***********************    Audio   ***********************/
    //  Prepare signal file for playing
    private void prepareSignalFile(String fileID){
        signalFile = new MediaPlayer();
        signalFile.setAudioAttributes( new AudioAttributes.Builder()
                .setLegacyStreamType(AudioManager.STREAM_VOICE_CALL)
                .build());
        Uri uri=Uri.parse("android.resource://"+getPackageName()+"/raw/"+fileID);
        try {
            signalFile.setDataSource(getApplicationContext(),uri);
            signalFile.prepare();
        }
        catch (IOException e) {
            Toast.makeText(DataCollectionActivity.this, "Failed to load signal" , Toast.LENGTH_SHORT).show();
        }
    }

    // Play signal and record echoes
    void playAndRecordSignal(){
        mSignalRecorder.prepare(saveFolder,recordCount);
        mSignalRecorder.startRecording();
        signalFile.start();
    }

    /***********************    Camera   ***********************/
    //  Camera set up
    //  Reference: https://developer.android.com/training/camerax/preview
    private void cameraSetup(){
        mCameraProvider = ProcessCameraProvider.getInstance(this);
        mCameraProvider.addListener(new Runnable() {
            @Override
            public void run() {
                try {
                    ProcessCameraProvider cameraProvider = mCameraProvider.get();
                    bindCamera(cameraProvider);
                } catch (ExecutionException | InterruptedException  e) {
                    e.printStackTrace();
                }
            }
        }, ContextCompat.getMainExecutor(this));
    }

    //  Bind Camera for preview and image capture
    //  Reference : https://developer.android.com/training/camerax/configuration
    //              https://developer.android.com/training/camerax/preview
    private void bindCamera(@NonNull ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder().build();
        imageCapture =
                new ImageCapture.Builder()
                        .setTargetRotation(previewView.getDisplay().getRotation())
                        .build();

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_FRONT).build();
        preview.setSurfaceProvider(previewView.createSurfaceProvider());

        cameraProvider.unbindAll();
        cameraProvider.bindToLifecycle((LifecycleOwner)this, cameraSelector,
                imageCapture, preview);
    }

    //  Save photo to local directory
    private void takePhoto(){
        String name = "img_" + String.valueOf(recordCount) + ".jpg";
        String filename = new File(saveFolder.getAbsolutePath(), name).getAbsolutePath();
        ImageCapture.OutputFileOptions mOutputFileOptions =
                new ImageCapture.OutputFileOptions.Builder(new File(filename)).build();
        Executor mExecutor = ContextCompat.getMainExecutor(this);
        ImageCapture.OnImageSavedCallback mSavedCallback =
                new ImageCapture.OnImageSavedCallback() {
                    @Override
                    public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults) {}
                    @Override
                    public void onError(ImageCaptureException error) {
                        error.printStackTrace();
                    }
                };
        imageCapture.takePicture(mOutputFileOptions, mExecutor,mSavedCallback);
    }

}
