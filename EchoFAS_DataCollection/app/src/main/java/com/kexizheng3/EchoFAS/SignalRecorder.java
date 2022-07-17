package com.kexizheng3.EchoFAS;

/**
 * Class for record and save wav file
 * Reference: https://www.cnblogs.com/renhui/p/7457321.html
 * Written and modified by Kexin ZHENG
 * Date: 15/06/2022
 */

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.widget.Toast;

import androidx.core.app.ActivityCompat;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;

public class SignalRecorder {
    static final int sampleRate = 44100;// samples per second
    static final int channel = AudioFormat.CHANNEL_IN_STEREO;// two channels
    static final int encodeFormat = AudioFormat.ENCODING_PCM_16BIT;
    static final int bpf = 16; //   bits per sample

    private boolean isRecording;
    private int bufferSize;
    private AudioRecord recorder;
    private Thread recordingThread;
    private File recordFile = null;

    Context context;

    public SignalRecorder(Context ctx) {
        context = ctx;
    }

    //  Prepare file and initialize audio recorder for recording
    public void prepare(File fPath, int idx) {
        try {
            bufferSize = AudioRecord.getMinBufferSize(sampleRate,
                    channel,
                    encodeFormat);//   Minimum buffer size for AudioRecord
            if (ActivityCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(context, "Audio recording permission not granted", Toast.LENGTH_SHORT).show();
                return;
            }
            recorder = new AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    sampleRate,
                    channel,
                    encodeFormat,
                    bufferSize
            );//    Initialize AudioRecord
        } catch (IllegalArgumentException e) {
            if (recorder != null) {
                recorder.release();
            }
        }

        recordFile = new File(fPath.getAbsolutePath(), "audio_"+String.valueOf(idx)+".wav");
    }

    //  Start AudioRecord
    //  Initialize thread for writing data to file
    public void startRecording(){
        if (recorder != null && recorder.getState() == AudioRecord.STATE_INITIALIZED) {
            try {
                recorder.startRecording();
                isRecording = true;
                recordingThread = new Thread(new Runnable() {
                    @Override
                    public void run() {
                        writeDataToFile();
                    }
                }, "Write Data");

                recordingThread.start();

            } catch (IllegalStateException e) {
                e.printStackTrace();
            }
        }
    }

    //  Stop AudioRecord
    //  Stop file-writing thread
    public void stopRecording() {
        if (recorder != null) {
            isRecording = false;
            if (recorder.getState() == AudioRecord.STATE_INITIALIZED) {
                try {
                    recorder.stop();
                } catch (IllegalStateException e) {
                    e.printStackTrace();
                }
            }
            recorder.release();
            recordingThread.interrupt();
        }
    }

    //  Read audio data from AudioRecord and write it to file
    private void writeDataToFile() {
        byte[] data = new byte[bufferSize];
        FileOutputStream fos;
        try {
            fos = new FileOutputStream(recordFile);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            fos = null;
        }
        if (null != fos) {
            //  write empty header
            byte[] headerbytes = new byte[44];
            try{
                fos.write(headerbytes);
            } catch (IOException e) {
                e.printStackTrace();
            }
            //  write recording data with loop
            int chunksCount = 0;
            while (isRecording) {
                chunksCount += recorder.read(data, 0, bufferSize);
                if (AudioRecord.ERROR_INVALID_OPERATION != chunksCount) {
                    try {
                        fos.write(data);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
            try {
                fos.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
            updateWAVHeader(recordFile, 2);
        }
    }

/***********************    Write Header   ***********************/
    //  Update the empty header with complete information
     private void updateWAVHeader(File file, int channels) {
         long fileSize = file.length();// length of file in bytes
         long byteRate = sampleRate * channels * (bpf/8);// 1 byte = 8 bits
         try {
             final RandomAccessFile finalFile = randomAccessFile(file);
             finalFile.seek(0);
             finalFile.write(generateHeader(fileSize, sampleRate, channels, byteRate));
             finalFile.close();
         } catch (IOException e) {
             e.printStackTrace();
         }
     }

     // Initialize random access file to read/write at specific location
     private RandomAccessFile randomAccessFile(File file) {
         RandomAccessFile randomAccessFile;
         try {
         randomAccessFile = new RandomAccessFile(file, "rw");
         } catch (FileNotFoundException e) {
         throw new RuntimeException(e);
         }
         return randomAccessFile;
     }

     // Generate header for the wav file
     // Header provide information including sample rate, sample size, bit size, and length.
     // Reference: https://blog.csdn.net/VNanyesheshou/article/details/113954154
     //            http://www.topherlee.com/software/pcm-tut-wavformat.html
     private byte[] generateHeader(long fileSize,long sampleRate, int channels, long byteRate) {
         byte[] header = new byte[44];
         long hdataSize = fileSize -44;
         long hfileSize = fileSize - 8;
         // Mark the file as riff file
         header[0] = 'R';
         header[1] = 'I';
         header[2] = 'F';
         header[3] = 'F';
         // Determine the file size in bytes
         header[4] = (byte) (hfileSize & 0xff);
         header[5] = (byte) ((hfileSize >> 8) & 0xff);
         header[6] = (byte) ((hfileSize >> 16) & 0xff);
         header[7] = (byte) ((hfileSize >> 24) & 0xff);
         // File type header
         header[8] = 'W';
         header[9] = 'A';
         header[10] = 'V';
         header[11] = 'E';
         // Format chunk header
         header[12] = 'f';
         header[13] = 'm';
         header[14] = 't';
         header[15] = ' ';
         // The length of formatted data
         header[16] = 16;
         header[17] = 0;
         header[18] = 0;
         header[19] = 0;
         // Type of format, 1 for PCM
         header[20] = 1;
         header[21] = 0;
         // Number of Channel
         header[22] = (byte) channels;
         header[23] = 0;
         // Sample Rate
         header[24] = (byte) (sampleRate & 0xff);
         header[25] = (byte) ((sampleRate >> 8) & 0xff);
         header[26] = (byte) ((sampleRate >> 16) & 0xff);
         header[27] = (byte) ((sampleRate >> 24) & 0xff);
         // Byte Rate
         header[28] = (byte) (byteRate & 0xff);
         header[29] = (byte) ((byteRate >> 8) & 0xff);
         header[30] = (byte) ((byteRate >> 16) & 0xff);
         header[31] = (byte) ((byteRate >> 24) & 0xff);
         // Bytes per sample frame
         header[32] = (byte) (channels * (bpf/8));
         header[33] = 0;
         // Bits per sample
         header[34] = bpf;
         header[35] = 0;
         // The beginning of data
         header[36] = 'd';
         header[37] = 'a';
         header[38] = 't';
         header[39] = 'a';
         // Data size
         header[40] = (byte) (hdataSize & 0xff);
         header[41] = (byte) ((hdataSize >> 8) & 0xff);
         header[42] = (byte) ((hdataSize >> 16) & 0xff);
         header[43] = (byte) ((hdataSize >> 24) & 0xff);
         return header;
     }

}


