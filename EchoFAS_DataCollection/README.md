# EchoFAS_DataCollection
Data Collection Android App for EchoFAS. </br>
This App will play EchoFAS signal and record data that includes the face echoes. The face image capture will be performed at the same time.</br>
The recorded data can be processed with our [preprocessing program](https://github.com/whosoever/EchoFAS_Processing) to extract the target face echoes.

# How to run
1. Clone the repository
```bash
git clone https://github.com/
```
2. Connect your Android device to the computer
3. Open and build the project with Android Studio
4. Open the App on the device
5. Grant the required permission of the App
6. Input the repeat time and click the record button
7. Start recording :)

# Where to find the data
We offers two save directory options in the current code. Both of them can be viewed and downloaded with the device file explorer in Android Studio.
The corresponding code can be found in the [DataCollectionActivity.java](https://github.com/whosoever/EchoFAS_DataCollection/blob/main/app/src/main/java/com/kexizheng3/EchoFAS/DataCollectionActivity.java) code.
- **Option 1: Save at application directory (Recommended)**</br>
Sample Directory: /data/data/com.kexizheng3.EchoFAS/files/07_16_23_16_55</br>
This allows you to save the data at the app-specific storage that other apps cannot access.

- **Option 2: Save at external directory (Required for Samsung NoteEdge)**</br>
Sample Directory: /sdcard/DCIM/EchoFAS/07_16_23_18_00</br>
This allows you to save the data at the external storage where you can view on your mobile directly. However, it might be cause error during recording.</br>
This is required for Samsung NoteEdge due to its system settings.

**To Open the Device File Explorer in Android Studio**</br>
On the Navigation bar: View -> Tool Windows -> Device File Explorer






