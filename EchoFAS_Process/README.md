# EchoFAS_Process
 This is the code of data-preprocessing for EchoFAS.
 
 The code will extract target face region echoes from the raw data.
 
 The input WAV file data can be collected with our data collection [Android APP](https://pip.pypa.io/en/stable/).
 
# Libraries
The code uses the following libraries
- Numpy
- Scipy
- tqdm

# How to Run
Clone the repository
```bash
git clone https://github.com/ChenqiKONG/EchoFAS.git
cd EchoFAS/EchoFAS_Process
```
1. Save the processed data in WAV format
```bash
python main.py --main_path /Path/That/Stores/Raw/Data --new_path /Path/That/Stores/Processed/Data --main_mic 0 --use_npy False 
```
2. Save the processed data in NPY format
```bash
python main.py --main_path /Path/That/Stores/Raw/Data --new_path /Path/That/Stores/Processed/Data --main_mic 0 --use_npy True 
```
NOTE: Environment interference may affect the quality of recording. The code will remove invalid data automatically, so the number of final processed data might be less than the number of raw data.

# How to define the main mic
Since our recording is stereo, we have two channels in the raw recording data, representing the top and bottom microphone. Due to system differences, the channel order of microphones in the recording might be different for different phones. 
We only use data from the top microphone so correctly define the main mic is very important.

From our experiment, the main mic channel is 0 for Samsung S9 and Samsung NoteEdge. The main mic channel is 1 for Samsung S21 and Xiaomi Redmi 7.
We provide code to define the main mic channel for new device.
```bash
python checkMic.py --main_path /Path/That/Stores/Raw/Data/Of/One/Device
```
The system prints Main mic is 0 if main mic channel is 0<br />
The system prints Main mic is 1 if main mic channel is 1

 
