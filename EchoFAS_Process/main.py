import argparse
import os
import sys

import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import *
from FASSignal import FASSignal



REPEAT_TIME = 9

def main(args):
    print("Start Processing")
    # Load Data
    mainPath = args.main_path
    dataNum = int(len(os.listdir(mainPath))/2)
    totalData,sampleRate = audioLoader(mainPath,dataNum,args.main_mic)
    # Initialize EchoFAS signal
    fasSig = FASSignal(sampleRate)

    realIdx = 0
    for i in tqdm(range(dataNum)):
        echoes = processSignal(totalData[i],fasSignal=fasSig)
        if echoes.all() == 0:
            print('Invalid Audio_' + str(i))
            continue

        oldfile = 'img_' + str(i) + '.jpg'
        newfile = 'img_' + str(realIdx) + '.jpg'
        newpath = args.new_path
        if args.use_npy:
            saveDataNpy(echoes,realIdx,newpath) # Save processed data to npy file
        else:
            saveDataWAV(echoes,realIdx,newpath,sampleRate)  # Save processed data to wav file
        moveImg(oldfile,newfile,mainPath,newpath)
        realIdx = realIdx+1

# Read recorded data
# mainMic could be different for different phone
def audioLoader(pathMain,numData,mainMic):
    dataset = np.ndarray((numData,68000,2))
    for cIdx in range(numData):
        fileName = 'audio_' + str(cIdx) + '.wav'
        Fs, data = read(os.path.join(pathMain,fileName))
        newdata = np.ndarray((68000,2))
        newdata[:,0] = data[:68000,mainMic]
        newdata[:,1] = data[:68000,1-mainMic]
        dataset[cIdx] = newdata
    return dataset,Fs

# Remove direct transmission
# Locate the direct transmission with matched filter
def removeDirectTransmission(data, impulseR):
    res = calcMatchedFilter(data, impulseR)
    lres = res * res
    highest_peak_index = np.argmax(lres)# the beginning of direct transmission
    if(highest_peak_index > 200):# abandon low quality clip
        return np.zeros((200))
    residual_reflection = cutSignal(highest_peak_index + 60, data, 200)
    return residual_reflection

# split receive signal into 9 3060 = 60 samples + 3000 samples clips
def splitReceivedSignal(data):
    result = np.ndarray((REPEAT_TIME, 400), dtype=float)
    for i in range(REPEAT_TIME):
        offset = 3060 * i
        res = np.array(data[offset: offset + 400], dtype=float)
        result[i] = res
    return result

# Estimate propagation delay time
def estimatePropagation(data, impulseR):
    idxl = 0
    idxr = 70
    stdRef = sys.maxsize
    tm1 = 0
    for pt in range(idxl, idxr):
        highest_peaks_a = []
        for i in range(REPEAT_TIME):
            res= calcMatchedFilter(data[i][pt:pt+60], impulseR[i])
            lres = res * res
            highest_peak = np.argmax(lres)
            highest_peaks_a.append(highest_peak + pt)
        nphighest_peaks_a = np.array(highest_peaks_a)
        stdA = np.std(nphighest_peaks_a)
        meanA = np.mean(nphighest_peaks_a)
        calcRef =  stdA
        if calcRef < stdRef:
            stdRef = calcRef
            tm1 = meanA
    return tm1

# Main signal process step
def processSignal(dataTotal,fasSignal):
    faceEchoes = np.zeros((REPEAT_TIME,60))

    data = dataTotal[:, 0]
    b, a = signal.butter(5, 0.45, 'highpass')    # high pass filter eliminate <10k noise
    filteredda = signal.filtfilt(b, a, data)

    #### SIGNAL SYNCHRONIZATION ###
    pilot = fasSignal.pilot
    corra = signal.correlate(data[:40000], pilot, mode='same')

    highest_peak_index = np.argmax(corra)
    if highest_peak_index >32000:    # Abandon low quality clip
        return faceEchoes
    slicedda = filteredda[highest_peak_index+8120:len(data)]
    splitedda = splitReceivedSignal(slicedda)     # Split data into nine clips

    ### DIRECT TRANSMISSION REMOVAL ###
    cleanSig = np.ndarray((REPEAT_TIME, 200))
    for i in range(REPEAT_TIME):
        cSig = removeDirectTransmission(splitedda[i],fasSignal.chirps[i])
        if cSig.all()==0:
            print("Invalid direct transmission removal")
            return faceEchoes
        cleanSig[i] = cSig

    ### ESTIMATE PROPAGATION DELAY TIME ###
    delayTime = estimatePropagation(cleanSig,fasSignal.chirps)

    ### EXTRACT TARGET FACE ECHOES ###
    for i in range(REPEAT_TIME):
        faceEchoes[i]=  cutSignal(round(delayTime), cleanSig[i], 60)

    return faceEchoes

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--main_path', type=str, default="/Users/kexinzheng/Documents/data/ZHENG_Kexin_paper/s21/35cm",help="Directory for raw recording")
    parser.add_argument('--new_path', type=str, default="/Users/kexinzheng/Documents/data/zkx1",help="Directory for processed data")
    parser.add_argument('--main_mic', type=int, default=0,help="The order of top and bottom microphone in recorded file might be different for different phone")
    parser.add_argument('--use_npy', type=bool, default=False,help="If True, save processed data as npy file, else save as wav")

    return parser.parse_args()

if __name__ == '__main__':
   main(parse_arguments(sys.argv[1:]))


