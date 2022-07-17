import shutil, os
from pathlib import Path

import numpy as np
from scipy.io.wavfile import write
from scipy import signal as signal
import math as math

#   Sine wave pilot signal
def getPilot(pilotLen, freqHz, sampleRate):
    result = []
    for i in range(pilotLen):
        val = math.sin(math.pi * 2 * i * freqHz / sampleRate)
        result.append(val)
    return result

#   Sweep signal
def getSweepSignal(signalLen, startFreq, endFreq, sampleRate):
    result = []
    for i in range(signalLen):
        numerator = i / signalLen
        currentFreq = startFreq + (numerator * (endFreq - startFreq))/2
        value = math.sin(math.pi * 2 * i * currentFreq/sampleRate)
        result.append(value)
    return result

#   Apply hamming window on generated sweeping signal
def getWindowedSweep(sweepSignal):
    hWindow = signal.windows.hamming(len(sweepSignal))
    result = np.multiply(sweepSignal,hWindow)
    return result

#   Matched filter
#   Measure the similarity between two signal
def calcMatchedFilter(data,impulseR):
    res = np.array(range(len(data)),dtype=float)
    for i in range(len(data)):
        sum = 0
        for j in range(len(impulseR)):
            if(j+i) >= len(data):
                break
            sum+=impulseR[j] * data[i+j]
        res[i] = sum
    return res

#   Cut signal
def cutSignal(hIdx, data, lenn):
    res = data[hIdx:hIdx + lenn]
    if len(data) - hIdx - 1 < lenn:
        temp = np.zeros((lenn, 1), float)
        for i in range(0, len(data) - hIdx):
            temp[i] = data[hIdx + i]
        res = temp.reshape(-1)
    return res
    
#   Save data to wav file
def saveDataWAV(data,cIdx,filePath,Fs):
    filePath = os.path.join(filePath,'audio')
    for i in range(data.shape[0]):
        echoes = np.ndarray((data.shape[1], 1), np.int16)
        echoes[:, 0] = data[i]
        subDir = 'clips_' + str(cIdx)
        try:
            Path(os.path.join(filePath,subDir)).mkdir(parents=True, exist_ok=False)
        except:
            pass
        fName = 'cut_' + str(i) + '.wav'
        newName = os.path.join(filePath, subDir,fName)
        write(newName, Fs, echoes.astype(np.int16))

#   Save data to npy file
def saveDataNpy(data,cIdx,filePath):
    filePath = os.path.join(filePath,'audio')
    try:
        Path(filePath).mkdir(parents=True, exist_ok=False)
    except:
        pass
    fName = 'clips_' + str(cIdx) + '.npy'
    np.save(os.path.join(filePath,fName), data.astype(np.int16))

#   Move and rename image
def moveImg(oldF,newF,oldP,newP):
    subP = os.path.join(newP,'img')
    try:
        Path(subP).mkdir(parents=True, exist_ok=False)
    except:
        pass
    shutil.copy(os.path.join(oldP,oldF), os.path.join(subP ,newF))

if __name__ == "__main__":
    print("Utils")