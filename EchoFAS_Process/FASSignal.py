from utils import *
from scipy.io.wavfile import write

class FASSignal:
    def __init__(self,Fs, chirpLenth = 60,chirpInterval = 3000):
        self.sampleRate = Fs
        self.pilot = None
        self.chirps = None
        self.completeSignal = None

        self.prepare(chirpLenth,chirpInterval)
    
    #   Prepare the signal including pilot tone, repeated chirps and intervals
    def prepare(self,chirpLen,intervalLen):
        self.pilot = getPilot(250, 11025, self.sampleRate)  # 11.025k Hz Pilot signal 
        self.completeSignal = np.append(self.pilot,np.zeros((1,8000)))  # 8000 interval between pilot tone and chirps
        # 9 chirps 12k-17k 14k-19k 16k-21k
        self.chirps = np.ndarray((9, chirpLen), float)
        for i in range(3):
            startFreq = 12000 + i * 2000
            endFreq = 17000 + i * 2000
            res = getSweepSignal(chirpLen, startFreq,endFreq, self.sampleRate)
            sig = getWindowedSweep(res)
            sigarray = np.array(sig, dtype=float)
            self.chirps[i * 3] = sigarray
            self.chirps[i * 3 + 1] = sigarray
            self.chirps[i * 3 + 2] = sigarray

            #   Interval between chirps should be 3500
            padZeros = np.zeros((1, intervalLen))
            padSig = np.append(sigarray,padZeros)
            self.completeSignal = np.append(self.completeSignal,padSig)
            self.completeSignal = np.append(self.completeSignal,padSig)
            self.completeSignal = np.append(self.completeSignal,padSig)
        
    # Save complete signal as wav file
    def saveSignal(self,savePath):
        finalSig = self.completeSignal*32767
        outputSig = np.ndarray((len(finalSig),2))
        outputSig[:,0] = finalSig
        outputSig[:,1] = finalSig
        write(savePath, self.sampleRate, outputSig.astype(np.int16))


if __name__ == '__main__':
    print("FAS Signal Class")
    sig = FASSignal(44100)
    sig.saveSignal("/Users/xxx/Documents/data/fassignal.wav")