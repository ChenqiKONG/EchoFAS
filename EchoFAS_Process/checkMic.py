from scipy.io.wavfile import read
from utils import *
import sys
import argparse

def main(args):
    pilot = getPilot(250, 11025, 44100) 
    mainPath = args.main_path
    dataNum = int(len(os.listdir(mainPath))/2)
    micmain = 0
    for cIdx in range(dataNum):
        fileName = 'audio_' + str(cIdx) + '.wav'
        Fs, data = read(os.path.join(mainPath,fileName))
        corr0 = signal.correlate(data[:40000,0], pilot, mode='same')
        corr1 = signal.correlate(data[:40000,1], pilot, mode='same')
        if np.amax(corr0)<np.amax(corr1):
            micmain = micmain + 1

    print(micmain)  
    if micmain/dataNum < 0.5:
        print("Main mic is 0")
    else:
        print("Main mic is 1")

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_path', type=str, default="/Users/kexinzheng/Documents/data/zkx2",help="Directory for raw recording")
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
