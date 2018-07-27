import numpy as np
import librosa
import librosa.display
import glob
import scipy.fftpack as fp
#import matplotlib.pyplot as plt

class Sound(object):

    # Constructor
    def __init__(self, path = None, extension = None):
        self.raw = []               # Get sound data
        self.path = path            # Get data path
        self.extension = extension  # Get data extension
        self.dataNames = glob.glob(self.path + "*." + self.extension)  # Make name package
        self.dataNum = self.dataNames.__len__() # Get data number
        print("Data path: ", self.path)         # For debugging
        print("Extension: ", self.extension)    # For debugging
        print("Data number: ", self.dataNum)    # For debugging

    # Load raw data & attach to one chunk
    def load(self):
        for i in range(self.dataNum):
            buf, self.sr = librosa.load(self.dataNames.pop(0))  # Get raw data and sampling rate
            self.raw.append(buf)    # Collect data as one array
        print("Data names: ", self.dataNames)   # For debugging
        print("Raw data set: ", self.raw)       # For debugging
        print("Sampling Rate: ", self.sr)       # For debugging

    # Cut data using sampling rate
    def dataCutting(self):
        self.cutSound = []
        # data is cut by Sampling Rate multiple
        for i in range(self.dataNum):
            tmp = len(self.raw[i])
            self.cutting = 0
            while tmp > self.sr:
                tmp -= self.sr
                self.cutting += self.sr
            self.raw[i] = self.raw[i][:self.cutting]
            # self.cutSound.append(self.raw[i][:self.cutting])
            print(i, "Cut sound is made")
            # print("cutSound[", i, "]shape: ", self.cutSound[i].shape)
            self.cutSound = np.hstack((self.cutSound, self.raw[i]))

    # Preprocess Work: Normalize data
    # Preprocess for get feature of data
    def Process(self):
        self.fft = []
        mask = int(self.sr / 5)     # Mask size: 4410
        print("Mask: ", mask)       # Time slice: 0.2 sec
        # Data Normalization
        self.cutSound = librosa.util.normalize(self.cutSound)
        for i in range(0, self.cutting + 1 - mask, 2205):
            buf = self.cutSound[i:i + mask]         # Cut data as mask size
            buf = np.fft.fft(buf)                   # Apply FFT
            print(i, "th fft file is made")
            print("FFT file shape: ", len(buf))
            self.fft.append(buf)
        print(i, "number of FFT file is made")
        return self.fft