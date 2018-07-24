import pyloudness as ld
import numpy as np
import librosa
import glob
#import matplotlib.pyplot as plt

class Sound(object):

    # Constructor
    def __init__(self, path = None, extension = None):
        self.raw = []               # Get sound data
        self.path = path            # Get data path
        self.extension = extension  # Get data extension
        self.dataNames = glob.glob(self.path + "*." + self.extension)  # Make name package
        self.dataNum = self.dataNames.__len__() # Get data number
        print("Data path: ", self.path)        # For debugging
        print("Extension: ", self.extension)   # For debugging
        print("Data number: ", self.dataNum)   # For debugging

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
        # Data Normalization
        self.cutSound = librosa.util.normalize(self.cutSound)

    # Preprocess for get feature of data
    def preProcess(self):
        self.mfcc = [[]] * 20 # Feature data
        mask = int(self.sr / 5)
        print("Mask: ", mask)
        for i in range(0, self.cutting + 1 - mask, 2205): # self.sr/5 => 1/5 sec
            # Cut as
            buf = self.cutSound[i:i + mask]
            buf = librosa.feature.mfcc(buf)
            self.mfcc = np.hstack((self.mfcc, buf))
            print(i, " MFCC is made")
            print("mfcc[", i, "] shape: ", self.mfcc.shape)
        print(i, " MFCC is made")


    # process data
    # Method:
    def process(self, method):
        if method == "MFCC" or "mfcc":
            mfcc = librosa.feature.mfcc(self.raw[0], self.sr)
            return mfcc
        else:
            return None
