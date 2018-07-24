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
            cutting = 0
            while tmp > self.sr:
                tmp -= self.sr
                cutting += self.sr
            self.cutSound.append(self.raw[i][:cutting])
            self.mfcc.append(librosa.feature.mfcc(self.cutSound[i]))
            print(i, "Cut sound is made")
            print("cutSound[", i, "]shape: ", self.cutSound[i].shape)
            print(i, " MFCC is made")
            print("mfcc[", i, "] shape: ", self.mfcc[i].shape)

    # Preprocess for get feature of data
    def preProcess(self):
        self.mfcc = [] # Feature data

    # process data
    # Method:
    def process(self, method):
        if method == "MFCC" or "mfcc":
            mfcc = librosa.feature.mfcc(self.raw[0], self.sr)
            return mfcc
        else:
            return None
