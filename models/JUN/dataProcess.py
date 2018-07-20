import librosa
import glob


class Sound(object):

    # Constructor
    def __init__(self, path = None, extension = None):
        self.raw = []               # Get sound data
        self.path = path            # Get data path
        self.extension = extension  # Get data extension
        self.dataNames = glob.glob(self.path + "*." + self.extension)  # Make name package
        self.dataNum = self.dataNames.__len__() # Get data number
        print(self.path)        # For debugging
        print(self.extension)   # For debugging

    # Load raw data & attach to one chunk
    def load(self):
        for i in range(self.dataNum):
            buf, self.sr = librosa.load(self.dataNames.pop(0))  # Get raw data and sampling rate
            self.raw.append(buf)    # Collect data as one array
        print(self.dataNames)   # For debugging
        print(self.raw)         # For debugging

    # Preprocess for get feature of data
    # Cut data using sampling rate
    def preProcess(self):
        for i in range(self.dataNum):
            tmp = len(self.raw[i])
            cutting = 0
            while tmp < self.sr:
                tmp -= self.sr
                cutting += self.sr
            self.base.append(self.raw[i][:cutting])



    # process data
    # Method:
    def process(self, method):
        if method == "MFCC" or "mfcc":
            mfcc = librosa.feature.mfcc(self.raw[0], self.sr)

            return
