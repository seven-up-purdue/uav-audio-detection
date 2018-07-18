import librosa
import glob


class Sound(object):

    raw = []

    # Constructor
    def __init__(self, path = None, extension = None):
        self.path = path            # Get data path
        self.extension = extension  # Get data extension
        print(self.path)        # For debugging
        print(self.extension)   # For debugging

    # Load raw data & attach to one chunk
    def load(self):
        self.dataNames = glob.glob(self.path + "*." + self.extension)
        for i in range(self.dataNames.__len__()):
            self.raw.append(librosa.load(self.dataNames.pop(0)))
        print(self.dataNames)   # For debugging
        print(self.raw)         # For debugging
        