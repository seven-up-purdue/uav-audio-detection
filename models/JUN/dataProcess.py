import librosa


class Sound(object):

    # Constructor
    # Get data path
    def __init__(self, path):
        self.path = path
        print(self.path)