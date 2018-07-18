import librosa


class Sound(object):

    # Constructor
    # Get data path
    def __init__(self, path = None, extension = None):
        self.path = path
        self.extension = extension
        print(self.path)
        print(self.extension)