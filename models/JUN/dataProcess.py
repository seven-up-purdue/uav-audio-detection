import librosa


class Sound(object):

    # Constructor
    # Get data path
    def __init__(self, path, extension):
        self.path = path
        self.extension = extension
        print(self.path)