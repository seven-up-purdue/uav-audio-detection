import yaml
import os
class Config(object):
    """
    config yaml file
    """
    # WAV_PATH =
    # name
    # which model
    # which feature

    def __init__(self, yaml_path):
        self.yaml_path = yaml_path
        print(yaml_path)
        assert os.path.isfile(yaml_path), 'yaml file does not exist'
        self.load()

    def load(self):
        with open(self.yaml_path, 'r') as f:
            config = yaml.load(f)
        self.NAME = config['NAME']
        self.DEVICE_NAME = config['DEVICE_NAME']
        self.WAV_PATH = self.wav_path(config['WAV_PATH'])
        self.MODEL_TYPE = config['MODEL_TYPE']
        self.MODEL_PATH = config['MODEL_PATH']
        self.FEATURE_TYPE = config['FEATURE_TYPE']
        self.SERVER_ADDRESS = config['SERVER_ADDRESS']
        self.BUFFER_HOURS = config['BUFFER_HOURS']

        # self.N_MFCC = config['N_MFCC']
        # self.N_FRAME = config['N_FRAME']
        # self.N_CHANNELS = config['N_CHANNELS']

    def wav_path(self, path):
        import time
        t = int(time.time())
        return path+self.NAME+'_%s.wav' % t
