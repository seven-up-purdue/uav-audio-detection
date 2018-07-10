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
        self.WAV_PATH = config['WAV_PATH']
        self.MODEL_TYPE = config['MODEL_TYPE']
        self.MODEL_PATH = config['MODEL_PATH']
        self.FEATURE_TYPE = config['FEATURE_TYPE']
        self.SERVER_ADDRESS = config['SERVER_ADDRESS']
        self.BUFFER_HOURS = config['BUFFER_HOURS']

