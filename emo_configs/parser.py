import json
import codecs

class EmoConfig(object):

    def __init__(self, config_path='./emosim_configs.json'):
        self.config_path = config_path

        self.configs = json.load(codecs.open(self.config_path, 'r', encoding='utf8'))

    def get_configs(self):
        return self.configs