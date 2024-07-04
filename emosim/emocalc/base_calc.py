import abc

class BaseEmoCalculator(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.func_name = '[BaseEmoCalculator]'

    @abc.abstractmethod
    def create_update_func(self):
        '''Create the update function'''
        return

    @abc.abstractmethod
    def create_decay_func(self):
        '''Create the decay function'''
        return

    @abc.abstractmethod
    def create_norm_func(self):
        '''Create the normalization function'''
        return

    @abc.abstractmethod
    def update_mood(self, emo, events):
        '''Core function'''
        return