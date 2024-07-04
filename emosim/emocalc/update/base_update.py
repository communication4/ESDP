import abc

class BaseUpdateFunc(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.func_name = '[BaseUpdateFunc]'

    @abc.abstractmethod
    def update_emo(self, emo, events):
        '''Update the mood state'''
        return