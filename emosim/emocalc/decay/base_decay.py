import abc

class BaseDecayFunc(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.func_name = '[BaseDecayFunc]'

    @abc.abstractmethod
    def decay_emo(self, mood):
        '''Decay the mood state '''
        return