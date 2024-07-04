import abc

class BaseNormFunc(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.func_name = '[BaseNormFunc]'

    @abc.abstractmethod
    def norm_emo(self, mood):
        '''Normalize the mood state '''
        return