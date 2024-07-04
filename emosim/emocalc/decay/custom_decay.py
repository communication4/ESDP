from emosim.emocalc.decay import BaseDecayFunc
from copy import deepcopy

class CustomDecayFunc(BaseDecayFunc):

    def __init__(self):
        super(CustomDecayFunc, self).__init__()
        self.func_name += '->' + '[CustomDecayFunc]'

    def decay_emo(self, mood):
        assert type(mood) == type([])

        raise NotImplementedError('CustomDecayFunc is not implemented yet.')
