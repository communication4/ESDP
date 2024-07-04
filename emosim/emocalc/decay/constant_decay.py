from emosim.emocalc.decay import BaseDecayFunc
from copy import deepcopy

class ConstantDecayFunc(BaseDecayFunc):

    def __init__(self, decay_step=0.01):
        super(ConstantDecayFunc, self).__init__()
        self.func_name += '->' + '[ConstantDecayFunc]'

        self.decay_step = decay_step
        assert self.decay_step > 0
        assert self.decay_step < 1

    def decay_emo(self, mood):
        assert type(mood) == type([])

        decay_momentum = [self.decay_step] * len(mood)

        return decay_momentum
