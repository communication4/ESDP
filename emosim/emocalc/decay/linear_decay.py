from emosim.emocalc.decay import BaseDecayFunc
from copy import deepcopy

class LinearDecayFunc(BaseDecayFunc):

    def __init__(self, decay_factor=[-0.01, -0.01, -0.01, -0.1]):
        super(LinearDecayFunc, self).__init__()
        self.func_name += '->' + '[LinearDecayFunc]'

        self.decay_factor = decay_factor

    def decay_emo(self, mood):
        assert type(mood) == type([])
        decay_momentum = deepcopy(mood)
        for idx, (md, factor) in enumerate(zip(mood, self.decay_factor)):
            decay_momentum[idx] = md * factor
        return decay_momentum
