from emosim.emocalc.normalize import BaseNormFunc
import numpy as np
from copy import deepcopy

class CustomNormFunc(BaseNormFunc):

    def __init__(self):
        super(CustomNormFunc, self).__init__()
        self.func_name += '->' + '[CustomNormFunc]'


    def norm_emo(self, mood):
        assert type(mood) == type([])

        new_mood = deepcopy(mood)
        for idx, intensity in enumerate(new_mood):
            if intensity > 0:
                new_mood[idx] = np.tanh(intensity)
            else:
                new_mood[idx] = np.tanh(-1 * intensity * np.power(1000, intensity))

        return new_mood


