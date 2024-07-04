from emosim.emocalc.normalize import BaseNormFunc
import numpy as np
from copy import deepcopy

class SoftmaxNormFunc(BaseNormFunc):

    def __init__(self, scale_factor=8.0):
        super(SoftmaxNormFunc, self).__init__()
        self.func_name += '->' + '[SoftmaxNormFunc]'

        self.scale_factor = scale_factor

    def softmax(self, x):
        x_exp = np.exp(x)
        s = x_exp / np.sum(x_exp, axis=0, keepdims=True)
        return s.tolist()

    def norm_emo(self, mood):
        assert type(mood) == type([])

        new_mood = np.tanh(np.array(deepcopy(mood))) * self.scale_factor
        new_mood = self.softmax(new_mood)

        return new_mood

