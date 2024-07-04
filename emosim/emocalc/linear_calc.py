from emosim.emocalc import BaseEmoCalculator
from emosim.emocalc.update import LinearUpdateFunc
from emosim.emocalc.decay import LinearDecayFunc
from emosim.emocalc.normalize import SoftmaxNormFunc

from copy import deepcopy
import numpy as np

class LinearEmoCalculator(BaseEmoCalculator):

    def __init__(self, configs):
        super(LinearEmoCalculator, self).__init__()
        self.func_name += '->' + '[LinearEmoCalculator]'

        self.update_func = None
        self.decay_func = None
        self.norm_func = None

        # parameters
        self.alpha = configs['alpha']
        self.decay_factor = configs['decay_factor']
        self.scale_factor = configs['scale_factor']
        self.pt_matrix = configs['pt_matrix']
        self.te_matrix = configs['te_matrix']
        self.pe_matrix = configs['pe_matrix']

        # initialize
        self.create_update_func()
        self.create_decay_func()
        self.create_norm_func()

    def create_update_func(self):
        self.update_func = LinearUpdateFunc(self.pt_matrix, self.te_matrix, self.pe_matrix, self.alpha)

    def create_decay_func(self):
        self.decay_func = LinearDecayFunc(self.decay_factor)

    def create_norm_func(self):
        self.norm_func = SoftmaxNormFunc(self.scale_factor)

    def update_mood(self, emo, events):

        update_momentum = self.update_func.update_emo(emo, events)
        decay_momentum = self.decay_func.decay_emo(emo.mood)
        new_mood = deepcopy(emo.mood)
        new_mood = np.sum([update_momentum, decay_momentum, np.array(new_mood)], axis=0).tolist()
        new_mood = self.norm_func.norm_emo(new_mood)

        return new_mood
