from emosim.emocalc.update import BaseUpdateFunc
import numpy as np
from copy import deepcopy

class CustomUpdateFunc(BaseUpdateFunc):

    def __init__(self):
        super(CustomUpdateFunc, self).__init__()
        self.func_name += '->' + '[CustomUpdateFunc]'

    def update_emo(self, emo, events):
        raise NotImplementedError('CustomUpdateFunc is not implemented yet.')
