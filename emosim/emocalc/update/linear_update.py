from emosim.emocalc.update import BaseUpdateFunc
import numpy as np
from copy import deepcopy

class LinearUpdateFunc(BaseUpdateFunc):

    def __init__(self, pt_matrix, te_matrix, pe_matrix, alpha=1.0):
        super(LinearUpdateFunc, self).__init__()
        self.func_name += '->' + '[LinearUpdateFunc]'

        # M_pt: the Personality-Trigger Matrix
        self.pt_matrix = pt_matrix


        # M_te: the Trigger-Emotion Matrix
        self.te_matrix = te_matrix


        # M_pe: the Personality-Emotion Matrix
        self.pe_matrix = pe_matrix


        self.alpha = alpha

    def update_emo(self, emo, events):

        personality = emo.personality

        ppt = np.array(personality).dot(np.array(self.pt_matrix))
        pte = (ppt * events).dot(self.te_matrix)
        ppe = np.array(personality).dot(np.array(self.pe_matrix)) * self.alpha

        update_momentum = ppe * pte

        return update_momentum.tolist()

