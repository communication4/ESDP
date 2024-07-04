from deep_dialog.emotion.emotion_conf import emotion_config
import numpy as np

class TriggerDetector(object):

    def __init__(self, turn_threshold=25):
        self.turn_threshold = turn_threshold
        self.TRIGGER_EMOTION = None

        self._initialize_emotion()

    def _initialize_emotion(self):
        self.TRIGGER_EMOTION = emotion_config['TRIGGER_EMOTION']

    def detect(self, state, sys_act, goal, personality, pt_matrix):
        l = [0, 0, 0, 0, 0]
        l = self._detect_longturns(state, sys_act, l)
        l = self._detect_repeated_response(state, sys_act, l)
        l = self._detect_related_response(state, sys_act, goal, l)
        l = self._detect_proactive_response(state, sys_act, goal, l)

        ppt = np.array(personality).dot(np.array(pt_matrix))
        pte = (ppt * l).dot(self.TRIGGER_EMOTION)

        return l, pte

    def _detect_longturns(self, state, sys_act, l):
        turn = state['turn']
        if turn > self.turn_threshold:
            l[0] = 1
            # print('find long turn')
        return l

    def _detect_related_response(self, state, sys_act, goal, l):
        # print(state)
        # print(sys_act)
        # print("HISTORICAL", state['history_slots'])

        if sys_act['diaact'] == 'request':
            if len(sys_act['inform_slots']) > 0:
                raise NotImplementedError('yield a request action but inform something.')

            for r_slot in sys_act['request_slots'].keys():
                if r_slot in state['rest_slots']:
                    l[3] = 1

        if sys_act['diaact'] == 'inform':
            if 'taskcomplete' not in sys_act['inform_slots'].keys():
                for i_slot in sys_act['inform_slots'].keys():
                    if i_slot in sys_act['request_slots'].keys():
                        l[3] = 1

                    if i_slot in goal['request_slots'].keys():
                        l[3] = 1

        if l[3] > 0:
            return l
        else:
            if sys_act['diaact'] == 'request':
                for r_slot in sys_act['request_slots'].keys():
                    if r_slot not in goal['request_slots'].keys() and r_slot not in goal['inform_slots'].keys():
                        l[1] = 1
            if sys_act['diaact'] == 'inform':
                if 'taskcomplete' not in sys_act['inform_slots'].keys():
                    for r_slot in sys_act['inform_slots'].keys():
                        if r_slot not in goal['request_slots'].keys() and r_slot not in goal['inform_slots'].keys():
                            l[1] = 1

            return l

    def _detect_repeated_response(self, state, sys_act, l):

        if sys_act['diaact'] == 'request':
            for r_slot in sys_act['request_slots'].keys():
                if r_slot in state['history_slots'].keys() or r_slot in state['inform_slots'].keys():
                    l[2] = 1

        return l


    def _detect_proactive_response(self, state, sys_act, goal, l):
        if sys_act['diaact'] == 'inform':
            if 'taskcomplete' not in sys_act['inform_slots'].keys():
                for i_slot in sys_act['inform_slots'].keys():
                    if i_slot in goal['request_slots'].keys() and i_slot in state['rest_slots']:
                        l[4] = 1

        if sys_act['diaact'] == 'request':
            for i_slot in sys_act['request_slots'].keys():
                if i_slot in goal['inform_slots'].keys() and i_slot in state['rest_slots']:
                    l[4] = 1

        return l

    def _check_assertion(self):
        raise NotImplementedError('987')
