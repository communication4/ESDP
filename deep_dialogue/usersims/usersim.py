"""
Created on June 7, 2016

a rule-based user simulator

@author: xiul, t-zalipt
"""

import random
import numpy as np
from deep_dialog.emotion.emotion_conf import emotion_config

class UserSimulator(object):
    """ Parent class for all user sims to inherit from """

    def __init__(self):
        """ Constructor shared by all user simulators """

        self.PERSONALITY_TRIGGER = None
        self.TRIGGER_EMOTION = None
        self.PERSONALITY_EMOTION = None

        self.personality_set = None
        self.emotion_set = None
        self.trigger_detector = None

        self._initialize_emotion()

    def _initialize_emotion(self):
        self.PERSONALITY_TRIGGER = emotion_config['PERSONALITY_TRIGGER']
        self.PERSONALITY_EMOTION = emotion_config['PERSONALITY_EMOTION']
        self.EMOTION_DECAY = emotion_config['EMOTION_DECAY']

    def initialize_episode(self):
        """ Initialize a new episode (dialog)"""
        pass

    def next(self, system_action):
        pass

    def _sample_personality(self):
        sampled_personality = random.choice(self.personality_set)
        return sampled_personality

    def _sample_emotion(self):
        sampled_emotion = [0.8, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0]
        return sampled_emotion

    def _update_user_emotion(self, sys_act):

        trigger, pte = self.trigger_detector.detect(self.state, sys_act, self.goal, self.personality, self.PERSONALITY_TRIGGER)

        self.state['trigger_counter'][0] += trigger[0]
        self.state['trigger_counter'][1] += trigger[1]
        self.state['trigger_counter'][2] += trigger[2]
        self.state['trigger_counter'][3] += trigger[3]
        self.state['trigger_counter'][4] += trigger[4]

        um = self._update_momentum(pte)

        dm = self._decay_momentum(self.state['emotion'])

        self.state['historical_emotion'].append(self.state['emotion'])

        new_emo = np.sum([um, dm, np.array(self.state['emotion'])], axis=0)

        new_emo = self.normalize(new_emo.tolist())

        self.state['emotion'] = new_emo

    def _update_user_emotion_dataset(self, sys_act):
        self.state['historical_emotion'].append(self.state['emotion'])
        if sys_act['diaact'] == 'request':
            for r_slot in sys_act['request_slots'].keys():
                if r_slot in self.state['history_slots'].keys() or r_slot in self.state['inform_slots'].keys():
                    self.state['emotion'] = [0.2, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0]
                if r_slot not in self.goal['request_slots'].keys() and r_slot not in self.goal['inform_slots'].keys():
                    self.state['emotion'] = [0.2, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0]

        if sys_act['diaact'] == 'inform':
            if 'taskcomplete' not in sys_act['inform_slots'].keys():
                for r_slot in sys_act['inform_slots'].keys():
                    if r_slot not in self.goal['request_slots'].keys() and r_slot not in self.goal['inform_slots'].keys():
                        self.state['emotion'] = [0.2, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0]

        if sys_act['diaact'] == 'thanks':
            self.state['emotion'] = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8]

        else:
            self.state['emotion'] = [0.8, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0]

    def normalize(self, emo_list):
        norm_emo = []

        for item in emo_list:
            if item > 0:
                norm_emo.append(np.tanh(item))
            else:
                norm_emo.append(np.tanh(-1*item*np.power(1000, item)))

        return norm_emo


    def _update_momentum(self, pte, alpha=1):

        ppe = np.array(self.personality).dot(np.array(self.PERSONALITY_EMOTION))*alpha
        um = ppe * pte

        return um

    def _decay_momentum(self, emo_state):
        dm = np.array(self.EMOTION_DECAY) * emo_state
        return dm
    
    
    def set_nlg_model(self, nlg_model):
        self.nlg_model = nlg_model  
    
    def set_nlu_model(self, nlu_model):
        self.nlu_model = nlu_model
    
    
    
    def add_nl_to_action(self, user_action):
        """ Add NL to User Dia_Act """
        
        # user_nlg_sentence = self.nlg_model.convert_diaact_to_nl(user_action, 'usr')
        user_action['nl'] = ''	# user_nlg_sentence
        
        if self.simulator_act_level == 1:            
            user_nlu_res = self.nlu_model.generate_dia_act(user_action['nl']) # NLU
            if user_nlu_res != None:
                #user_nlu_res['diaact'] = user_action['diaact'] # or not?
                user_action.update(user_nlu_res)
