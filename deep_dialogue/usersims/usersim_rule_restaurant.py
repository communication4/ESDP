"""
Created on May 14, 2016

a rule-based user simulator

-- user_goals_first.v2.p: all goals
-- user_goals_first.v1.p: all goals with duplicates

@author: xiul, t-zalipt
"""

from .usersim import UserSimulator
import argparse, json, random, copy

from deep_dialog import dialog_config
from deep_dialog.emotion import TriggerDetector
import numpy as np
# import pdb


class RuleRestSimulator(UserSimulator):
    """ A rule-based user simulator for testing dialog policy """

    def __init__(self, rest_dict=None, act_set=None, slot_set=None, start_set=None, personality_set=None,
                 emotion_set=None, params=None):
        """ Constructor shared by all user simulators """

        super(RuleRestSimulator, self).__init__()

        self.rest_dict = rest_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.start_set = start_set

        self.max_turn = params['max_turn']
        self.slot_err_probability = params['slot_err_probability']
        self.slot_err_mode = params['slot_err_mode']
        self.intent_err_probability = params['intent_err_probability']

        self.simulator_run_mode = params['simulator_run_mode']
        self.simulator_act_level = params['simulator_act_level']

        self.learning_phase = params['learning_phase']

        self.personality_set = personality_set
        self.emotion_set = emotion_set
        self.trigger_detector = TriggerDetector(turn_threshold=35)

        self.params = params

    def initialize_episode(self, goal_idx=None):
        """ Initialize a new episode (dialog)
        state['history_slots']: keeps all the informed_slots
        state['rest_slots']: keep all the slots (which is still in the stack yet)
        """

        self.state = {}
        self.state['history_slots'] = {}
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['rest_slots'] = []
        self.state['turn'] = 0

        self.episode_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET

        self.personality = self._sample_personality()
        self.state['emotion'] = self._sample_emotion()
        self.state['historical_emotion'] = []

        self.state['trigger_counter'] = [0, 0, 0, 0, 0]

        # self.goal =  random.choice(self.start_set)
        if goal_idx:
            self.goal = self._fetch_goal(self.start_set, goal_idx)
        else:
            self.goal = self._sample_goal(self.start_set)
        # self.goal['request_slots'] = {}
        self.goal['request_slots']['Name'] = 'UNK'
        self.constraint_check = dialog_config.CONSTRAINT_CHECK_FAILURE

        if 'greeting' in list(self.goal['inform_slots'].keys()):
            del self.goal['inform_slots']['greeting']

        """ Debug: build a fake goal mannually """
        # import pdb
        # pdb.set_trace()
        # self.debug_falk_goal()

        # sample first action
        user_action = self._sample_action()
        assert (self.episode_over != 1), ' but we just started'
        return user_action

    def _sample_action(self):
        """ randomly sample a start action based on user goal """

        self.state['diaact'] = random.choice(list(dialog_config.start_dia_acts.keys()))

        # "sample" informed slots
        if len(self.goal['inform_slots']) > 0:
            known_slot = random.choice(list(self.goal['inform_slots'].keys()))
            self.state['inform_slots'][known_slot] = self.goal['inform_slots'][known_slot]

            if 'Name' in list(self.goal['inform_slots'].keys()):  # 'taxiname' must appear in the first user turn
                self.state['inform_slots']['Name'] = self.goal['inform_slots']['Name']

            # if 'greeting' in self.goal['inform_slots'].keys():
            #     self.state['inform_slots']['greeting'] = self.goal['inform_slots']['greeting']

            for slot in list(self.goal['inform_slots'].keys()):
                if known_slot == slot or slot == 'Name' or slot == 'greeting': continue
                self.state['rest_slots'].append(slot)

        self.state['rest_slots'].extend(self.goal['request_slots'].keys())

        # "sample" a requested slot
        request_slot_set = list(self.goal['request_slots'].keys())
        request_slot_set.remove('Name')
        if len(request_slot_set) > 0:
            request_slot = random.choice(request_slot_set)
        else:
            request_slot = 'Name'
        self.state['request_slots'][request_slot] = 'UNK'

        if len(self.state['request_slots']) == 0:
            self.state['diaact'] = 'inform'

        if (self.state['diaact'] in ['thanks', 'closing']):
            self.episode_over = True  # episode_over = True
        else:
            self.episode_over = False  # episode_over = False

        sample_action = {}
        sample_action['diaact'] = self.state['diaact']
        sample_action['inform_slots'] = self.state['inform_slots']
        sample_action['request_slots'] = self.state['request_slots']
        sample_action['turn'] = self.state['turn']
        sample_action['nl'] = "hello"

        self.add_nl_to_action(sample_action)
        return sample_action

    def _sample_goal(self, goal_set):
        """ sample a user goal  """

        self.sample_goal = random.choice(self.start_set[self.learning_phase])
        # print(self.sample_goal)
        return self.sample_goal

        # sample_goal = random.choice(self.start_set[self.learning_phase])
        # return sample_goal

    def _fetch_goal(self, goal_set, goal_idx):
        self.sample_goal = self.start_set[self.learning_phase][goal_idx]
        return self.sample_goal

    def corrupt(self, user_action):
        """ Randomly corrupt an action with error probs (slot_err_probability and slot_err_mode) on Slot and Intent (intent_err_probability). """

        for slot in user_action['inform_slots'].keys():
            slot_err_prob_sample = random.random()
            if slot_err_prob_sample < self.slot_err_probability:  # add noise for slot level
                if self.slot_err_mode == 0:  # replace the slot_value only
                    if slot in self.rest_dict.keys(): user_action['inform_slots'][slot] = random.choice(
                        self.rest_dict[slot])
                elif self.slot_err_mode == 1:  # combined
                    slot_err_random = random.random()
                    if slot_err_random <= 0.33:
                        if slot in self.rest_dict.keys(): user_action['inform_slots'][slot] = random.choice(
                            self.rest_dict[slot])
                    elif slot_err_random > 0.33 and slot_err_random <= 0.66:
                        del user_action['inform_slots'][slot]
                        random_slot = random.choice(self.rest_dict.keys())
                        user_action[random_slot] = random.choice(self.rest_dict[random_slot])
                    else:
                        del user_action['inform_slots'][slot]
                elif self.slot_err_mode == 2:  # replace slot and its values
                    del user_action['inform_slots'][slot]
                    random_slot = random.choice(self.rest_dict.keys())
                    user_action[random_slot] = random.choice(self.rest_dict[random_slot])
                elif self.slot_err_mode == 3:  # delete the slot
                    del user_action['inform_slots'][slot]

        intent_err_sample = random.random()
        if intent_err_sample < self.intent_err_probability:  # add noise for intent level
            user_action['diaact'] = random.choice(self.act_set.keys())

    def debug_falk_goal(self):
        """ Debug function: build a fake goal mannually (Can be moved in future) """

        self.goal['inform_slots'].clear()
        self.goal['inform_slots']['Food'] = 'indian'
        self.goal['inform_slots']['People'] = '10'
        # self.goal['inform_slots']['theater'] = 'amc pacific place 11 theater'
        # self.goal['inform_slots']['starttime'] = '10:00 pm'
        # self.goal['inform_slots']['date'] = 'tomorrow'
        self.goal['inform_slots']['Area'] = 'west side of town'
        self.goal['inform_slots']['Time'] = '11:15'
        self.goal['inform_slots']['Price'] = 'cheaper'
        self.goal['inform_slots']['Day'] = 'wednesday'

        self.goal['request_slots'].clear()
        self.goal['request_slots']['Name'] = 'UNK'
        self.goal['request_slots']['Name'] = 'UNK'
        self.goal['request_slots']['Addr'] = 'UNK'
        self.goal['request_slots']['Post'] = 'UNK'

    def next(self, system_action):
        """ Generate next User Action based on last System Action """

        self.state['turn'] += 2
        self.episode_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET

        sys_act = system_action['diaact']

        self._update_user_emotion_dataset(system_action)

        if (self.max_turn > 0 and self.state['turn'] > self.max_turn):
            self.dialog_status = dialog_config.FAILED_DIALOG
            # print("FAILED Because TOO LONG")
            self.episode_over = True
            self.state['request_slots'].clear()
            self.state['inform_slots'].clear()
            self.state['diaact'] = "closing"
        else:
            if self.params['user_abort'] and \
                    (self.state['emotion'][0] > self.params['abort_threhold'] or \
                     self.state['emotion'][1] > self.params['abort_threhold']) and \
                    np.random.rand() < self.params['abort_rate']:
                self.dialog_status = dialog_config.FAILED_DIALOG
                self.episode_over = True
                self.state['request_slots'].clear()
                self.state['inform_slots'].clear()
                self.state['diaact'] = "closing"
            else:
                self.state['history_slots'].update(self.state['inform_slots'])
                self.state['inform_slots'].clear()

                if sys_act == "inform":
                    self.response_inform(system_action)
                elif sys_act == "multiple_choice":
                    self.response_multiple_choice(system_action)
                elif sys_act == "request":
                    self.response_request(system_action)
                elif sys_act == "thanks":
                    self.response_thanks(system_action)
                elif sys_act == "confirm_answer":
                    self.response_confirm_answer(system_action)
                elif sys_act == "closing":
                    self.episode_over = True
                    self.state['diaact'] = "thanks"
                    self.state['request_slots'].clear()

        if self.state['diaact'] == "thanks":
            self.state['request_slots'].clear()
            self.state['inform_slots'].clear()

        self.corrupt(self.state)

        response_action = {}
        response_action['diaact'] = self.state['diaact']
        response_action['inform_slots'] = self.state['inform_slots']
        response_action['request_slots'] = self.state['request_slots']
        response_action['turn'] = self.state['turn']
        response_action['nl'] = ""

        # add NL to dia_act
        # pdb.set_trace()
        self.add_nl_to_action(response_action)
        return response_action, self.episode_over, self.dialog_status

    def response_confirm_answer(self, system_action):
        """ Response for Confirm_Answer (System Action) """

        if len(self.state['rest_slots']) > 0:
            request_slot = random.choice(self.state['rest_slots'])

            if request_slot in self.goal['request_slots'].keys():
                self.state['request_slots'].clear()
                self.state['diaact'] = "request"
                self.state['request_slots'][request_slot] = "UNK"
            elif request_slot in list(self.goal['inform_slots'].keys()):
                self.state['diaact'] = "inform"
                self.state['inform_slots'][request_slot] = self.goal['inform_slots'][request_slot]
                self.state['request_slots'].clear()
                if request_slot in self.state['rest_slots']:
                    self.state['rest_slots'].remove(request_slot)
        else:
            self.state['diaact'] = "thanks"
            self.state['request_slots'].clear()

    def response_thanks(self, system_action):
        """ Response for Thanks (System Action) """

        self.episode_over = True
        self.dialog_status = dialog_config.SUCCESS_DIALOG

        request_slot_set = copy.deepcopy(list(self.state['request_slots'].keys()))
        if 'Name' in request_slot_set:
            request_slot_set.remove('Name')
        if 'Ref' in request_slot_set:
            request_slot_set.remove('Ref')
        rest_slot_set = copy.deepcopy(self.state['rest_slots'])
        if 'Name' in rest_slot_set:
            rest_slot_set.remove('Name')
        if 'Ref' in rest_slot_set:
            rest_slot_set.remove('Ref')

        if len(request_slot_set) > 0 or len(rest_slot_set) > 0:
            self.dialog_status = dialog_config.FAILED_DIALOG
            print("FAILED Because len(request_slot_set) = %s and len(rest_slot_set) = %s, and %s" % (len(request_slot_set),
                                                                                             len(rest_slot_set), rest_slot_set))

        for info_slot in self.state['history_slots'].keys():
            if self.state['history_slots'][info_slot] == dialog_config.NO_VALUE_MATCH:
                # pdb.set_trace()
                self.dialog_status = dialog_config.FAILED_DIALOG
                print("FAILED Because NO MATCH1")
            if info_slot in list(self.goal['inform_slots'].keys()):
                if self.state['history_slots'][info_slot] != self.goal['inform_slots'][info_slot]:
                    self.dialog_status = dialog_config.FAILED_DIALOG
                    print("FAILED Because THE INFORMATION ARE CHANGED")

        if 'Name' in system_action['inform_slots'].keys():
            if system_action['inform_slots']['Name'] == dialog_config.NO_VALUE_MATCH:
                self.dialog_status = dialog_config.FAILED_DIALOG
                print("FAILED Because NO MATCH2")

        if self.constraint_check == dialog_config.CONSTRAINT_CHECK_FAILURE:
            print("FAILED Because CONSTRAINT_CHECK_FAILURE")
            self.dialog_status = dialog_config.FAILED_DIALOG

    def response_request(self, system_action):
        """ Response for Request (System Action) """
        # pdb.set_trace()
        if len(system_action['request_slots'].keys()) > 0:
            slot = list(system_action['request_slots'].keys())[0]  # only one slot
            if slot in self.goal[
                'inform_slots'].keys():  # request slot in user's constraints  #and slot not in self.state['request_slots'].keys():
                self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
                self.state['diaact'] = "inform"
                if slot in self.state['rest_slots']: self.state['rest_slots'].remove(slot)
                if slot in self.state['request_slots'].keys(): del self.state['request_slots'][slot]
                self.state['request_slots'].clear()
            elif slot in self.goal['request_slots'].keys() and slot not in self.state['rest_slots'] and slot in \
                    self.state['history_slots'].keys():  # the requested slot has been answered
                self.state['inform_slots'][slot] = self.state['history_slots'][slot]
                self.state['request_slots'].clear()
                self.state['diaact'] = "inform"
            elif slot in self.goal['request_slots'].keys() and slot in self.state[
                'rest_slots']:  # request slot in user's goal's request slots, and not answered yet
                # self.state['request_slots'].clear()
                # self.state['diaact'] = "request" # "confirm_question"
                # self.state['request_slots'][slot] = "UNK"

                self.state['request_slots'].clear()
                flag = False
                for info_slot in self.state['rest_slots']:
                    if info_slot in list(self.goal['inform_slots'].keys()):
                        self.state['inform_slots'][info_slot] = self.goal['inform_slots'][info_slot]
                        self.state['rest_slots'].remove(info_slot)
                        self.state['diaact'] = "inform"  # "confirm_question"
                        flag = True
                        break
                if flag == False:
                    self.state['diaact'] = "request"  # "confirm_question"
                    self.state['request_slots'][slot] = "UNK"

                ########################################################################
                # Inform the rest of informable slots
                ########################################################################
                # for info_slot in self.state['rest_slots']:
                #     if info_slot in self.goal['inform_slots'].keys():
                #         self.state['inform_slots'][info_slot] = self.goal['inform_slots'][info_slot]
                #
                # for info_slot in self.state['inform_slots'].keys():
                #     if info_slot in self.state['rest_slots']:
                #         self.state['rest_slots'].remove(info_slot)
            else:
                if len(self.state['request_slots']) == 0 and len(self.state['rest_slots']) == 0:
                    self.state['diaact'] = "thanks"
                else:
                    self.state['diaact'] = "inform"
                self.state['inform_slots'][slot] = dialog_config.I_DO_NOT_CARE
                self.state['request_slots'].clear()
        else:  # this case should not appear
            if len(self.state['rest_slots']) > 0:
                random_slot = random.choice(self.state['rest_slots'])
                if random_slot in list(self.goal['inform_slots'].keys()):
                    self.state['inform_slots'][random_slot] = self.goal['inform_slots'][random_slot]
                    self.state['rest_slots'].remove(random_slot)
                    self.state['diaact'] = "inform"
                elif random_slot in self.goal['request_slots'].keys():
                    self.state['request_slots'][random_slot] = self.goal['request_slots'][random_slot]
                    self.state['diaact'] = "request"

    def response_multiple_choice(self, system_action):
        """ Response for Multiple_Choice (System Action) """

        slot = system_action['inform_slots'].keys()[0]
        if slot in list(self.goal['inform_slots'].keys()):
            self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
        elif slot in self.goal['request_slots'].keys():
            self.state['inform_slots'][slot] = random.choice(system_action['inform_slots'][slot])

        self.state['diaact'] = "inform"
        if slot in self.state['rest_slots']: self.state['rest_slots'].remove(slot)
        if slot in self.state['request_slots'].keys(): del self.state['request_slots'][slot]

    def response_inform(self, system_action):
        """ Response for Inform (System Action) """

        if 'taskcomplete' in system_action[
            'inform_slots'].keys():  # check all the constraints from agents with user goal
            self.state['diaact'] = "thanks"
            # if 'taxi' in self.state['rest_slots']: self.state['request_slots']['taxi'] = 'UNK'
            self.constraint_check = dialog_config.CONSTRAINT_CHECK_SUCCESS

            if system_action['inform_slots']['taskcomplete'] == dialog_config.NO_VALUE_MATCH:
                self.state['history_slots']['Name'] = dialog_config.NO_VALUE_MATCH
                if 'Name' in self.state['rest_slots']: self.state['rest_slots'].remove('Name')
                if 'Name' in self.state['request_slots'].keys(): del self.state['request_slots']['Name']

                self.state['request_slots'].clear()

            # if len(self.state['request_slots']) > 0: self.state['request_slots'].clear() # add clear rule [2018-07-20]

            for slot in list(self.goal['inform_slots'].keys()):
                #  Deny, if the answers from agent can not meet the constraints of user
                if slot not in system_action['inform_slots'].keys() or (
                        self.goal['inform_slots'][slot].lower() != system_action['inform_slots'][slot].lower()):
                    self.state['diaact'] = "deny"
                    self.state['request_slots'].clear()
                    self.state['inform_slots'].clear()
                    self.constraint_check = dialog_config.CONSTRAINT_CHECK_FAILURE
                    break

            self.state['request_slots'].clear()
        else:
            for slot in system_action['inform_slots'].keys():
                self.state['history_slots'][slot] = system_action['inform_slots'][slot]

                if slot in list(self.goal['inform_slots'].keys()):
                    if system_action['inform_slots'][slot] == self.goal['inform_slots'][slot]:
                        if slot in self.state['rest_slots']: self.state['rest_slots'].remove(slot)

                        if len(self.state['request_slots']) > 0:
                            self.state['diaact'] = "request"
                        elif len(self.state['rest_slots']) > 0:
                            rest_slot_set = copy.deepcopy(self.state['rest_slots'])
                            if 'Name' in rest_slot_set:
                                rest_slot_set.remove('Name')
                            if 'Ref' in rest_slot_set:
                                rest_slot_set.remove('Ref')

                            if len(rest_slot_set) > 0:
                                inform_slot = random.choice(rest_slot_set)  # self.state['rest_slots']
                                if inform_slot in list(self.goal['inform_slots'].keys()):
                                    self.state['inform_slots'][inform_slot] = self.goal['inform_slots'][inform_slot]
                                    self.state['diaact'] = "inform"
                                    self.state['rest_slots'].remove(inform_slot)
                                elif inform_slot in self.goal['request_slots'].keys():
                                    self.state['request_slots'][inform_slot] = 'UNK'
                                    self.state['diaact'] = "request"
                            else:
                                self.state['request_slots']['Name'] = 'UNK'
                                self.state['diaact'] = "request"
                        else:  # how to reply here?
                            self.state['diaact'] = "thanks"  # replies "closing"? or replies "confirm_answer"
                            self.state['request_slots'].clear()
                    else:  # != value  Should we deny here or ?
                        ########################################################################
                        # TODO When agent informs(slot=value), where the value is different with the constraint in user goal, Should we deny or just inform the correct value?
                        ########################################################################
                        self.state['diaact'] = "inform"
                        self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
                        if slot in self.state['rest_slots']: self.state['rest_slots'].remove(slot)
                        self.state['request_slots'].clear()
                else:
                    if slot in self.state['rest_slots']:
                        self.state['rest_slots'].remove(slot)
                    if slot in self.state['request_slots'].keys():
                        del self.state['request_slots'][slot]

                    if len(self.state['request_slots']) > 0:
                        request_set = list(self.state['request_slots'].keys())
                        if 'Name' in request_set:
                            request_set.remove('Name')

                        if len(request_set) > 0:
                            request_slot = random.choice(request_set)
                        else:
                            request_slot = 'Name'

                        self.state['request_slots'][request_slot] = "UNK"
                        self.state['diaact'] = "request"
                    elif len(self.state['rest_slots']) > 0:
                        rest_slot_set = copy.deepcopy(self.state['rest_slots'])
                        if 'Name' in rest_slot_set:
                            rest_slot_set.remove('Name')
                        if 'Ref' in rest_slot_set:
                            rest_slot_set.remove('Ref')
                        if len(rest_slot_set) > 0:
                            inform_slot = random.choice(rest_slot_set)  # self.state['rest_slots']
                            if inform_slot in list(self.goal['inform_slots'].keys()):
                                self.state['inform_slots'][inform_slot] = self.goal['inform_slots'][inform_slot]
                                self.state['diaact'] = "inform"
                                self.state['rest_slots'].remove(inform_slot)

                                # if 'taxi' in self.state['rest_slots']:
                                #     self.state['request_slots']['taxi'] = 'UNK'
                                #     self.state['diaact'] = "request"
                            elif inform_slot in self.goal['request_slots'].keys():
                                self.state['request_slots'][inform_slot] = self.goal['request_slots'][inform_slot]
                                self.state['diaact'] = "request"
                        else:
                            self.state['request_slots']['Name'] = 'UNK'
                            self.state['diaact'] = "request"
                    else:
                        self.state['diaact'] = "thanks"  # or replies "confirm_answer"
                        self.state['request_slots'].clear()


def main(params):
    user_sim = RuleRestSimulator()
    user_sim.initialize_episode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    params = vars(args)

    print("User Simulator Parameters:")
    print(json.dumps(params, indent=2))

    main(params)
