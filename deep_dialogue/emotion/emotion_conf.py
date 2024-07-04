
emotion_config = dict()

##################################################
#           Personality-Trigger Matrix           #
# - This is a linear implementation. This matrix #
#   indicates how sensitive the user is for each #
#   trigger event given a certain personality    #
# - The matrix shape is:                         #
#   size(Personality) x size(Trigger)            #
##################################################
emotion_config['PERSONALITY_TRIGGER'] = [
    [0.6, 0.4, 0.5, 0.6, 0.7],
    [0.9, 0.8, 0.8, 0.9, 0.7],
    [0.4, 0.4, 0.5, 0.7, 0.8],
    [0.3, 0.3, 0.4, 0.7, 0.8],
    [0.1, 0.2, 0.1, 0.1, 0.1]
]

##################################################
#             Trigger-Emotion Matrix             #
# - This is a linear implementation. This matrix #
#   indicates how a trigger event alter the user #
#   emotion                                      #
# - The matrix shape is:                         #
#   size(Trigger) x size(Emotion)                #
##################################################
emotion_config['TRIGGER_EMOTION'] = [
    [0.35, 0.05, -0.15, 0.07],
    [0., 0.4, -0.1, 0.05],
    [0.1, 0.15, -0.2, 0.03],
    [-0.05, -0.02, 0.6, -0.1],
    [-0.1, -0.07, 0.5, 0.02]
]

##################################################
#           Personality-Emotion Matrix           #
# - This is a linear implementation. This matrix #
#   indicates how sensitive a user is to his/her #
#   emotion change given a certain personality   #
# - The matrix shape is:                         #
#   size(Personality) x size(Emotion)            #
##################################################
emotion_config['PERSONALITY_EMOTION'] = [
    [0.5, 0.4, 0.4, 0.2],
    [0.5, 0.4, 0.5, 0.6],
    [0.5, 0.6, 0.5, 0.3],
    [0.1, 0.2, 0.7, 0.4],
    [0.6, 0.4, 0.6, 0.1]
]

##################################################
#                 Emotion Decay Rate             #
# - This is a linear implementtion. This vector  #
#   indicates how the emotion is decay in each   #
#   time step.                                   #
# - The vector shape is:                         #
#   size(Emotion)                                #
##################################################
emotion_config['EMOTION_DECAY'] = [-0.01, -0.01, -0.01, -0.1]