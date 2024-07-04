"""
Created on May 18, 2016
Modified on Aug 24, 2020

@author: xiul, t-zalipt
@editor: rzhang
"""
import json

def text_to_dict(path):
    """ Read in a text file as a dictionary where keys are text and values are indices (line numbers) """
    
    slot_set = {}
    with open(path, 'r') as f:
        index = 0
        for line in f.readlines():
            slot_set[line.strip('\n').strip('\r')] = index
            index += 1
    return slot_set

def json_to_dict(path):

    slot_set = json.load(open(path, 'r'))

    return slot_set
