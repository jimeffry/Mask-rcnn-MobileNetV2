# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/10/16 10:09
#project: Face recognize
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  papers:
####################################################
import numpy as np

def set_threshold(score):
    '''
    input: the range of score is [0,100]
    output:distance_threshold, confidence_threshold,
    '''
    confidence_base = 0.04
    distance_base = 1.6
    mult_bin = 0.003
    global confidence_threshold, distance_threshold
    if score > 90:
        confidence_threshold = mult_bin * score
    else:
        confidence_threshold = confidence_base + score/100

    distance_threshold = distance_base - score/100

    return confidence_threshold, distance_threshold

def similarity_score(confidence):
    '''
    input: confidence distance
    output: similarity score
    '''
    confidence_max = 0.4
    if confidence >= confidence_threshold:
        score = np.exp(confidence) / np.exp(confidence_max) * 100
    else:
        score = (np.exp(confidence) -1) / np.exp(confidence_max) * 100
    
    return score


