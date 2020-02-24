import numpy as np
from collections import Counter

def get_english_alphabet(file='txt/alphabet_english.txt'):
    '''
    Parameters
    __________
    file - dtype str, path to location of english alphabet txt file

    Returns
    __________
    Numpy array containing all letters of the alphabet

    '''
    return np.loadtxt(file, dtype='str')  ##load text file containing english alphabet

def charCount(word):
    dict = {}
    for i in word:
        dict[i] = dict.get(i, 0) + 1
    return dict

def possible_words(lwords, charSet):
    pos_words = []
    for word in lwords:
        flag = 1
        chars = charCount(word)
        for key in chars:
            if key not in charSet:
                flag = 0
#             else:
#                 if charSet.count(key) != chars[key]:
#                     flag = 0
        if flag == 1:
            pos_words.append(word)
    return pos_words

def hamming_distance(string1, string2):
    '''
    Parameters
    __________
    string1: dtype str
    string1: dtype str

    Returns
    __________
    Computes and returns the Hamming distance between a target word and a current set of characters contained within the
    english alphabet.
    Used to determine how far an agent is far the target word and drives their decision making process at each timestep.

    '''

    # Start with a distance of zero, and count up
    distance = 0
    L = len(string1)
    for i in range(L):
        # Add 1 to the distance if these two characters are not equal
        if string1[i] != string2[i]:
            distance += 1
    # Return the final count of differences
    return distance
