import numpy as np


def calc_character_data(inData):
    outData = inData + 1
    return outData


rawData = np.load()

labelData = np.array()

characterData = calc_character_data(rawData)
