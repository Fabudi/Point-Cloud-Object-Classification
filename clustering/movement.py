import numpy as np


def movement(a, b, threshold=0.0000001):
    '''
    :param a: Point Cloud A
    :param b: Point Cloud B
    :param threshold: Threshold value.
    :return: moving points array
    '''
    diff = np.linalg.norm(np.asarray(a) - np.asarray(b), axis=1)
    print("asd")
    print(diff)
    significant_diff = diff[np.linalg.norm(diff) > threshold]
    print(significant_diff.shape)
    return np.asarray(significant_diff).tolist()