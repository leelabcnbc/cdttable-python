import numpy as np


def unique_rows(a):
    assert a.ndim == 2
    # from http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    if a.size > 0:
        result = np.vstack({tuple(row) for row in a})
    else:
        result = a
    return result


def run():
    print('hello!')
