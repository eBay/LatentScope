import json
import math
import numpy as np

from typing import *


def metric_type_map(dataset: str, key: str):
    if dataset == 'dataset_b':
        key_info = json.loads(key)
        if key_info['kpi'] in ['mr', 'rr']:
            cur_type = 'E'
        elif key_info['kpi'] in ['mrt']:
            cur_type = 'L'
        elif key_info['kpi'] in ['count']:
            cur_type = 'T'
        else:
            cur_type = 'S'
    else:
        raise NotImplementedError(f"Unrecognized dataset: {dataset}")

    return cur_type


def ccs(x, y):
    """
    Calculate the correlation coefficient score for two time series
    :param x: (array_like list or np.array or pd.Series): The input time series
    :param y: (array_like list or np.array or pd.Series): The target time series
    :return:  (tuple) (score, shift) optimized
    """
    length = 2 ** (math.ceil(math.log(2 * len(x) - 1, 2)))
    m = len(x)

    # use fft and ifft to accelerate the cross-correlation calculation per convolution theorem
    # use abs func to cover both positive and negative correlation
    value = np.abs(np.fft.ifft(np.prod([np.fft.fft(x, length), np.conj(np.fft.fft(y, length))], axis=0)))

    r_max_val = np.real(np.max(value[:m]))
    l_max_val = np.real(np.max(value[-m + 1:]))
    max_val = max(r_max_val, l_max_val)

    # slide y through x
    if r_max_val >= l_max_val:  # positive lag, y shift towards right
        index = np.argmax(value[:m])
        shift = index
    elif r_max_val < l_max_val:  # negative lag, y shift towards left.
        index = np.argmax(value[-m + 1:])
        shift = index + 1 - m
    else:
        raise ValueError(x + "index error.")

    score = max_val / ((np.linalg.norm(x, ord=2) + 1e-7) * (np.linalg.norm(y, ord=2) + 1e-7))

    return score, shift
