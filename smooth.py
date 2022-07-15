import pandas as pd
import numpy as np
import os

from typing import Sequence
from copy import copy

"""推导
[0.9 1] 即 arr_s[1] = (0.9 * arr[0] + arr[1]) / 1.9
[0.81 0.9 1] 即 arr_s[2] = (0.81 * arr[0] + 0.9 * arr[1] + arr[2]) / 2.71
= (0.9 * (0.9 * arr[0] + arr[1]) + arr[2]) / 2.71
= (0.9 * 1.9 * arr_s[1] + arr[2]) / 2.71
= (weight * arr_s[1] + arr[2]) / (weight + 1); weight' = smooth * weight
"""


def tensorboard_smoothing(arr: Sequence, smooth: float = 0.9) -> Sequence:
    """tensorboard smoothing  底层算法实现

    :param arr: shape(N,). const.
    :param smooth: smoothing系数
    :return: new_x
    """
    arr = copy(arr)
    weight = smooth  # 权重 (动态规划)
    for i in range(1, len(arr)):
        arr[i] = (arr[i - 1] * weight + arr[i]) / (weight + 1)
        weight = (weight + 1) * smooth  # `* smooth` 是为了让下一元素 权重为1
    return arr


def smooth(csv_path,weight=0.99):
    data = pd.read_csv(filepath_or_buffer=csv_path, header=0,names=['Step','Value'],dtype={'Step':np.int,'Value':np.float})
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    save = pd.DataFrame({'Step':data['Step'].values,'Value':smoothed})
    save.to_csv('smooth.csv')



if __name__=='__main__':
    smooth('data/3x3Traffic-domain-QMIX-0516_notask.csv')