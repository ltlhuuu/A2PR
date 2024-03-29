"""
============================
# -*- coding: utf-8 -*-
# @Time    : 2023/9/11 下午4:05
# @Author  : ltl
# @FileName: util.py
# @Software: PyCharm
# @Github ：https://github.com/ltlhuuu
===========================
"""
import numpy as np
import torch


def advantage_weighted(Q, value):
    advantage_list = []
    for i in range(value.shape[0]):
        advantage = Q[i] - value[i]
        advantage = torch.exp(advantage)
        if advantage > 0:
            advantage_list.append(advantage)
        else:
            advantage_list.append(0.0)
    return torch.FloatTensor(advantage_list)