#!/usr/bin/env python
# -*- coding: utf-8 -*-
import em
import numpy as np
import random

data = np.load('data1000_2016.npy')
data = data[:, :, :5]
for i in range(5):
    data[:, :, i] /= 255.
params, LL = em.run_EM(data, h=100, w=78)
print(params[0])
