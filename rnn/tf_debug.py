# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:57:30 2019

@author: johnk
"""

import tensorflow as tf 

print(tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
))

