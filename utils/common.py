# -*- coding: utf-8 -*-

import tensorflow as tf
import time
import functools

def choose_gpu(is_choose=True, num=0):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if is_choose:
                tf.config.experimental.set_visible_devices(gpus[num], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def timeit(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        begin = time.time()
        result = fn(*args, **kwargs)
        print("cost %.2fs for {%s}" % (time.time()-begin, fn.__name__))
        return result
    return wrapper
