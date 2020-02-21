# -*- coding: utf-8 -*-

import tensorflow as tf

def load_mnist_data():
    mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train, x_test, y_test