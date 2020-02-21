# -*- coding: utf-8 -*-

import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, Activation, Reshape
from tensorflow.keras import Model
from tensorflow import keras
import tempfile
import shutil
from utils.common import choose_gpu

def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=['acc'])
    return model

def build_model_1():
    input = Input(shape=(28,28))
    flatten = Flatten()(input)
    dense_1 = Dense(512, activation='relu')(flatten)
    dense_2 = Dense(10)(dense_1)
    dense_2_reshape = Reshape((10, 1))(dense_2)
    with_gamma = Conv1D(1, 1, kernel_initializer=keras.initializers.Ones(), use_bias=False)(dense_2_reshape)
    with_gamma = Flatten()(with_gamma)
    prob = Activation("softmax")(with_gamma)
    model = Model(inputs=input, outputs=prob)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    return model

def load_mnist_data():
    mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    choose_GPU()
    x_train, y_train, x_test, y_test = load_mnist_data()
    model = build_model()
    checkpoint_dir = "../checkpoints_test"
    log_dir = './logs_test'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 保留整个模型
    checkpoint_path = os.path.join(checkpoint_dir, "weights.hdf5")
    # save_best_only=True
    # checkpoint_path = os.path.join(checkpoint_dir, "weights.hdf5")
    callbacks = [
        ModelCheckpoint(checkpoint_path,
                        monitor="val_loss",
                        save_best_only=True),
        EarlyStopping(patience=10, monitor='val_loss'),
        TensorBoard(log_dir='./logs')
    ]
    model = build_model()
    model.fit(x_train, y_train,
              validation_data=[x_test, y_test],
              epochs=10,batch_size=64,
              callbacks=callbacks,
              verbose=2)
    MODEL_DIR = tempfile.gettempdir()
    version = 1
    export_path = os.path.join(MODEL_DIR, "dssm_model", str(version))
    print('export_path = {}'.format(export_path))
    shutil.rmtree(export_path, ignore_errors=True)
    # 模型serving时使用
    tf.saved_model.save(
        model,
        export_path
    )
    print("save model done.")
    # 直接恢复整个模型
    model_path = checkpoint_path
    restore_model1 = load_model(model_path)
    print(restore_model1.evaluate(x_test, y_test))




