# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from common import load_mnist_data

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

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist_data()
    model = build_model()
    checkpoint_dir = "../checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    # 只保留权重
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_prefix,
        save_weights_only=True
    )
    model.fit(x_train, y_train,
              validation_data=[x_test, y_test],
              epochs=10,batch_size=64,
              callbacks=[checkpoint_callback])
    # 从权重中恢复模型
    restore_model = build_model()
    restore_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    restore_model.summary()
    restore_model.evaluate(x_test, y_test)

    # 保留整个模型
    checkpoint_path = os.path.join(checkpoint_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5")
    # save_best_only=True
    # checkpoint_path = os.path.join(checkpoint_dir, "weights.hdf5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                             monitor="val_loss",
                                                             # save_best_only=True
    )
    model = build_model()
    model.fit(x_train, y_train,
              validation_data=[x_test, y_test],
              epochs=10,batch_size=64,
              callbacks=[checkpoint_callback])
    # 直接恢复整个模型
    model_path = ""
    restore_model1 = load_model(model_path)
    restore_model1.evaluate(x_test, y_test)




