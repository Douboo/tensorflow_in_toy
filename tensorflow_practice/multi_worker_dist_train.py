# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow_datasets as tfds
import os
import json

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

def resize(image, label):
    image = tf.image.resize(image, [224, 224]) / 255.0
    return image, label


if __name__ == "__main__":
    choose_gpu(is_choose=False)
    num_epochs = 50
    batch_size_per_replica = 64
    learning_rate = 0.001

    num_workers = 2
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': ["10.163.1.131:20000", "10.163.1.134:20001"]
        },
        'task': {'type': 'worker', 'index': 0}
    })
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    batch_size = batch_size_per_replica * num_workers

    dataset = tfds.load("cats_vs_dogs", split=tfds.Split.TRAIN, as_supervised=True)
    dataset = dataset.map(resize).shuffle(1024).batch(batch_size)

    with strategy.scope():
        model = tf.keras.applications.MobileNetV2()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=[tf.keras.metrics.sparse_categorical_accuracy]
        )

    model.fit(dataset, epochs=num_epochs)