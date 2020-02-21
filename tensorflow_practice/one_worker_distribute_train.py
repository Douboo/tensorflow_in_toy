# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow_datasets as tfds
import time

t1 = time.time()

# gpu显存动态增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    # 指定卡
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


num_epochs = 5
batch_size_per_replica = 80
learning_rate = 0.001

# strategy = tf.distribute.MirroredStrategy()
# print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
# batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
batch_size = batch_size_per_replica * 2

# 载入数据集并预处理
def resize(image, label):
    image = tf.image.resize(image, [224, 224]) / 255.0
    return image, label

# 当as_supervised为True时，返回image和label两个键值
dataset = tfds.load("cats_vs_dogs", split=tfds.Split.TRAIN, as_supervised=True)
dataset = dataset.map(resize, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10240).batch(batch_size).prefetch(1)

# with strategy.scope():
model = tf.keras.applications.MobileNetV2()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy]
)

model.fit(dataset, epochs=num_epochs)

print("cost time:", time.time()-t1)
