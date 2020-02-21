"""
Tensorflow implementation of DeepFM [1]

Reference:
[1] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Embedding, concatenate, Flatten, BatchNormalization


class DeepFM:
    def __init__(self, feature_size, field_size,
                 embedding_size=8,
                 deep_layers=[32, 32],
                 use_fm=True, use_deep=True
                 ):
        assert (use_fm or use_deep)

        self.feature_size = feature_size  # denote as M, size of the feature dictionary
        self.field_size = field_size  # denote as F, size of the feature fields
        self.embedding_size = embedding_size  # denote as K, size of the feature embedding

        self.feat_emb = Embedding(self.feature_size, self.embedding_size)
        self.feat_bias = Embedding(self.feature_size, 1)

        self.deep_layers = deep_layers
        self.use_fm = use_fm
        self.use_deep = use_deep

    def build_model(self):
        feat_index = Input(shape=(self.field_size,), dtype='int32', name='feat_index')
        feat_value = Input(shape=(self.field_size,), dtype='float32', name='feat_value')
        feat_value_reshape = tf.reshape(feat_value, shape=(-1, self.field_size, 1))

        embedding = self.feat_emb(feat_index)  # (None, F, K)
        embedding = tf.math.multiply(embedding, feat_value_reshape)  # (None, F, K)

        # ----------------- FM -----------------
        # ---------- first order term ----------
        first_order = self.feat_bias(feat_index)  # (None, F, 1)
        first_order = tf.math.multiply(first_order, feat_value_reshape)  # (None, F, 1)
        first_order = Flatten()(first_order)  # (None, F)

        # ---------- second order term ---------------
        summed_square = tf.math.reduce_sum(embedding, axis=1)  # (None, K)
        summed_square = tf.math.square(summed_square)

        squared_sum = tf.math.square(embedding)  # (None, F, K)
        squared_sum = tf.reduce_sum(squared_sum, axis=1)  # (None, K)

        second_order = 0.5 * tf.math.subtract(summed_square, squared_sum)  # (None, K)

        # ----------------- Deep -----------------
        deep = Flatten()(embedding)
        for units in self.deep_layers:
            # deep = BatchNormalization()(deep)
            deep = Dense(units, activation='relu')(deep)

        # DeepFM
        if self.use_deep and self.use_fm:
            concat_input = concatenate([first_order, second_order, deep])
        elif self.use_deep:
            concat_input = deep
        elif self.use_fm:
            concat_input = concatenate([first_order, second_order])
        # concat_input = BatchNormalization()(concat_input)
        output = Dense(1, activation='sigmoid')(concat_input)

        model = Model(inputs=[feat_index, feat_value], outputs=output, name='deepFM')
        return model
