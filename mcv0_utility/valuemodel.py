"""
@author sourabhxiii
"""

import numpy as np
import tensorflow as tf
from tensorflow import layers

'''
# because of TF bug: https://github.com/tensorflow/tensorflow/issues/15736
Model = tf.keras.models.Model
Dense = tf.keras.layers.Dense
Activation = tf.keras.layers.Activation
Dropout = tf.keras.layers.Dropout
optimizers = tf.keras.optimizers
'''

# predicts V(s)
class ValueModel:
    def __init__(self, input_dim, FT, lr=0.01):
        # 1. define the model
        # 2. initialize class variables

        # 1
        with tf.name_scope('input'):
            features = tf.placeholder(dtype=tf.float32, shape=(None, input_dim), name='Features')
            V_target = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='V_target')

        with tf.name_scope('value_net'):
            nn = layers.Dense(8, kernel_initializer=tf.random_normal_initializer()
                , bias_initializer=tf.random_normal_initializer()
                , activation=tf.nn.relu, name='VM_D1')(features)

            nn = layers.Dense(4, kernel_initializer=tf.random_normal_initializer()
                , bias_initializer=tf.random_normal_initializer()
                , activation=tf.nn.relu, name='VM_D3')(nn)

            V_pred = layers.Dense(1, kernel_initializer=tf.random_normal_initializer()
                , bias_initializer=tf.random_normal_initializer()
                , activation=None, name='VM_D4')(nn)

        # 2
        self.FT = FT
        self.init = tf.global_variables_initializer()
        self.features = features
        self.V_target = V_target
        self.predict_op = V_pred
        # self.model = Model(inputs=features, outputs=V_pred)
        with tf.name_scope('gradient'):
            with tf.name_scope('VM_dxcent'):
                self.loss = tf.reduce_mean(tf.square(V_pred - V_target))
            with tf.name_scope('VM_train'):
                self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
    

    def set_session(self, sess):
        self.session = sess
        self.session.run(self.init)

    def update_weight(self, obs, target):
        features = self.FT.transform(np.atleast_2d(obs))
        self.session.run(self.train_op
                , feed_dict={self.features: features
                    , self.V_target: np.atleast_1d(target)})
        
    def predict(self, obs):
        features = self.FT.transform(np.atleast_2d(obs))
        return self.session.run(self.predict_op
                , feed_dict={self.features: features})
