"""
@author sourabhxiii
"""
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import layers

# approximates pi(a | s)
class PolicyModel:
    def __init__(self, input_dim, output_dim, FT, lr=0.001):
        # 1. define the model
        # 2. initialize class variables

        # 1
        with tf.name_scope('input'):
            self.features = tf.placeholder(tf.float32, shape=(None, input_dim), name='features')
            self.action = tf.placeholder(tf.int32, shape=(None,), name='action')
            self.advantages = tf.placeholder(tf.float32, shape=(None, 1), name='advantages')

        with tf.name_scope('policy_net'):
            nn = layers.Dense(8, kernel_initializer=tf.initializers.truncated_normal() #random_normal_initializer()
                , bias_initializer=tf.random_normal_initializer()
                , activation=tf.nn.relu, name='PM_D1')(self.features)

            nn = layers.Dense(4, kernel_initializer=tf.random_normal_initializer()
                , bias_initializer=tf.random_normal_initializer()
                , activation=tf.nn.relu, name='PM_D3')(nn)

            self.logits = layers.Dense(output_dim, kernel_initializer=tf.random_normal_initializer()
                , bias_initializer=tf.random_normal_initializer()
                , activation=None, name='PM_D4')(nn)

        with tf.name_scope("PM_axndist"):
            self.action_probs = tf.squeeze(tf.nn.softmax(self.logits))
        with tf.name_scope("PM_axnproba"):
            self.picked_action_prob = tf.gather(self.action_probs, self.action) 
        
        log_probs = tf.log(self.picked_action_prob)

        # 2
        self.FT = FT
        self.init = tf.global_variables_initializer()
        self.predict_op = tf.argmax(self.action_probs, None)
        self.log_file="./summary_log/run"+datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        with tf.name_scope('gradient'):
            with tf.name_scope("PM_xcent"):
                self.loss = -tf.reduce_mean(self.advantages * log_probs)
                # tf.summary.scalar("loss", self.loss)
            with tf.name_scope("PM_train"):
                self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

        # merged = tf.summary.merge_all()
        # train_writer = tf.summary.FileWriter('/log', sess.graph)

    def set_session(self, sess):
        self.session = sess
        self.session.run(self.init)

    def model_session(self):
        return self.session

    def get_action(self, obs):
        features = self.FT.transform(np.atleast_2d(obs))
        # print("logits")
        # print(self.session.run(self.logits, feed_dict={self.features: features}))
        # print("axn proba")
        # print(self.session.run(self.action_probs, feed_dict={self.features: features}))
        axn_proba = self.session.run(self.action_probs, feed_dict={self.features: features})
        axn = np.random.choice(range(len(axn_proba.ravel())), p=axn_proba.ravel())
        return axn

    def update_weight(self, obs, action, advantage, i):
        features = self.FT.transform(np.atleast_2d(obs))
        feed_dict = {self.features: features
            , self.advantages:  np.atleast_1d(advantage)
            , self.action: np.atleast_1d(action)}
        _, loss = self.session.run([self.train_op, self.loss], feed_dict)

        # tvars = tf.trainable_variables()
        # tvars_vals = self.session.run(tvars)

        # for var, val in zip(tvars, tvars_vals):
        #     if var.name == 'PM_D4/kernel:0' or var.name == 'VM_D4/kernel:0':
        #         print(var.name, val)  # Prints the name of the variable alongside its value.
        return loss
    


        