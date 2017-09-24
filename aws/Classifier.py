import tensorflow as tf
import numpy as np


class Classifier:
    def __init__(self, init = False,size = (500,500), channels = 3, n_out = 1):
        self.session = tf.Session()
        if init:
            with self.session.as_default():
                self.initialize(size = size, channels = 3, n_out = n_out)

    def initialize(self, size = (500,500), channels = 3, n_out = 1):
        self.input = tf.placeholder(tf.float32, shape = (None, size[0],size[1], channels), name = 'input')
        self.labels = tf.placeholder(tf.float32, shape = (None, n_out), name = 'label')
        
        conv1 = tf.layers.conv2d(self.input, filters = 16, kernel_size = (4,4), padding = 'same')
        
        mp1 = tf.layers.max_pooling2d(conv1, pool_size = (2,2), strides = (1,1))

        conv2 = tf.layers.conv2d(mp1, filters = 8, kernel_size = (2,2), padding = 'same')

        mp2 = tf.layers.max_pooling2d(conv2, pool_size = (2,2), strides = (1,1))
        
        d1 = tf.layers.dense(tf.contrib.layers.flatten(mp2), 500)

        self.output = tf.layers.dense(d1, n_out, name = 'output')


        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.labels,self.output), name = 'loss')

        self.trainer = tf.train.AdamOptimizer(name = 'trainer').minimize(self.loss)
        self.session.run(tf.global_variables_initializer())

    def train(self, data, labels, n = 10):
        
        feed_dict = {'input:0':data, 'label:0':labels}
        for _ in range(n):
            print(self.session.run(self.loss, feed_dict = feed_dict))
            self.session.run('trainer', feed_dict = feed_dict)
    
    def predict(self,sample):
        feed_dict = {'input:0' : sample}
        return self.session.run(self.output, feed_dict = feed_dict)

    def save(self, path):
        with self.session.as_default():
            saver = tf.train.Saver()
            saver.save(self.session, path)
    def kill(self):
        self.session.close()
    def load(self,path):
        saver = tf.train.Saver()
        saver.restore(self.session,path)
