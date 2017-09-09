from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense
from keras.layers.core import Flatten
import numpy as np


class Classifier:
    def __init__(self, size = (500,500), channels = 3):
        self.model = Sequential()
        self.model.add(Conv2D(64, (4, 4), padding = 'same', batch_input_shape = (None,size[0],size[1], channels)))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(32, (2,2), padding = 'same'))
        self.model.add(MaxPooling2D())
        self.model.add(Flatten())
        self.model.add(Dense(500))
        self.model.add(Dense(2))
        
        self.model.compile(loss = 'mean_squared_error',
            optimizer='adam',
            metrics = ['accuracy'])
    def train(self, data, labels, n = 10):
        self.model.fit(np.array(data), np.array(labels), epochs = n)

    def predict(self,sample):
        return self.model.predict(sample)

