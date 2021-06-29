from abc import ABC
from keras.models import Sequential
from keras.layers import Dense, Flatten, TimeDistributed, BatchNormalization, LSTM
from keras.layers.convolutional import Conv2D, MaxPooling2D

import keras
import tensorflow as tf
import functools


class Select_model(tf.keras.layers.Layer, ABC):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00, amsgrad=False)
        self.loss = 'categorical_crossentropy'

        self.top2_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=2)
        self.top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
        self.top5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)
        self.top10_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=10)

    def model_ROI(self, class_num, input_size):
        model = Sequential()

        model.add(TimeDistributed(Conv2D(96, (3, 3), activation='relu', padding='same'), input_shape=input_size))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(MaxPooling2D((3, 3), strides=2)))

        model.add(TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same')))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(MaxPooling2D((3, 3), strides=2)))

        model.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same')))
        model.add(TimeDistributed(BatchNormalization()))

        model.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same')))
        model.add(TimeDistributed(BatchNormalization()))

        model.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same')))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(MaxPooling2D((3, 3), strides=2)))

        model.add(TimeDistributed(Flatten()))
        model.add(TimeDistributed(Dense(512, activation='relu')))

        model.add(LSTM(512, return_sequences=True))
        model.add(LSTM(512, return_sequences=False))

        model.add(Dense(class_num, activation='softmax'))
        model.summary()

        return model

    def model_landmark(self, class_num, input_size):
        model = Sequential()

        model.add(TimeDistributed(Conv2D(96, (3, 2), activation='relu', padding='same'), input_shape=input_size))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(MaxPooling2D((3, 1))))

        model.add(TimeDistributed(Conv2D(256, (3, 2), activation='relu', padding='same')))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(MaxPooling2D((3, 1))))

        model.add(TimeDistributed(Conv2D(512, (3, 2), activation='relu', padding='same')))
        model.add(TimeDistributed(BatchNormalization()))

        model.add(TimeDistributed(Conv2D(512, (3, 2), activation='relu', padding='same')))
        model.add(TimeDistributed(BatchNormalization()))

        model.add(TimeDistributed(Conv2D(512, (3, 2), activation='relu', padding='same')))
        model.add(TimeDistributed(BatchNormalization()))

        model.add(TimeDistributed(MaxPooling2D((3, 1))))

        model.add(TimeDistributed(Flatten()))
        model.add(TimeDistributed(Dense(512, activation='relu')))

        model.add(LSTM(512, return_sequences=True))
        model.add(LSTM(512, return_sequences=False))

        model.add(Dense(class_num, activation='softmax'))
        # model.summary()

        return model

    def train_compile(self, model):
        model.compile(loss=self.loss, optimizer=self.adam, metrics=['acc'])

        return model

    def predict_compile(self, model):
        self.top2_acc.__name__ = 'top2_acc'
        self.top3_acc.__name__ = 'top3_acc'
        self.top5_acc.__name__ = 'top5_acc'
        self.top10_acc.__name__ = 'top10_acc'

        model.compile(loss='categorical_crossentropy', optimizer='Adam',
                      metrics=['acc', self.top2_acc, self.top3_acc,
                               self.top5_acc, self.top10_acc])

        return model
