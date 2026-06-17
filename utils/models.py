import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers 
from tensorflow.keras.applications.densenet import DenseNet121


def conv_3D_performant(input_shape, num_classes): # didnt end up working super great - GRU needed
    model = tf.keras.models.Sequential([
        # reversed the order of the filters to decrease the amount of processing needed (by a lot!)
        layers.Conv3D(32, (3,3,3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool3D((2,2,2)),

        layers.Conv3D(64, (3,3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool3D((2,2,2)),

        layers.Conv3D(128, (3,3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool3D((2,2,2)),

        # flatten makes an insane number of parameters (more than my GPU can handle) so using GlobalAveragePooling3D
        layers.GlobalAveragePooling3D(),

        # reducing the dense layer at the end for parameters reasons (this can be changed freely not too sure on it yet)
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
     
    ])
    return model

def conv_GRU(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        # changed all the layers from 3D to time distributed 2D (2D on each frame)
        layers.TimeDistributed(layers.Conv2D(32, (3, 3), padding='same'), input_shape=input_shape),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.Activation('relu')),
        

        layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same')),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.Activation('relu')),
        layers.TimeDistributed(layers.MaxPool2D((2, 2))),

        layers.TimeDistributed(layers.GlobalAveragePooling2D()),

        # GRU layer after the pooling
        # need reset_after=False so it works with amd hardware
        layers.GRU(64, return_sequences=False, reset_after=False), 
        layers.Dropout(0.3),

        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model 

def convLSTM_model(input_shape, num_classes): # first shot at a LSTM model (with some help from AI), didn't work great but better than 3DCNN alone
    if len(input_shape) == 3: 
      input_shape = (input_shape[0], input_shape[1], input_shape[2], 1) # adding channel dimension
      # have to be 4D since convLSTM2D requires a 5D tensor (batch_size, shape(3), channel)
    model = tf.keras.models.Sequential([
        layers.Input(shape=input_shape),

        layers.Bidirectional(
            layers.ConvLSTM2D(
                filters=32,
                kernel_size=(3, 3),
                padding='same',
                return_sequences=True,
                activation='tanh',
                recurrent_activation='sigmoid'
            )
        ),

        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2))),

        layers.Bidirectional(
            layers.ConvLSTM2D(
                filters=64,
                kernel_size=(3, 3),
                padding='same',
                return_sequences=False,
                activation='tanh',
                recurrent_activation='sigmoid'
            )
        ),

        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        ),

        layers.Dropout(0.4),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation='softmax')
    ])

    return model


