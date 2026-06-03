import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers 
from tensorflow.keras.models import Model 
from tensorflow.keras.applications.densenet import DenseNet121

def multi_conv_3D(input_shape, num_classes_task1, num_classes_task2):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv3D(128, (3,3,3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv3D(128, (3,3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPool3D((2,2,2), strides=(2,2,2))(x)

    x = layers.Conv3D(64, (3,3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv3D(64, (3,3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPool3D((2,2,2), strides=(2,2,2))(x)

    x = layers.Conv3D(32, (3,3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv3D(32, (3,3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPool3D((2,2,2), strides=2)(x)

    
    flattened = layers.Flatten()(x)

    # Task 1 Head 
    task1_dense = layers.Dense(256, activation='relu')(flattened)
    task1_dropout = layers.Dropout(0.2)(task1_dense)
    out_task1 = layers.Dense(num_classes_task1, activation='softmax', name='task1_output')(task1_dropout)

    # Task 2 Head
    task2_dense = layers.Dense(256, activation='relu')(flattened)
    task2_dropout = layers.Dropout(0.2)(task2_dense)
    out_task2 = layers.Dense(num_classes_task2, activation='softmax', name='task2_output')(task2_dropout)

    model = Model(inputs=inputs, outputs=[out_task1, out_task2], name="multi_task_3d_cnn")
    
    return model