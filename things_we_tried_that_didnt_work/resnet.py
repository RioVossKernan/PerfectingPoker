
import tensorflow as tf
import pickle
from PIL import Image
import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


#
#       This is a resnet architecture that I made but it did not work as well as the conventional CNN
#


class ResidualBlock(tf.keras.layers.Layer):
    ''' Res Block for an attempt at a ResNet architecture clone'''
    def __init__(self, filters, downsample=False):
        super(ResidualBlock, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.filters = filters
        self.downsample = downsample

        # downsample will halve the image size
        strides = 2 if downsample else 1

        self.conv1 = tf.keras.layers.Conv2D(filters, 3, strides=strides, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.bn2 = tf.keras.layers.BatchNormalization()

        # Optional downsampling for the shortcut path
        if downsample:
            self.downsample_conv = tf.keras.layers.Conv2D(filters, 1, strides=2, padding='same', use_bias=False)
            self.downsample_bn = tf.keras.layers.BatchNormalization()
        else:
            self.downsample_conv = None
            
    def call(self, inputs, training=False):
        shortcut = inputs

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        if self.downsample_conv:
            shortcut = self.downsample_conv(shortcut)
            shortcut = self.downsample_bn(shortcut, training=training)

        x = tf.keras.layers.add([x, shortcut])
        return self.relu(x)
        

class MockResNet(tf.keras.Model):
    def __init__(self):
        super(MockResNet, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        
        data_augmentation = tf.keras.models.Sequential([
                                tf.keras.layers.RandomRotation(0.1),
                                tf.keras.layers.RandomZoom(0.1),
                            ])
        
        self.architecture = [
            tf.keras.Input(shape=(300, 300, 3)),
            #data_augmentation,
            
            tf.keras.layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same'),
            
            tf.keras.layers.Dropout(0.4),
            ResidualBlock(64),
            tf.keras.layers.Dropout(0.4),
            ResidualBlock(64),

            ResidualBlock(128, downsample=True),
            tf.keras.layers.Dropout(0.4),
            ResidualBlock(128),
            tf.keras.layers.Dropout(0.4),

            ResidualBlock(256, downsample=True),
            tf.keras.layers.Dropout(0.4),
            # ResidualBlock(256),
            # tf.keras.layers.Dropout(0.4),

            ResidualBlock(512, downsample=True),
            tf.keras.layers.Dropout(0.4),
            # ResidualBlock(512),
            # tf.keras.layers.Dropout(0.3),

            tf.keras.layers.GlobalAveragePooling2D(),
            
            #tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(52, activation='softmax')
        ]
        self.sequential = tf.keras.Sequential(self.architecture)
    
    def call(self, inputs, training=False):
        return self.sequential(inputs, training=training)
        
    @staticmethod
    def loss_fn(labels, predictions): 
           """ Loss function for the model. """
           return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)