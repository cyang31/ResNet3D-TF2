#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chunhui Yang
mimic from https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection/blob/master/3DNet/models/resnet.py a pytorch version
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras import layers as Layers
from tensorflow_addons.layers import *
from functools import partial

def conv3x3x3(channels, strides=1, dilation_rate=1):
    return Conv3D(channels, kernel_size=3, strides=strides, padding='same', dilation_rate=dilation_rate, kernel_initializer='he_normal', bias_initializer='zeros', use_bias=False)

def downsample_basic_block(x, channels, stride):
    out = AveragePooling3D(pool_size=1, strides=stride)(x)
    zeros_pads = tf.zeros((x.shape[0], channels-x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
    out = Concatenate(axis=-1)([out,zeros_pads])
    return out
    
class ResBlock(Model):
    expansion = 1
    def __init__(self, channels, strides=1, dilation_rate=1, downsample=None, names=''):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3x3(channels, strides=strides, dilation_rate=dilation_rate)
        self.bn1 = BatchNormalization(axis=4)
        self.conv2 = conv3x3x3(channels, dilation_rate=dilation_rate)
        self.bn2 = BatchNormalization(axis=4)
        self.relu = ReLU()
        self.downsample = downsample
        self.names=names
        self.dilation = dilation_rate
        
    def call(self, x):
        residual = x
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        if self.downsample is not None:
            residual = self.downsample(x)
        x1 = Layers.add([residual, x1])
        x1 = self.relu(x1)
        return x1

class BottleNeck(Model):
    expansion = 4
    def __init__(self, channels, strides=1, dilation_rate=1, downsample=None, names=''):
        super(BottleNeck, self).__init__()
        self.conv1 = Conv3D(channels, kernel_size=1, kernel_initializer='he_normal', bias_initializer='zeros', use_bias=False)
        self.bn1 = BatchNormalization(axis=4)
        self.conv2 = Conv3D(channels, kernel_size=3, strides=strides, padding='same', kernel_initializer='he_normal', bias_initializer='zeros', use_bias=False)
        self.bn2 = BatchNormalization(axis=4)
        self.conv3 = Conv3D(channels*4, kernel_size=1, kernel_initializer='he_normal', bias_initializer='zeros', use_bias=False)
        self.bn3 = BatchNormalization(axis=4)
        self.relu = ReLU()
        self.downsample = downsample
        self.strides = strides
        self.dilation_rate = dilation_rate
    
    def call(self, x):
        residual = x
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu(x1)
        
        x1 = self.conv3(x1)
        x1 = self.bn3(x1)
        if self.downsample is not None:
            residual = self.downsample(x)
        
        x1 = Layers.add([residual, x1])
        x1 = self.relu(x1)
        return x1
        
        
class ResNet3D(Model):
    def __init__(self, block, layers, feat_num=128, shortcut_type='B', num_class=2):
        super(ResNet3D, self).__init__(name='ResNet3D')
        self.in_channels = 64
        self.conv1 = Conv3D(64, kernel_size=7, strides=(2,2,2), padding='same', kernel_initializer='he_normal', bias_initializer='zeros', use_bias=False)
        self.bn = BatchNormalization(axis=4)
        self.relu = ReLU()
        self.mp1 = MaxPooling3D(pool_size=(3,3,3), strides=2, padding='valid', data_format='channels_last')
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type, stride=1, name='Layer1')
        self.layer2 = self._make_layer(block, 64*2, layers[1], shortcut_type, stride=2, name='Layer2')
        self.layer3 = self._make_layer(block, 128*2, layers[2], shortcut_type, stride=1, dilation_rate=2, name='Layer3')
        self.layer4 = self._make_layer(block, 256*2, layers[3], shortcut_type, stride=1, dilation_rate=4, name='Layer4')
        
        self.pool = GlobalAveragePooling3D(data_format='channels_last')
        self.flatten = Flatten()
        self.fc_feat = Dense(feat_num, activation='relu')
        self.bn2 = BatchNormalization(axis=-1)
        self.fc = Dense(num_class, activation='linear')
        
    def _make_layer(self, block, channels, blocks_num, shortcut_type, stride=1, dilation_rate=1, name=''):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block, 
                                     channels = channels*channels * block.expansion,
                                     stride=stride)
            else:
                downsample = keras.Sequential([Conv3D(channels*block.expansion, kernel_size=1, strides=stride, use_bias=False), BatchNormalization(axis=4)], name='Downsample')
        
        layers = []
        layers.append(block(channels, strides=stride, dilation_rate=dilation_rate, downsample=downsample, names=name))
        self.in_channels = channels * block.expansion
        for i in range(1, blocks_num):
            layers.append(block(channels, dilation_rate=dilation_rate))
        return keras.Sequential(layers=layers, name=name)
    
    def call(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.mp1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.pool(x)
        #x = AdaptiveAveragePooling3D((1,1,1), data_format='channels_last')(x)
        x = self.flatten(x)
        #x = self.fc_feat(x)
        #x = self.bn2(x) 
        #x = self.relu(x)
        x = self.fc(x)
        return x
    
    def model(self, input_shape):
        input = Input(shape=input_shape,name='input')
        model=Model(inputs=input,outputs=self.call(input))
        
        return model


# resnet10
"""Constructs a ResNet-10 model.
model = ResNet3D(BasicBlock, [1, 1, 1, 1])
"""
# resnet18
"""Constructs a ResNet-18 model.
model = ResNet3D(BasicBlock, [2, 2, 2, 2])
"""
# resnet34
"""Constructs a ResNet-34 model.
model = ResNet3D(BasicBlock, [3, 4, 6, 3])
"""
# resnet50
"""Constructs a ResNet-34 model.
model = ResNet3D(BottleNeck, [3, 4, 6, 3])
"""
# resnet101
"""Constructs a ResNet-34 model.
model = ResNet3D(BottleNeck, [3, 4, 23, 3])
"""
# resnet152
"""Constructs a ResNet-34 model.
model = ResNet3D(BottleNeck, [3, 8, 36, 3])
"""
# resnet200
"""Constructs a ResNet-34 model.
model = ResNet3D(BottleNeck, [3, 24, 36, 3])
"""
#model = ResNet34()
#model.build(input_shape=(1, 480, 480, 3))
#model.summary()
