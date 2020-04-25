#    Copyright 2020-2022 AIBharata Emerging Technologies Pvt. Ltd.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from __future__ import absolute_import
from .nnets import resnet,covid_net

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import sys
import copy
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
import tensorflow as tf
class NetworkInit(object):
    """Base class for parameter Network initializers.

    The :class:`NetworkInit` class represents a network initializer used
    to initialize network/model parameters for numerous medical ai networks. It should be
    subclassed when implementing new types of network initializers.
    """
    def __call__(self, inputSize, OutputSize, convLayers=None):
        """Makes :class:`NetworkInit` instances callable like a function, invoking
        their :meth:`call()` method.
        """
        return self.call(inputSize, OutputSize, convLayers)

    def call(self, inputSize, OutputSize, convLayers=None):
        """Sample should return model initialized with input and output Sizes.
        
        Parameters
        ----------
        inputSize : tuple or int.
            Integer or tuple specifying the input of network.

        OutputSize : tuple or int.
            Integer or tuple specifying the output classes of network.

        Returns
        -------
        numpy.array. 
            Initialized Model.
        """
        raise NotImplementedError()

    def __str__(self):
        return self.__class__.__name__


class tinyMedNet(NetworkInit):
    """tinyMedNet is a classification network that consumes very less resources and can be trained even on CPUs. This network can be used to demonstrate the framework working.
       Additionally this acts a starting point for example/tutorial for getting started to know the Medical AI library.
    """
    def call(self, inputSize, OutputSize, convLayers=None):
        try:
            model = Sequential([
                Conv2D(64, kernel_size=(5, 5), strides=(1, 1),activation='relu', padding = 'valid',input_shape=inputSize, name='CNN1'),
                MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
                Conv2D(64, kernel_size=(5, 5), strides=(1, 1),activation='relu', padding = 'valid', name='CNN2'),
                MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
                Conv2D(64, kernel_size=(5, 5), strides=(1, 1),activation='relu', padding = 'valid', name='CNN3'),
                MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
                Flatten(),
                Dense(384, activation='relu', name='FC1'),
                Dense(192, activation='relu', name='FC2'),
                Dense(OutputSize, activation='softmax', name='FC3')
            ])
        except ValueError:
            model = Sequential([
                Conv2D(64, kernel_size=(5, 5), strides=(1, 1),activation='relu', padding = 'valid',input_shape=inputSize, name='CNN1'),
                MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
                Conv2D(64, kernel_size=(5, 5), strides=(1, 1),activation='relu', padding = 'valid', name='CNN2'),
                MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
                Conv2D(64, kernel_size=(5, 5), strides=(1, 1),activation='relu', padding = 'same', name='CNN3'),
                MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
                Flatten(),
                Dense(384, activation='relu', name='FC1'),
                Dense(192, activation='relu', name='FC2'),
                Dense(OutputSize, activation='softmax', name='FC3')
            ])            
        return model

class tinyMedNet_v2(NetworkInit):
    """tinyMedNet_v2 allows users to configure the number of Conv/CNN layers.
       tinyMedNet_v2 is a classification network that consumes very less resources and can be trained even on CPUs. This network can be used to demonstrate the framework working.
       Additionally this acts a starting point for example/tutorial for getting started to know the Medical AI library.
    """
    def call(self, inputSize, OutputSize, convLayers=2):
        try:
            model = Sequential()
            model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),activation='relu', padding = 'valid',input_shape=inputSize, name='CNN1'))
            model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
            for cnnLayerNum in range(0,convLayers-1):
                model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),activation='relu', padding = 'valid', name='CNN'+str(cnnLayerNum+2)))
                model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
            model.add(Flatten())
            model.add(Dense(384, activation='relu', name='FC1'))
            model.add(Dense(192, activation='relu', name='FC2'))
            model.add(Dense(OutputSize, activation='softmax', name='FC3'))
            return model
        except ValueError:
            print(20*'-')
            print('Dimension Error Occured')
            print('SOLUTION: Try increasing the Input Dimension or Reducing the number of Layers')
            print(20*'-')
            sys.exit(1)
        
class tinyMedNet_v3(NetworkInit):
    """tinyMedNet_v3 has 3 FC layers with Dropout and Configurable number of Conv/CNN Layers.
    """
    def call(self, inputSize, OutputSize, convLayers=2):
        try:
            model = Sequential()
            model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),activation='relu', padding = 'valid',input_shape=inputSize, name='CNN1'))
            model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
            for cnnLayerNum in range(0,convLayers-1):
                model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),activation='relu', padding = 'valid', name='CNN'+str(cnnLayerNum+2)))
                model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
            model.add(Dropout(rate=0.2))
            model.add(Flatten())
            model.add(Dense(512, activation='relu', name='FC1'))
            model.add(Dropout(rate=0.7))
            model.add(Dense(384, activation='relu', name='FC2'))
            model.add(Dropout(rate=0.5))
            model.add(Dense(192, activation='relu', name='FC3'))
            model.add(Dropout(rate=0.5))
            model.add(Dense(OutputSize, activation='softmax', name='FC4'))
            return model
        except ValueError as err:
            print(err)
            print(20*'-')
            print('Dimension Error Occured')
            print('SOLUTION: Try increasing the Input Dimension or Reducing the number of Layers')
            print(20*'-')
            sys.exit(1)            
        
class resNet20(NetworkInit):
    """resnet20
    """
    def call(self, inputSize, OutputSize, convLayers=None):
        img_input = tf.keras.layers.Input(shape=inputSize)
        return resnet.resnet20(img_input=img_input,classes=OutputSize)

class resNet32(NetworkInit):
    """resnet32
    """
    def call(self, inputSize, OutputSize, convLayers=None):
        img_input = tf.keras.layers.Input(shape=inputSize)
        return resnet.resnet32(img_input=img_input,classes=OutputSize)

class resNet56(NetworkInit):
    """RESNET56
    """
    def call(self, inputSize, OutputSize, convLayers=None):
        img_input = tf.keras.layers.Input(shape=inputSize)
        return resnet.resnet56(img_input=img_input,classes=OutputSize)

class resNet110(NetworkInit):
    """resnet110
    """
    def call(self, inputSize, OutputSize, convLayers=None):
        img_input = tf.keras.layers.Input(shape=inputSize)
        return resnet.resnet110(img_input=img_input,classes=OutputSize)

class megaNet(NetworkInit):
    """
    megaNet is based on COVID-NET.
    This is a tensorflow 2.0 network variant for COVID-Net described in Paper "COVID-Net: A Tailored Deep Convolutional Neural Network Design for Detection of COVID-19 Cases from Chest Radiography Images" by Linda Wang et al. 
    Reference: https://github.com/busyyang/COVID-19/
    """
    def call(self, inputSize, OutputSize, convLayers=None):
        return covid_net.COVIDNET_Keras(img_input=inputSize,classes=OutputSize)

def get(networkInitialization):
    if networkInitialization.__class__.__name__ == 'str':
        if networkInitialization in ['tinyMedNet', 'tiny_Medical_Network']:
            return tinyMedNet()
        elif networkInitialization in ['tinyMedNet_v2', 'tiny_Medical_Network_v2']:
            return tinyMedNet_v2()
        elif networkInitialization in ['tinyMedNet_v3', 'tiny_Medical_Network_v3']:
            return tinyMedNet_v3()
        elif networkInitialization in ['resNet20', 'resnet20']:
            return resNet20()
        elif networkInitialization in ['resNet32', 'resnet32']:
            return resNet32()
        elif networkInitialization in ['resNet56', 'resnet56']:
            return resNet56()
        elif networkInitialization in ['resNet110', 'resnet110']:
            return resNet110()
        elif networkInitialization in ['megaNet', 'meganet']:
            return megaNet()
        raise ValueError('Unknown network Initialization name: {}.'.format(networkInitialization))

    elif isinstance(networkInitialization, NetworkInit):
        return copy.deepcopy(networkInitialization)

    else:
        raise ValueError("Unknown type: {}.".format(networkInitialization.__class__.__name__))


if __name__ == "__main__":
    v=get('resNet56')
    print(10*'~', 'Tiny Net V1')
    INPUT_DIM= 96
    m = v((INPUT_DIM,INPUT_DIM,3),10)
    m.summary()

    v=get('tinyMedNet_v2')
    print(10*'~', 'Tiny Net V2')
    for i in range(1,10):
        print(10*'-', 'CNN LAYERS =', i)
        m = v((INPUT_DIM,INPUT_DIM,3),10,i)
        m.summary()