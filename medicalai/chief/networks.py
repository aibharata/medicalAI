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



from .nnets import resnet,covid_net,densenet,vgg16,mobilenet,mobilenetv2,xception,inceptionv3,inceptionResnet

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
    def __call__(self, inputSize, outputSize, **kwargs):
        """Makes :class:`NetworkInit` instances callable like a function, invoking
        their :meth:`call()` method.
        """
        return self.call(inputSize, outputSize, **kwargs)

    def call(self, inputSize, outputSize, **kwargs):
        """Sample should return model initialized with input and output Sizes.
        
        Parameters
        ----------
        inputSize : tuple or int.
            Integer or tuple specifying the input of network.

        outputSize : tuple or int.
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
    def call(self, inputSize, outputSize, **kwargs):
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
                Dense(outputSize, activation='softmax', name='FC3')
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
                Dense(outputSize, activation='softmax', name='FC3')
            ])            
        return model

class tinyMedNet_v2(NetworkInit):
    """tinyMedNet_v2 allows users to configure the number of Conv/CNN layers.
       Should Pass `convLayers` when initializing.
       tinyMedNet_v2 is a classification network that consumes very less resources and can be trained even on CPUs. This network can be used to demonstrate the framework working.
       Additionally this acts a starting point for example/tutorial for getting started to know the Medical AI library.
    """
    def call(self, inputSize, outputSize, **kwargs):
        try:
            convLayers = kwargs["convLayers"]
        except:
            print('convLayers Not Passed in Network Parameters. Pass the parameter using **ai_params')
            print('Using Default 3 convLayers')
            convLayers = 3
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
            model.add(Dense(outputSize, activation='softmax', name='FC3'))
            return model
        except ValueError:
            print(20*'-')
            print('Dimension Error Occured')
            print('SOLUTION: Try increasing the Input Dimension or Reducing the number of Layers')
            print(20*'-')
            sys.exit(1)
        
class tinyMedNet_v3(NetworkInit):
    """tinyMedNet_v3 has 3 FC layers with Dropout and Configurable number of Conv/CNN Layers.
       Should Pass `convLayers` when initializing.
    """
    def call(self, inputSize, outputSize, **kwargs):
        try:
            convLayers = kwargs["convLayers"]
        except:
            print('convLayers Not Passed in Network Parameters. Pass the parameter using **ai_params')
            print('Using Default 3 convLayers')
            convLayers = 3
        try:
            model = Sequential()
            model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),activation='relu', padding = 'valid',input_shape=inputSize, name='CNN1'))
            model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
            for cnnLayerNum in range(0,convLayers-1):
                model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1),activation='relu', padding = 'valid', name='CNN'+str(cnnLayerNum+2)))
                model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
            model.add(Dropout(rate=0.2))
            model.add(Flatten())
            # model.add(Dense(512, activation='relu', name='FC1'))
            # model.add(Dropout(rate=0.5))
            model.add(Dense(384, activation='relu', name='FC2'))
            model.add(Dropout(rate=0.5))
            model.add(Dense(192, activation='relu', name='FC3'))
            model.add(Dropout(rate=0.5))
            model.add(Dense(outputSize, activation='softmax', name='FC4'))
            return model
        except ValueError as err:
            print(err)
            print(20*'-')
            print('Dimension Error Occured')
            print('SOLUTION: Try increasing the Input Dimension or Reducing the number of Layers')
            print(20*'-')
            sys.exit(1)            
        
class resNet50(NetworkInit):
    """RESNET50
    """
    def call(self, inputSize, outputSize, **kwargs):
        return resnet(img_input=inputSize,classes=outputSize, name='ResNet50', **kwargs)

class resNet50V2(NetworkInit):
    """RESNET50V2
    """
    def call(self, inputSize, outputSize, **kwargs):
        return resnet(img_input=inputSize,classes=outputSize, name='ResNet50V2', **kwargs)

class resNet101(NetworkInit):
    """RESNET101
    """
    def call(self, inputSize, outputSize, **kwargs):
        return resnet(img_input=inputSize,classes=outputSize, name='ResNet101', **kwargs)

class resNet101V2(NetworkInit):
    """RESNET101V2
    """
    def call(self, inputSize, outputSize, **kwargs):
        return resnet(img_input=inputSize,classes=outputSize, name='ResNet101V2', **kwargs)

class resNet152V2(NetworkInit):
    """RESNET152V2
    """
    def call(self, inputSize, outputSize, **kwargs):
        return resnet(img_input=inputSize,classes=outputSize, name='ResNet152V2', **kwargs)

class resNet152(NetworkInit):
    """RESNET152
    """
    def call(self, inputSize, outputSize, **kwargs):
        return resnet(img_input=inputSize,classes=outputSize, name='ResNet152', **kwargs)


class megaNet(NetworkInit):
    """
    megaNet is based on COVID-NET.
    This is a tensorflow 2.0 network variant for COVID-Net described in Paper "COVID-Net: A Tailored Deep Convolutional Neural Network Design for Detection of COVID-19 Cases from Chest Radiography Images" by Linda Wang et al. 
    Reference: https://github.com/busyyang/COVID-19/
    """
    def call(self, inputSize, outputSize, **kwargs):
        return covid_net.COVIDNET_Keras(img_input=inputSize,classes=outputSize)

class DenseNet121(NetworkInit):
    """
    DenseNet121 model, with weights pre-trained on ImageNet
    inputSize: input image size tuple
    outputSize: Number of classes for prediction
    """
    def call(self, inputSize, outputSize, **kwargs):
        return densenet.DenseNet121_Model(img_input=inputSize,classes=outputSize)

class VGG16(NetworkInit):
    """
    VGG16 model, with weights pre-trained on ImageNet
    inputSize: input image size tuple,default : (224,223,3)
    outputSize: Number of classes for prediction
    """
    def call(self, inputSize, outputSize, **kwargs):
        return vgg16.VGG16_Model(img_input=inputSize,classes=outputSize)

class MobileNet(NetworkInit):
    """
    MobileNet model, with weights pre-trained on ImageNet
    inputSize: input image size tuple,default : (224,223,3)
    outputSize: Number of classes for prediction
    """
    def call(self, inputSize, outputSize, **kwargs):
        return mobilenet.MobileNet(img_input=inputSize,classes=outputSize)

class MobileNetV2(NetworkInit):
    """
    MobileNet model, with weights pre-trained on ImageNet
    inputSize: input image size tuple,default : (224,223,3)
    outputSize: Number of classes for prediction
    """
    def call(self, inputSize, outputSize, **kwargs):
        return mobilenetv2.MobileNetV2(img_input=inputSize,classes=outputSize)

class Xception(NetworkInit):
    """
    Xception model, with weights pre-trained on ImageNet
    inputSize: input image size tuple,default : (224,223,3)
    outputSize: Number of classes for prediction
    """
    def call(self, inputSize, outputSize, **kwargs):
        return xception.Xception(img_input=inputSize,classes=outputSize)

class InceptionV3(NetworkInit):
    """
    InceptionV3 model, with weights pre-trained on ImageNet
    inputSize: input image size tuple,default : (224,223,3)
    outputSize: Number of classes for prediction
    """
    def call(self, inputSize, outputSize, **kwargs):
        return inceptionv3.InceptionV3(img_input=inputSize,classes=outputSize)

class InceptionResNetV2(NetworkInit):
    """
    InceptionResNetV2 model, with weights pre-trained on ImageNet
    inputSize: input image size tuple,default : (224,223,3)
    outputSize: Number of classes for prediction
    """
    def call(self, inputSize, outputSize, **kwargs):
        return inceptionResnet.InceptionResNetV2_Model(img_input=inputSize,classes=outputSize)

def get(networkInitialization):
    if networkInitialization.__class__.__name__ == 'str':
        if networkInitialization in ['tinyMedNet', 'tiny_Medical_Network']:
            return tinyMedNet()
        elif networkInitialization in ['tinyMedNet_v2', 'tiny_Medical_Network_v2']:
            return tinyMedNet_v2()
        elif networkInitialization in ['tinyMedNet_v3', 'tiny_Medical_Network_v3']:
            return tinyMedNet_v3()
        elif networkInitialization in ['resNet50', 'resnet50']:
            return resNet50()
        elif networkInitialization in ['resNet50V2', 'resnet50V2']:
            return resNet50V2()
        elif networkInitialization in ['resNet101', 'resnet101']:
            return resNet101()
        elif networkInitialization in ['resNet101V2', 'resnet101V2']:
            return resNet101V2()
        elif networkInitialization in ['resNet152', 'resnet152']:
            return resNet152()
        elif networkInitialization in ['resNet152V2', 'resnet152V2']:
            return resNet152V2()
        elif networkInitialization in ['megaNet', 'meganet']:
            return megaNet()
        elif networkInitialization in ['densenet','DenseNet','DenseNet121']:
            return DenseNet121()
        elif networkInitialization in ['vgg16','VGG16','vgg','VGG']:
            return VGG16()
        elif networkInitialization in ['mobilenet','MobileNet']:
            return MobileNet()
        elif networkInitialization in ['mobilenetv2','MobileNetV2']:
            return MobileNetV2()
        elif networkInitialization in ['xception','Xception']:
            return Xception()
        elif networkInitialization in ['inception','InceptionV3','inceptionv3']:
            return InceptionV3()
        elif networkInitialization in ['inceptionresnet','InceptionResNet','InceptionResNetV2']:
            return InceptionResNetV2()
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