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
import copy
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential

class NetworkInit(object):
    """Base class for parameter Network initializers.

    The :class:`NetworkInit` class represents a network initializer used
    to initialize network/model parameters for numerous medical ai networks. It should be
    subclassed when implementing new types of network initializers.
    """
    def __call__(self, inputSize, OutputSize):
        """Makes :class:`NetworkInit` instances callable like a function, invoking
        their :meth:`call()` method.
        """
        return self.call(inputSize, OutputSize)

    def call(self, inputSize, OutputSize):
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
    """tinyMedNet is a classification network that consumes very less resources and can be trained even on CPUs
    """
    def call(self, inputSize, OutputSize):
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
        return model


def get(networkInitialization):
    if networkInitialization.__class__.__name__ == 'str':
        if networkInitialization in ['tinyMedNet', 'tiny_Medical_Network']:
            return tinyMedNet()
        raise ValueError('Unknown network Initialization name: {}.'.format(networkInitialization))

    elif isinstance(networkInitialization, NetworkInit):
        return copy.deepcopy(networkInitialization)

    else:
        raise ValueError("Unknown type: {}.".format(networkInitialization.__class__.__name__))


if __name__ == "__main__":
    v=get('tinyMedNet')
    m = v((32,32,3),10)
    m.summary()
