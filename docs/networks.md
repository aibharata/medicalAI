
# medicalai.chief.networks


## NetworkInit
```python
NetworkInit()
```
Base class for parameter Network initializers.

The :class:`NetworkInit` class represents a network initializer used
to initialize network/model parameters for numerous medical ai networks. It should be
subclassed when implementing new types of network initializers.


### call
```python
NetworkInit.call(inputSize, OutputSize, convLayers=None)
```
Sample should return model initialized with input and output Sizes.

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


## tinyMedNet
```python
tinyMedNet()
```
tinyMedNet is a classification network that consumes very less resources and can be trained even on CPUs. This network can be used to demonstrate the framework working.
Additionally this acts a starting point for example/tutorial for getting started to know the Medical AI library.


## tinyMedNet_v2
```python
tinyMedNet_v2()
```
tinyMedNet_v2 allows users to configure the number of Conv/CNN layers.
tinyMedNet_v2 is a classification network that consumes very less resources and can be trained even on CPUs. This network can be used to demonstrate the framework working.
Additionally this acts a starting point for example/tutorial for getting started to know the Medical AI library.


## tinyMedNet_v3
```python
tinyMedNet_v3()
```
tinyMedNet_v3 has 3 FC layers with Dropout and Configurable number of Conv/CNN Layers.


## resNet20
```python
resNet20()
```
resnet20


## resNet32
```python
resNet32()
```
resnet32


## resNet56
```python
resNet56()
```
RESNET56


## resNet110
```python
resNet110()
```
resnet110


## megaNet
```python
megaNet()
```

megaNet is based on COVID-NET.
This is a tensorflow 2.0 network variant for COVID-Net described in Paper "COVID-Net: A Tailored Deep Convolutional Neural Network Design for Detection of COVID-19 Cases from Chest Radiography Images" by Linda Wang et al.
Reference: https://github.com/busyyang/COVID-19/


## DenseNet121
```python
DenseNet121()
```

DenseNet121 model, with weights pre-trained on ImageNet
inputSize: input image size tuple
outputSize: Number of classes for prediction


## VGG16
```python
VGG16()
```

VGG16 model, with weights pre-trained on ImageNet
inputSize: input image size tuple,default : (224,223,3)
outputSize: Number of classes for prediction


## MobileNet
```python
MobileNet()
```

MobileNet model, with weights pre-trained on ImageNet
inputSize: input image size tuple,default : (224,223,3)
outputSize: Number of classes for prediction


## MobileNetV2
```python
MobileNetV2()
```

MobileNet model, with weights pre-trained on ImageNet
inputSize: input image size tuple,default : (224,223,3)
outputSize: Number of classes for prediction


## Xception
```python
Xception()
```

Xception model, with weights pre-trained on ImageNet
inputSize: input image size tuple,default : (224,223,3)
outputSize: Number of classes for prediction


## InceptionV3
```python
InceptionV3()
```

InceptionV3 model, with weights pre-trained on ImageNet
inputSize: input image size tuple,default : (224,223,3)
outputSize: Number of classes for prediction


## InceptionResNetV2
```python
InceptionResNetV2()
```

InceptionResNetV2 model, with weights pre-trained on ImageNet
inputSize: input image size tuple,default : (224,223,3)
outputSize: Number of classes for prediction

