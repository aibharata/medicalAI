# medicalai.chief.nnets package

## Submodules

## medicalai.chief.nnets.covid_net module


### medicalai.chief.nnets.covid_net.COVIDNET_Keras(img_input=(224, 224, 3), classes=4)
This is a tensorflow 2.0 network variant for COVID-Net described in Paper “COVID-Net: A Tailored Deep Convolutional Neural Network Design for Detection of COVID-19 Cases from Chest Radiography Images” by Linda Wang et al.
Reference: [https://github.com/busyyang/COVID-19/](https://github.com/busyyang/COVID-19/)


### medicalai.chief.nnets.covid_net.PEPXModel(input_tensor, filters, name)
## medicalai.chief.nnets.densenet module


### medicalai.chief.nnets.densenet.DenseNet121_Model(img_input=(224, 224, 3), classes=3)
Loaded the DenseNet121 network, ensuring the head FC layer sets are left off


* **Parameters**

    
    * **img_input** – optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (224, 224, 3)
    (with ‘channels_last’ data format) or (3, 224, 224) (with ‘channels_first’ data format). It should have exactly 3 inputs channels,
    and width and height should be no smaller than 32. E.g. (200, 200, 3) would be one valid value.


    * **classes** – Number of classes to be predicted.


    * **Returns** – model


## medicalai.chief.nnets.inceptionResnet module


### medicalai.chief.nnets.inceptionResnet.InceptionResNetV2_Model(img_input=(224, 224, 3), classes=3)
Loaded the InceptionResNetV2 network, ensuring the head FC layer sets are left off


* **Parameters**

    
    * **img_input** – optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (224, 224, 3)
    (with ‘channels_last’ data format) or (3, 224, 224) (with ‘channels_first’ data format). It should have exactly 3 inputs channels,
    and width and height should be no smaller than 32. E.g. (200, 200, 3) would be one valid value.


    * **classes** – Number of classes to be predicted.


    * **Returns** – model


## medicalai.chief.nnets.inceptionv3 module


### medicalai.chief.nnets.inceptionv3.InceptionV3(img_input=(224, 224, 3), classes=3)
Loaded the InceptionV3 network, ensuring the head FC layer sets are left off


* **Parameters**

    
    * **img_input** – optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (224, 224, 3)
    (with ‘channels_last’ data format) or (3, 224, 224) (with ‘channels_first’ data format). It should have exactly 3 inputs channels,
    and width and height should be no smaller than 32. E.g. (200, 200, 3) would be one valid value.


    * **classes** – Number of classes to be predicted.


    * **Returns** – model


## medicalai.chief.nnets.mobilenet module


### medicalai.chief.nnets.mobilenet.MobileNet(img_input=(224, 224, 3), classes=3)
Loaded the MobileNet network, ensuring the head FC layer sets are left off


* **Parameters**

    
    * **img_input** – optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (224, 224, 3)
    (with ‘channels_last’ data format) or (3, 224, 224) (with ‘channels_first’ data format). It should have exactly 3 inputs channels,
    and width and height should be no smaller than 32. E.g. (200, 200, 3) would be one valid value.


    * **classes** – Number of classes to be predicted.


    * **Returns** – model


## medicalai.chief.nnets.mobilenetv2 module


### medicalai.chief.nnets.mobilenetv2.MobileNetV2(img_input=(224, 224, 3), classes=3)
Loaded the MobileNetV2 network, ensuring the head FC layer sets are left off


* **Parameters**

    
    * **img_input** – optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (224, 224, 3)
    (with ‘channels_last’ data format) or (3, 224, 224) (with ‘channels_first’ data format). It should have exactly 3 inputs channels,
    and width and height should be no smaller than 32. E.g. (200, 200, 3) would be one valid value.


    * **classes** – Number of classes to be predicted.


    * **Returns** – model


## medicalai.chief.nnets.resnet module


### medicalai.chief.nnets.resnet.conv_building_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), training=None)
A block that has a conv layer at shortcut.


* **Parameters**

    
    * **input_tensor** – input tensor


    * **kernel_size** – default 3, the kernel size of
    middle conv layer at main path


    * **filters** – list of integers, the filters of 3 conv layer at main path


    * **stage** – integer, current stage label, used for generating layer names


    * **block** – current block label, used for generating layer names


    * **strides** – Strides for the first conv layer in the block.


    * **training** – Only used if training keras model with Estimator.  In other
    scenarios it is handled automatically.



* **Returns**

    Output tensor for the block.


Note that from stage 3,
the first conv layer at main path is with strides=(2, 2)
And the shortcut should have strides=(2, 2) as well


### medicalai.chief.nnets.resnet.identity_building_block(input_tensor, kernel_size, filters, stage, block, training=None)
The identity block is the block that has no conv layer at shortcut.


* **Parameters**

    
    * **input_tensor** – input tensor


    * **kernel_size** – default 3, the kernel size of
    middle conv layer at main path


    * **filters** – list of integers, the filters of 3 conv layer at main path


    * **stage** – integer, current stage label, used for generating layer names


    * **block** – current block label, used for generating layer names


    * **training** – Only used if training keras model with Estimator.  In other
    scenarios it is handled automatically.



* **Returns**

    Output tensor for the block.



### medicalai.chief.nnets.resnet.resnet(num_blocks, img_input=None, classes=10, training=None)
Instantiates the ResNet architecture.


* **Parameters**

    
    * **num_blocks** – integer, the number of conv/identity blocks in each block.
    The ResNet contains 3 blocks with each block containing one conv block
    followed by (layers_per_block - 1) number of idenity blocks. Each
    conv/idenity block has 2 convolutional layers. With the input
    convolutional layer and the pooling layer towards the end, this brings
    the total size of the network to (6\*num_blocks + 2)


    * **classes** – optional number of classes to classify images into


    * **training** – Only used if training keras model with Estimator.  In other


    * **it is handled automatically.** (*scenarios*) – 



* **Returns**

    A Keras model instance.



### medicalai.chief.nnets.resnet.resnet110(\*, num_blocks=110, img_input=None, classes=10, training=None)
Instantiates the ResNet architecture.


* **Parameters**

    
    * **num_blocks** – integer, the number of conv/identity blocks in each block.
    The ResNet contains 3 blocks with each block containing one conv block
    followed by (layers_per_block - 1) number of idenity blocks. Each
    conv/idenity block has 2 convolutional layers. With the input
    convolutional layer and the pooling layer towards the end, this brings
    the total size of the network to (6\*num_blocks + 2)


    * **classes** – optional number of classes to classify images into


    * **training** – Only used if training keras model with Estimator.  In other


    * **it is handled automatically.** (*scenarios*) – 



* **Returns**

    A Keras model instance.



### medicalai.chief.nnets.resnet.resnet20(\*, num_blocks=3, img_input=None, classes=10, training=None)
Instantiates the ResNet architecture.


* **Parameters**

    
    * **num_blocks** – integer, the number of conv/identity blocks in each block.
    The ResNet contains 3 blocks with each block containing one conv block
    followed by (layers_per_block - 1) number of idenity blocks. Each
    conv/idenity block has 2 convolutional layers. With the input
    convolutional layer and the pooling layer towards the end, this brings
    the total size of the network to (6\*num_blocks + 2)


    * **classes** – optional number of classes to classify images into


    * **training** – Only used if training keras model with Estimator.  In other


    * **it is handled automatically.** (*scenarios*) – 



* **Returns**

    A Keras model instance.



### medicalai.chief.nnets.resnet.resnet32(\*, num_blocks=5, img_input=None, classes=10, training=None)
Instantiates the ResNet architecture.


* **Parameters**

    
    * **num_blocks** – integer, the number of conv/identity blocks in each block.
    The ResNet contains 3 blocks with each block containing one conv block
    followed by (layers_per_block - 1) number of idenity blocks. Each
    conv/idenity block has 2 convolutional layers. With the input
    convolutional layer and the pooling layer towards the end, this brings
    the total size of the network to (6\*num_blocks + 2)


    * **classes** – optional number of classes to classify images into


    * **training** – Only used if training keras model with Estimator.  In other


    * **it is handled automatically.** (*scenarios*) – 



* **Returns**

    A Keras model instance.



### medicalai.chief.nnets.resnet.resnet56(\*, num_blocks=9, img_input=None, classes=10, training=None)
Instantiates the ResNet architecture.


* **Parameters**

    
    * **num_blocks** – integer, the number of conv/identity blocks in each block.
    The ResNet contains 3 blocks with each block containing one conv block
    followed by (layers_per_block - 1) number of idenity blocks. Each
    conv/idenity block has 2 convolutional layers. With the input
    convolutional layer and the pooling layer towards the end, this brings
    the total size of the network to (6\*num_blocks + 2)


    * **classes** – optional number of classes to classify images into


    * **training** – Only used if training keras model with Estimator.  In other


    * **it is handled automatically.** (*scenarios*) – 



* **Returns**

    A Keras model instance.



### medicalai.chief.nnets.resnet.resnet_block(input_tensor, size, kernel_size, filters, stage, conv_strides=(2, 2), training=None)
A block which applies conv followed by multiple identity blocks.


* **Parameters**

    
    * **input_tensor** – input tensor


    * **size** – integer, number of constituent conv/identity building blocks.


    * **conv block is applied once****, ****followed by** (*A*) – 


    * **kernel_size** – default 3, the kernel size of
    middle conv layer at main path


    * **filters** – list of integers, the filters of 3 conv layer at main path


    * **stage** – integer, current stage label, used for generating layer names


    * **conv_strides** – Strides for the first conv layer in the block.


    * **training** – Only used if training keras model with Estimator.  In other
    scenarios it is handled automatically.



* **Returns**

    Output tensor after applying conv and identity blocks.


## medicalai.chief.nnets.vgg16 module


### medicalai.chief.nnets.vgg16.VGG16_Model(img_input=(224, 224, 3), classes=3)
Loaded the VGG16 network, ensuring the head FC layer sets are left off


* **Parameters**

    
    * **img_input** – optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (224, 224, 3)
    (with ‘channels_last’ data format) or (3, 224, 224) (with ‘channels_first’ data format). It should have exactly 3 inputs channels,
    and width and height should be no smaller than 32. E.g. (200, 200, 3) would be one valid value.


    * **classes** – Number of classes to be predicted.


    * **Returns** – model


## medicalai.chief.nnets.xception module


### medicalai.chief.nnets.xception.Xception(img_input=(224, 224, 3), classes=3)
Loaded the Xception network, ensuring the head FC layer sets are left off


* **Parameters**

    
    * **img_input** – optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (224, 224, 3)
    (with ‘channels_last’ data format) or (3, 224, 224) (with ‘channels_first’ data format). It should have exactly 3 inputs channels,
    and width and height should be no smaller than 32. E.g. (200, 200, 3) would be one valid value.


    * **classes** – Number of classes to be predicted.


    * **Returns** – model


## Module contents
