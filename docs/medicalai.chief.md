# medicalai.chief package

## Subpackages


* medicalai.chief.model_metrics package


    * Submodules


    * medicalai.chief.model_metrics.modelstats module


    * Module contents


* medicalai.chief.nnets package


    * Submodules


    * medicalai.chief.nnets.covid_net module


    * medicalai.chief.nnets.densenet module


    * medicalai.chief.nnets.inceptionResnet module


    * medicalai.chief.nnets.inceptionv3 module


    * medicalai.chief.nnets.mobilenet module


    * medicalai.chief.nnets.mobilenetv2 module


    * medicalai.chief.nnets.resnet module


    * medicalai.chief.nnets.vgg16 module


    * medicalai.chief.nnets.xception module


    * Module contents


* medicalai.chief.xai package


    * Submodules


    * medicalai.chief.xai.xcams module


    * Module contents


## Submodules

## medicalai.chief.core module


### class medicalai.chief.core.INFERENCE_ENGINE(modelName, testSet=None, classNames=None)
Bases: `object`

TODO: Need to add Metaloader support


#### decode_predictions(pred, top_preds=4, retType='tuple')

#### explain(input, predictions=None, layer_to_explain='CNN3', classNames=None, selectedClasses=None, expectedClass=None, showPlot=False)

#### generate_evaluation_report(testSet=None, predictions=None, printStat=False, returnPlot=False, showPlot=False, pdfName=None, \*\*kwargs)

#### getLayerNames()

#### load_model_and_weights(modelName)

#### load_network(fileName)

#### load_weights(wgtfileName)

#### predict(input)

#### predict_pipeline(input)
Slightly Faster version of predict. Useful for deployment.


#### preprocessor_from_meta(metaFile=None)

#### summary()

### class medicalai.chief.core.TRAIN_ENGINE(modelName=None)
Bases: `medicalai.chief.core.INFERENCE_ENGINE`


#### plot_train_acc_loss()

#### train_and_save_model(AI_NAME, MODEL_SAVE_NAME, trainSet, testSet, OUTPUT_CLASSES, RETRAIN_MODEL, BATCH_SIZE, EPOCHS, LEARNING_RATE, convLayers=None, SAVE_BEST_MODEL=True, BEST_MODEL_COND=None, callbacks=None, loss='sparse_categorical_crossentropy', metrics=['accuracy'], showModel=False, CLASS_WEIGHTS=None)
”
CLASS_WEIGHTS: Dictionary containing class weights for model.fit()


### medicalai.chief.core.check_model_exists(outputName)

### medicalai.chief.core.create_model_output_folder(outputName)

### medicalai.chief.core.decode_predictions(pred, labelNames, top_preds=4, retType='tuple')

### medicalai.chief.core.load_model_and_weights(modelName, summary=False)

### medicalai.chief.core.modelManager(modelName, x_train, OUTPUT_CLASSES, RETRAIN_MODEL, AI_NAME='tinyMedNet', convLayers=None)

### medicalai.chief.core.plot_training_metrics(result, theme='light')

### medicalai.chief.core.predict_labels(model, input, expected_output=None, labelNames=None, top_preds=4)
predict(x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)


### medicalai.chief.core.save_model_and_weights(model, outputName)

### medicalai.chief.core.show_model_details(model)

### medicalai.chief.core.train(model, x_train, batch_size=1, epochs=1, learning_rate=0.001, callbacks=None, class_weights=None, saveBestModel=False, bestModelCond=None, validation_data=None, TRAIN_STEPS=None, TEST_STEPS=None, loss='sparse_categorical_crossentropy', metrics=['accuracy'], verbose=None, y_train=None)
## medicalai.chief.dataset_prepare module


### class medicalai.chief.dataset_prepare.AUGMENTATION(rotation_range=12, fill_mode='constant', width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=False, vertical_flip=False, brightness_range=(0.9, 1.1), zoom_range=(0.85, 1.15), rescale=0.00392156862745098, shear_range=0, channel_shift_range=0, samplewise_center=False, samplewise_std_normalization=False, featurewise_center=False, featurewise_std_normalization=False, cval=0, preprocessing_function=None)
Bases: `object`


#### create_aug()

### class medicalai.chief.dataset_prepare.INPUT_PROCESSOR(targetDim=(31, 31), samplingMethod=None, normalize=False, color_mode='RGB', rescale=None, dtype='float32')
Bases: `object`


#### processImage(image)

#### resizeDataSet(dataset)

#### resizeDataSetfromFolder(folder)

### class medicalai.chief.dataset_prepare.InputProcessorFromMeta(metaFile)
Bases: `medicalai.chief.dataset_prepare.INPUT_PROCESSOR`


### medicalai.chief.dataset_prepare.convertlist2tuple(lst)

### medicalai.chief.dataset_prepare.datasetFolderStructureValidate(folder)

### class medicalai.chief.dataset_prepare.datasetFromFolder(folder, targetDim=(31, 31), normalize=False, name=None, useCache=True, forceCleanCache=False)
Bases: `medicalai.chief.dataset_prepare.datasetManager`

TODO: Fix samplingMethodName assignment


#### load_dataset()

### class medicalai.chief.dataset_prepare.datasetGenFromDataframe(folder, csv_path='.', x_col='name', y_col='labels', targetDim=(224, 224), normalize=False, batch_size=16, augmentation=True, color_mode='rgb', class_mode='sparse', shuffle=True, seed=17)
Bases: `object`

Creates Keras Dataset Generator for Handling Large Datasets from DataFrame.


* **Parameters**

    
    * **csv_path** – folder containing train.csv and test.csv.


    * **folder** – The directory must be set to the path where your training images are present.


    * **x_col** – Name of column containing image name, default = name.


    * **y_col** – Name of column for labels, default = labels.


    * **targetDim** – The target_size is the size of your input images to the neural network.


    * **class_mode** – Set binary if classifying only two classes, if not set to categorical, in case of an Autoencoder system, both input and the output would probably be the same image, for this case set to input.


    * **color_mode** – grayscale for black and white or grayscale, rgb for three color channels.


    * **batch_size** – Number of images to be yielded from the generator per batch. If training fails lower this number.



#### get_class_weights()

#### get_numpy(generator)

#### load_generator()

### class medicalai.chief.dataset_prepare.datasetGenFromFolder(folder, targetDim=(224, 224), normalize=False, batch_size=16, augmentation=True, color_mode='rgb', class_mode='sparse', shuffle=True, seed=17)
Bases: `object`

folder : The directory must be set to the path where your n classes of folders are present.
targetDim : The target_size is the size of your input images to the neural network.
class_mode : Set binary if classifying only two classes, if not set to categorical, in case of an Autoencoder system, both input and the output would probably be the same image, for this case set to input.
color_mode: grayscale for black and white or grayscale, rgb for three color channels.
batch_size: Number of images to be yielded from the generator per batch. If training fails lower this number.


#### get_class_weights()

#### get_numpy(generator)

#### load_generator()

### class medicalai.chief.dataset_prepare.datasetManager(folder, targetDim=(31, 31), normalize=False, name=None, useCache=True, forceCleanCache=False)
Bases: `medicalai.chief.dataset_prepare.INPUT_PROCESSOR`


#### compress_and_cache_data(\*\*kw)

#### convert_dataset(\*\*kw)

#### load_data()

#### process_dataset()

#### reload_data(\*\*kw)

### medicalai.chief.dataset_prepare.datasetManagerFunc(folder, targetDim=(31, 31), normalize=False)

### medicalai.chief.dataset_prepare.getLabelsFromFolder(folder)

### class medicalai.chief.dataset_prepare.medicalai_generator()
Bases: `tensorflow.python.keras.preprocessing.image.ImageDataGenerator`


### medicalai.chief.dataset_prepare.metaLoader(metaFile)

### medicalai.chief.dataset_prepare.metaSaver(labelMap, labels, normalize=None, rescale=None, network_input_dim=None, samplingMethodName=None, outputName=None)

### class medicalai.chief.dataset_prepare.myDict()
Bases: `dict`


### medicalai.chief.dataset_prepare.safe_labelmap_converter(labelMap)
## medicalai.chief.download_utils module


### class medicalai.chief.download_utils.DLProgress(iterable=None, desc=None, total=None, leave=True, file=None, ncols=None, mininterval=0.1, maxinterval=10.0, miniters=None, ascii=None, disable=False, unit='it', unit_scale=False, dynamic_ncols=False, smoothing=0.3, bar_format=None, initial=0, position=None, postfix=None, unit_divisor=1000, write_bytes=None, gui=False, \*\*kwargs)
Bases: `tqdm._tqdm.tqdm`


#### hook(block_num=1, block_size=1, total_size=None)

#### last_block( = 0)

### medicalai.chief.download_utils.check_if_url(x)

### medicalai.chief.download_utils.getFile(url, storePath=None, cacheDir=None, subDir='dataset')

### medicalai.chief.download_utils.load_image(link, target_size=(32, 32), storePath=None, cacheDir=None, subDir='images')

### medicalai.chief.download_utils.untar(tar_file, destination)

### medicalai.chief.download_utils.unzip(zip_file, destination)
## medicalai.chief.networks module


### class medicalai.chief.networks.DenseNet121()
Bases: `medicalai.chief.networks.NetworkInit`

DenseNet121 model, with weights pre-trained on ImageNet
inputSize: input image size tuple
outputSize: Number of classes for prediction


#### call(inputSize, OutputSize, convLayers=None)
Sample should return model initialized with input and output Sizes.


* **Parameters**

    
    * **inputSize** (*tuple** or **int.*) – Integer or tuple specifying the input of network.


    * **OutputSize** (*tuple** or **int.*) – Integer or tuple specifying the output classes of network.



* **Returns**

    Initialized Model.



* **Return type**

    numpy.array.



### class medicalai.chief.networks.InceptionResNetV2()
Bases: `medicalai.chief.networks.NetworkInit`

InceptionResNetV2 model, with weights pre-trained on ImageNet
inputSize: input image size tuple,default : (224,223,3)
outputSize: Number of classes for prediction


#### call(inputSize, OutputSize, convLayers=None)
Sample should return model initialized with input and output Sizes.


* **Parameters**

    
    * **inputSize** (*tuple** or **int.*) – Integer or tuple specifying the input of network.


    * **OutputSize** (*tuple** or **int.*) – Integer or tuple specifying the output classes of network.



* **Returns**

    Initialized Model.



* **Return type**

    numpy.array.



### class medicalai.chief.networks.InceptionV3()
Bases: `medicalai.chief.networks.NetworkInit`

InceptionV3 model, with weights pre-trained on ImageNet
inputSize: input image size tuple,default : (224,223,3)
outputSize: Number of classes for prediction


#### call(inputSize, OutputSize, convLayers=None)
Sample should return model initialized with input and output Sizes.


* **Parameters**

    
    * **inputSize** (*tuple** or **int.*) – Integer or tuple specifying the input of network.


    * **OutputSize** (*tuple** or **int.*) – Integer or tuple specifying the output classes of network.



* **Returns**

    Initialized Model.



* **Return type**

    numpy.array.



### class medicalai.chief.networks.MobileNet()
Bases: `medicalai.chief.networks.NetworkInit`

MobileNet model, with weights pre-trained on ImageNet
inputSize: input image size tuple,default : (224,223,3)
outputSize: Number of classes for prediction


#### call(inputSize, OutputSize, convLayers=None)
Sample should return model initialized with input and output Sizes.


* **Parameters**

    
    * **inputSize** (*tuple** or **int.*) – Integer or tuple specifying the input of network.


    * **OutputSize** (*tuple** or **int.*) – Integer or tuple specifying the output classes of network.



* **Returns**

    Initialized Model.



* **Return type**

    numpy.array.



### class medicalai.chief.networks.MobileNetV2()
Bases: `medicalai.chief.networks.NetworkInit`

MobileNet model, with weights pre-trained on ImageNet
inputSize: input image size tuple,default : (224,223,3)
outputSize: Number of classes for prediction


#### call(inputSize, OutputSize, convLayers=None)
Sample should return model initialized with input and output Sizes.


* **Parameters**

    
    * **inputSize** (*tuple** or **int.*) – Integer or tuple specifying the input of network.


    * **OutputSize** (*tuple** or **int.*) – Integer or tuple specifying the output classes of network.



* **Returns**

    Initialized Model.



* **Return type**

    numpy.array.



### class medicalai.chief.networks.NetworkInit()
Bases: `object`

Base class for parameter Network initializers.

The `NetworkInit` class represents a network initializer used
to initialize network/model parameters for numerous medical ai networks. It should be
subclassed when implementing new types of network initializers.


#### call(inputSize, OutputSize, convLayers=None)
Sample should return model initialized with input and output Sizes.


* **Parameters**

    
    * **inputSize** (*tuple** or **int.*) – Integer or tuple specifying the input of network.


    * **OutputSize** (*tuple** or **int.*) – Integer or tuple specifying the output classes of network.



* **Returns**

    Initialized Model.



* **Return type**

    numpy.array.



### class medicalai.chief.networks.VGG16()
Bases: `medicalai.chief.networks.NetworkInit`

VGG16 model, with weights pre-trained on ImageNet
inputSize: input image size tuple,default : (224,223,3)
outputSize: Number of classes for prediction


#### call(inputSize, OutputSize, convLayers=None)
Sample should return model initialized with input and output Sizes.


* **Parameters**

    
    * **inputSize** (*tuple** or **int.*) – Integer or tuple specifying the input of network.


    * **OutputSize** (*tuple** or **int.*) – Integer or tuple specifying the output classes of network.



* **Returns**

    Initialized Model.



* **Return type**

    numpy.array.



### class medicalai.chief.networks.Xception()
Bases: `medicalai.chief.networks.NetworkInit`

Xception model, with weights pre-trained on ImageNet
inputSize: input image size tuple,default : (224,223,3)
outputSize: Number of classes for prediction


#### call(inputSize, OutputSize, convLayers=None)
Sample should return model initialized with input and output Sizes.


* **Parameters**

    
    * **inputSize** (*tuple** or **int.*) – Integer or tuple specifying the input of network.


    * **OutputSize** (*tuple** or **int.*) – Integer or tuple specifying the output classes of network.



* **Returns**

    Initialized Model.



* **Return type**

    numpy.array.



### medicalai.chief.networks.get(networkInitialization)

### class medicalai.chief.networks.megaNet()
Bases: `medicalai.chief.networks.NetworkInit`

megaNet is based on COVID-NET.
This is a tensorflow 2.0 network variant for COVID-Net described in Paper “COVID-Net: A Tailored Deep Convolutional Neural Network Design for Detection of COVID-19 Cases from Chest Radiography Images” by Linda Wang et al.
Reference: [https://github.com/busyyang/COVID-19/](https://github.com/busyyang/COVID-19/)


#### call(inputSize, OutputSize, convLayers=None)
Sample should return model initialized with input and output Sizes.


* **Parameters**

    
    * **inputSize** (*tuple** or **int.*) – Integer or tuple specifying the input of network.


    * **OutputSize** (*tuple** or **int.*) – Integer or tuple specifying the output classes of network.



* **Returns**

    Initialized Model.



* **Return type**

    numpy.array.



### class medicalai.chief.networks.resNet110()
Bases: `medicalai.chief.networks.NetworkInit`

resnet110


#### call(inputSize, OutputSize, convLayers=None)
Sample should return model initialized with input and output Sizes.


* **Parameters**

    
    * **inputSize** (*tuple** or **int.*) – Integer or tuple specifying the input of network.


    * **OutputSize** (*tuple** or **int.*) – Integer or tuple specifying the output classes of network.



* **Returns**

    Initialized Model.



* **Return type**

    numpy.array.



### class medicalai.chief.networks.resNet20()
Bases: `medicalai.chief.networks.NetworkInit`

resnet20


#### call(inputSize, OutputSize, convLayers=None)
Sample should return model initialized with input and output Sizes.


* **Parameters**

    
    * **inputSize** (*tuple** or **int.*) – Integer or tuple specifying the input of network.


    * **OutputSize** (*tuple** or **int.*) – Integer or tuple specifying the output classes of network.



* **Returns**

    Initialized Model.



* **Return type**

    numpy.array.



### class medicalai.chief.networks.resNet32()
Bases: `medicalai.chief.networks.NetworkInit`

resnet32


#### call(inputSize, OutputSize, convLayers=None)
Sample should return model initialized with input and output Sizes.


* **Parameters**

    
    * **inputSize** (*tuple** or **int.*) – Integer or tuple specifying the input of network.


    * **OutputSize** (*tuple** or **int.*) – Integer or tuple specifying the output classes of network.



* **Returns**

    Initialized Model.



* **Return type**

    numpy.array.



### class medicalai.chief.networks.resNet56()
Bases: `medicalai.chief.networks.NetworkInit`

RESNET56


#### call(inputSize, OutputSize, convLayers=None)
Sample should return model initialized with input and output Sizes.


* **Parameters**

    
    * **inputSize** (*tuple** or **int.*) – Integer or tuple specifying the input of network.


    * **OutputSize** (*tuple** or **int.*) – Integer or tuple specifying the output classes of network.



* **Returns**

    Initialized Model.



* **Return type**

    numpy.array.



### class medicalai.chief.networks.tinyMedNet()
Bases: `medicalai.chief.networks.NetworkInit`

tinyMedNet is a classification network that consumes very less resources and can be trained even on CPUs. This network can be used to demonstrate the framework working.
Additionally this acts a starting point for example/tutorial for getting started to know the Medical AI library.


#### call(inputSize, OutputSize, convLayers=None)
Sample should return model initialized with input and output Sizes.


* **Parameters**

    
    * **inputSize** (*tuple** or **int.*) – Integer or tuple specifying the input of network.


    * **OutputSize** (*tuple** or **int.*) – Integer or tuple specifying the output classes of network.



* **Returns**

    Initialized Model.



* **Return type**

    numpy.array.



### class medicalai.chief.networks.tinyMedNet_v2()
Bases: `medicalai.chief.networks.NetworkInit`

tinyMedNet_v2 allows users to configure the number of Conv/CNN layers.
tinyMedNet_v2 is a classification network that consumes very less resources and can be trained even on CPUs. This network can be used to demonstrate the framework working.
Additionally this acts a starting point for example/tutorial for getting started to know the Medical AI library.


#### call(inputSize, OutputSize, convLayers=2)
Sample should return model initialized with input and output Sizes.


* **Parameters**

    
    * **inputSize** (*tuple** or **int.*) – Integer or tuple specifying the input of network.


    * **OutputSize** (*tuple** or **int.*) – Integer or tuple specifying the output classes of network.



* **Returns**

    Initialized Model.



* **Return type**

    numpy.array.



### class medicalai.chief.networks.tinyMedNet_v3()
Bases: `medicalai.chief.networks.NetworkInit`

tinyMedNet_v3 has 3 FC layers with Dropout and Configurable number of Conv/CNN Layers.


#### call(inputSize, OutputSize, convLayers=2)
Sample should return model initialized with input and output Sizes.


* **Parameters**

    
    * **inputSize** (*tuple** or **int.*) – Integer or tuple specifying the input of network.


    * **OutputSize** (*tuple** or **int.*) – Integer or tuple specifying the output classes of network.



* **Returns**

    Initialized Model.



* **Return type**

    numpy.array.


## medicalai.chief.prettyloss module


### class medicalai.chief.prettyloss.prettyLoss(show_percentage=False)
Bases: `object`


#### STYLE( = {'bold': '\\x1b[1m', 'green': '\\x1b[32m', 'red': '\\x1b[91m'})

#### STYLE_END( = '\\x1b[0m')
## medicalai.chief.uFuncs module


### medicalai.chief.uFuncs.timeit(func)
## Module contents
