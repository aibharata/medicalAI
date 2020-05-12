
# medicalai.chief.core


## create_model_output_folder
```python
create_model_output_folder(outputName)
```

Creates model output folder if model doesn't exist.

__Arguments__

- __outputName__: (Type - `filepath`): name of the folder where model needs to be created.

__Returns__

`None`: None



## check_model_exists
```python
check_model_exists(outputName)
```

Checks if the given model's network file exists or not.
Model name expected is `modelName + _arch.json`.

__Arguments__

- __outputName__: (Type - `filepath`): model name to check.

__Returns__

`Bool`: If model network exists returns `True` else `False`.



## save_model_and_weights
```python
save_model_and_weights(model, outputName)
```

Saves the passed model to MedicalAI Format. Accepts a model and converts to MedicalAI Format.
Produces weight file (`outputName + _wgts.h5`) and network file (`outputName + _arch.json`)

!!! danger "IMPORTANT"
 DO NOT PASS ANY EXTENTION TO `outputName` argument

__Arguments__

- __model__: (Type - `model` class): MedicalAI/Keras/Tensorflow 2.0+ model class.
- __outputName__: (Type - `filepath`): model path/name to save.

__Returns__

`None`: None



## load_model_and_weights
```python
load_model_and_weights(modelName, summary=False)
```

Loads model from the given filepath.
Function Expects weight file (`modelName + _wgts.h5`) and network file (`modelName + _arch.json`).

!!! danger "NOTE"
 DO NOT PASS ANY EXTENTION TO `outputName` argument

For Example:
```Python
# If Model files are `devModel/testmodel1_wgts.h5` and `devModel/testmodel1_arch.json`
# Then `modelName=devModel/testmodel1`

modelName = 'devModel/testmodel1'
load_model_and_weights(modelName, summary = False)

load_model_and_weights(modelName='devModel/testmodel1')

load_model_and_weights('devModel/testmodel1', summary = True)
```


__Arguments__

- __modelName__: (Type - `filepath`): model path/name to save.
- __summary__: (Type - `Bool`): Show loaded network architecture and parameter summary.

__Returns__

`model`: (Type - `model` class): MedicalAI/Keras/Tensorflow 2.0+ model class.



## modelManager
```python
modelManager(modelName,
             x_train,
             OUTPUT_CLASSES,
             RETRAIN_MODEL,
             AI_NAME='tinyMedNet',
             convLayers=None)
```

Model manager is used to build new model for given networks/AI or reload existing AI model.
This function can be used to retrain existing models or create new models.


!!! danger "IMPORTANT"
 DO NOT PASS ANY EXTENTION TO `modelName` argument

__Arguments__

- __modelName__: (Type - `filepath`): model path/name to load existing model or create new model.
- __x_train__: (Type - `numpy.array`): training dataset - expected shape [num_samples*dimension_of_input].
- __OUTPUT_CLASSES__: (Type - `Int`): Number of unique classes in dataset.
- __RETRAIN_MODEL__: (Type - `Bool`): Whether to retrain existing model. If set to `True` and model does not
         exist, then it creates a new model and subsequent runs will retrain model.
- __AI_NAME__: (Type - `String` or `Custom Network Class`): Select AI Networks from existing catalogue in MedicalAI.
         See AI_NAME Page for More Details.
- __convLayers__: (Type - `Int`): [Optional] Default is None. Only applicable for certain networks where convolution
         layers are reconfigurable. This parameter can be used to change the num of conv
         layers in Network. See AI_NAME Page for More Details.

__Returns__

`model`: (Type - `model` class): MedicalAI/Keras/Tensorflow 2.0+ model class.

See Also:
 TRAIN_ENGINE, INFERENCE_ENGINE


## show_model_details
```python
show_model_details(model)
```

Show model network structure and print parameters summary.

__Arguments__

- __model__: (Type - `model` class): MedicalAI/Keras/Tensorflow 2.0+ model class.

__Returns__

`None`: None; Prints the model summary


## predict_labels
```python
predict_labels(model,
               input,
               expected_output=None,
               labelNames=None,
               top_preds=4)
```

```Python
predict_labels(model , input, expected_output = expected_output, labelNames=classNames,top_preds=4)
```


## INFERENCE_ENGINE
```python
INFERENCE_ENGINE(self, modelName=None, testSet=None, classNames=None)
```

Initializes Inference Engine to perform inference/prediction on a trained model.
Can be used during production.

__Arguments__

- __modelName__: (Type - `filepath`): model path/name to load existing model or create new model.
- __testSet__: (Type - `numpy.array` or `generator`): [Optional] : Test/Validation Dataset either as generator
         or numpy array. Only passed if performing evaluation. No need to set
         this during production.
- __classNames__: (Type - `list` or `numpy.array`): [Optional] : classNames or labelNames for the dataset.

__Returns__

`INFERENCE_ENGINE Object`: If `modelName` is supplied, returns an object with loaded model.



### load_model_and_weights
```python
INFERENCE_ENGINE.load_model_and_weights(modelName, summary=False)
```

Loads model from the given filepath.
Function Expects path to weight file (`modelName + _wgts.h5`) and network file (`modelName + _arch.json`).

!!! info "NOTE"
 You can use `load_network` and `load_weights` if the model files are in MedicalAI Format.

!!! danger "WARNING"
 DO NOT PASS ANY EXTENTION TO `outputName` argument

For Example:
```Python
# If Model files are `devModel/testmodel1_wgts.h5` and `devModel/testmodel1_arch.json`
# Then `modelName=devModel/testmodel1`

modelName = 'devModel/testmodel1'
infEngine = INFERENCE_ENGINE()
infEngine.load_model_and_weights(modelName)

infEngine.load_model_and_weights(modelName, summary = True)
```


__Arguments__

- __modelName__: (Type - `filepath`): model path/name to load.
- __summary__: (Type - `Bool`): [Optional] : `Default = False`. Show loaded network architecture
        and parameter summary.

__Returns__

`None`: Intializes Object with model.



### load_network
```python
INFERENCE_ENGINE.load_network(fileName)
```

Loads network from given filepath. Function Expects path to network file with `.json` extension.

!!! info "NOTE"
 Use this function only if the model files are not in MedicalAI Format.

Example:
```Python

networkFile = 'devModel/testmodel1.json'

infEngine = INFERENCE_ENGINE()
infEngine.load_network(networkFile)
```


__Arguments__

- __modelName__: (Type - `filepath`): model network path/name to load. File should have `.json` extension.

__Returns__

`None`: Intializes Object with model network initialized. After this model weights can be loaded.



### load_weights
```python
INFERENCE_ENGINE.load_weights(wgtfileName)
```

Loads weight from given filepath. Function Expects path to weight file with `.h5` extension.

!!! danger "NOTE"
 Use this function only if the model files are not in MedicalAI Format.
 Before calling this function, network needs to loaded using `load_network` function.

Example:
```Python

networkFile = 'devModel/testmodel1.json'
wgtFile = 'devModel/testmodel1.h5'

infEngine = INFERENCE_ENGINE()
infEngine.load_network(networkFile)
infEngine.load_weights(wgtFile)
```


__Arguments__

- __wgtfileName__: (Type - `filepath`): model weight filepath/name to load. File should have `.h5` extension.

__Returns__

`None`: Intializes Object with model loaded with weights.


### preprocessor_from_meta
```python
INFERENCE_ENGINE.preprocessor_from_meta(metaFile=None)
```

Loads preprocessor parameter and initializes preprocessor from meta file generated by MedicalAI.

If the model is trained using this framework, then the metafile is automatically available and initialized.

!!! danger "WARNING"
 If model is not trained using this framework, then one can use this engine by creating metafile
 similar to one generated by this framework. Please see repo for more details.

Example:
```Python
# If Model files are `devModel/testmodel1_wgts.h5` and `devModel/testmodel1_arch.json`
# Then `modelName=devModel/testmodel1`

modelName = 'devModel/testmodel1'
infEngine = INFERENCE_ENGINE()
infEngine.load_model_and_weights(modelName)

# There is no need to perform this op if model trained using this framework. It is automatically Initialized.
# There is no need to pass modelName if the model is trained using framework
infEngine.preprocessor_from_meta()

infEngine.preprocessor_from_meta(metaFile='myMetaFile.json') [`Else`](#Else) pass the metafile
```


__Arguments__

- __metaFile__: (Type - `filepath`): [Optional] : if no parameter is passed, then it will look for
         `modelname + _meta.json` file. If modelname is set during
         INFERENCE_ENGINE initialization, then it automatically handles this.

__Returns__

`None`: Intializes Object with Preprocessor into process pipeline.


### predict
```python
INFERENCE_ENGINE.predict(input)
```

Peform prediction on Input. Input can be Numpy Array or Image or Data Generator (in case of Test/Validation).

Example:
```Python
# If Model files are `devModel/testmodel1_wgts.h5` and `devModel/testmodel1_arch.json`
# Then `modelName=devModel/testmodel1`

modelName = 'devModel/testmodel1'

infEngine = INFERENCE_ENGINE()
infEngine.load_model_and_weights(modelName)
infEngine.preprocessor_from_meta()

# Predict an input image
infEngine.predict(input = 'test.jpg')
```


__Arguments__

- __input__: (Type - `numpy.array`|`imagePath`|`generator` ): Can be single image file or numpy array of multiple
         images or data generator class.

__Returns__

`Numpy.Array`: of Predictions. Shape of Output [Number of Inputs, Number of Output Classes in Model]


### predict_pipeline
```python
INFERENCE_ENGINE.predict_pipeline(input)
```

Slightly Faster version of predict. Useful for deployment. Do not use `INFERENCE_ENGINE.predict` in production.
Peform prediction on Input. Input can be Numpy Array or Image or Data Generator (in case of Test/Validation).

Example:
```Python
# If Model files are `devModel/testmodel1_wgts.h5` and `devModel/testmodel1_arch.json`
# Then `modelName=devModel/testmodel1`

modelName = 'devModel/testmodel1'

infEngine = INFERENCE_ENGINE()
infEngine.load_model_and_weights(modelName)
infEngine.preprocessor_from_meta()

# Predict an input image
infEngine.predict_pipeline(input = 'test.jpg')
```


__Arguments__

- __input__: (Type - `numpy.array`|`imagePath`|`generator` ): Can be single image file or numpy array of multiple
         images or data generator class.

__Returns__

`Numpy.Array`: of Predictions. Shape of Output [Number of Inputs, Number of Output Classes in Model]


### decode_predictions
```python
INFERENCE_ENGINE.decode_predictions(pred, top_preds=4, retType='tuple')
```

Returns Decodes predictions with label/class names with output probabilites.
During production this can be used to return a json serializable dictionary instead of tuple.

Example:
```Python
# If Model files are `devModel/testmodel1_wgts.h5` and `devModel/testmodel1_arch.json`
# Then `modelName=devModel/testmodel1`

modelName = 'devModel/testmodel1'

infEngine = INFERENCE_ENGINE()
infEngine.load_model_and_weights(modelName)
infEngine.preprocessor_from_meta()

# Predict an input image
pred = infEngine.predict_pipeline(input = 'test.jpg')
pred_tuple = infEngine.decode_predictions(pred, top_preds=2)

# Get a json serializable dictionary instead of tuple
pred_dict  = infEngine.decode_predictions(pred, top_preds=2, retType = 'dict')

```

__Arguments__

- __pred__: (Type - `numpy.array`): Prediction output of either `INFERENCE_ENGINE.predict` or
         `INFERENCE_ENGINE.predict_pipleline`.
- __top_preds__: (Type - `Integer`): [Optional] : `Default = 4` - Number of top prediction to return. If the number is
          set to higher than number of classes in network, it returns all predictions.
- __retType__: (Type - `String`): [Optional] : `Default = tuple`. Options - [`dict` or `tuple`]. `Dict` helpful in production.

__Returns__

`Tuple or Dict`: of Predictions with probabilities. Shape of Output [Number of Inputs, Max(top_preds,Number of Output Classes in Model)]


### getLayerNames
```python
INFERENCE_ENGINE.getLayerNames()
```

Get the layer names of the network. Useful for when using Explainable-AI function as it expects `layer name` as argument.

Example:
```Python
# If Model files are `devModel/testmodel1_wgts.h5` and `devModel/testmodel1_arch.json`
# Then `modelName=devModel/testmodel1`

modelName = 'devModel/testmodel1'

infEngine = INFERENCE_ENGINE()
infEngine.load_model_and_weights(modelName)

# Print the Layer Names
print('
'.join(infEngine.getLayerNames()))
```



### summary
```python
INFERENCE_ENGINE.summary()
```

Show model network structure and print parameters summary.

__Arguments__

- __None__: None

__Returns__

`None`: None; Prints the model summary


### generate_evaluation_report
```python
INFERENCE_ENGINE.generate_evaluation_report(testSet=None,
                                            predictions=None,
                                            printStat=False,
                                            returnPlot=False,
                                            showPlot=False,
                                            pdfName=None,
                                            **kwargs)
```

Generate a comprehensive PDF report with model sensitivity, specificity, accuracy, confidence intervals,
ROC Curve Plot, Precision Recall Curve Plot, and Confusion Matrix Plot for each class.
This function can be used when evaluating a model with Test or Validation Data Set.

Example:

```Python
# Load Dataset
trainSet,testSet,labelNames =ai.datasetFromFolder(datasetFolderPath, targetDim = (224,224)).load_dataset()

# Intialize Inference Engine
infEngine = ai.INFERENCE_ENGINE(MODEL_SAVE_NAME)

# Preform Prediction on DataSet
predsG = infEngine.predict(testSet.data)

# Generate Report
infEngine.generate_evaluation_report(testSet,predictions = predsG , pdfName = "expt_evaluation_report.pdf")

```

Alternatively:
```Python
# Load Dataset
trainSet,testSet,labelNames =ai.datasetFromFolder(datasetFolderPath, targetDim = (224,224)).load_dataset()

# Intialize Inference Engine
infEngine = ai.INFERENCE_ENGINE(MODEL_SAVE_NAME)

# Generate Report - If predictions are not passed, then automatically prediction is performed.
infEngine.generate_evaluation_report(testSet, pdfName = "expt_evaluation_report.pdf")

```
__Arguments__

- __testSet__: (Type - `numpy.array` or `generator`) : Test Data Set to perform evaluation on.
- __predictions__: (Type - `numpy.array`): [Optional] : Prediction output of either `INFERENCE_ENGINE.predict` or
         `INFERENCE_ENGINE.predict_pipleline`. If this parameter is not set, then prediction
         is perfomred internally and evaluation report is generated.
- __pdfName__: (Type - `Bool`): [Optional] : `Default = ModelName + _report.pdf` - Pdf Output Name.
- __printStat__: (Type - `Bool`): [Optional] : `Default = False` - Print Statistics on console.
- __returnPlot__: (Type - `Bool`): [Optional] : `Default = False` - Return Plot Figure Handle.
- __showPlot__: (Type - `Bool`): [Optional] : `Default = False` - Show Plot figure.

__Returns__

`None or Plot Handle`: If `returnPlot = True` then Plot Handle will be returned else None.


### explain
```python
INFERENCE_ENGINE.explain(input,
                         predictions=None,
                         layer_to_explain='CNN3',
                         classNames=None,
                         selectedClasses=None,
                         expectedClass=None,
                         showPlot=False)
```

Explains a model layer with respect to Input and Output using Grad-cam. Basically, see what the AI is seeing to arrive at
a certain prediction. More methods to be updated in next versions.

``` Python
# Load a sample
image = load(Image)

# Intialize Inference Engine
infEngine = ai.INFERENCE_ENGINE(MODEL_SAVE_NAME)

# Print Layer Names
print('
'.join(infEngine.getLayerNames()))

# If predictions are not passed, then automatically prediction is performed. You can perform prediction first then pass
  to the below function. Pass one of the layer name from above output to `layer_to_explain`.
infEngine.explain(image, layer_to_explain='CNN3')
```

__Arguments__

- __input__: (Type - `numpy.array` or `image`) : Input to perform explanation on. For safety, pass single or few samples only.
- __predictions__: (Type - `numpy.array`): [Optional] : Prediction output of either `INFERENCE_ENGINE.predict` or
         `INFERENCE_ENGINE.predict_pipleline`. If this parameter is not set, then prediction
         is perfomred internally and explanation is generated.
- __layer_to_explain__: (Type - `String`):  Layer to explain.
- __classNames__: (Type - `Numpy.Array` or `List`): [Optional] : `Default = None| Loaded from Meta File` - Class Names or Label Names of Dataset.
- __selectedClasses__: (Type - `Bool`): [Optional] : `Default = None` - Explain only few subset of Class Names. If `None` then all classes will be explained.
- __expectedClass__: (Type - `Bool`): [Optional] : `Default = None` - Expected Label/Class Name for the Input.

__Returns__

`None`: Shows a plot figure with explanations.



## TRAIN_ENGINE
```python
TRAIN_ENGINE(self, modelName=None)
```

Initializes Training Engine to perform training/prediction. TRAIN_ENGINE is a superclass of INFERENCE_ENGINE.
Meaning, all the methods and functions of INFERENCE_ENGINE are available with TRAIN_ENGINE with additional methods of
its own.

__Arguments__

- __modelName__: (Type - `filepath`): [Optional] model path/name to load existing model or create new model.

__Returns__

`TRAIN_ENGINE Object`: Ready to Train a given dataset.



### train_and_save_model
```python
TRAIN_ENGINE.train_and_save_model(AI_NAME,
                                  MODEL_SAVE_NAME,
                                  trainSet,
                                  testSet,
                                  OUTPUT_CLASSES,
                                  RETRAIN_MODEL,
                                  BATCH_SIZE,
                                  EPOCHS,
                                  LEARNING_RATE,
                                  convLayers=None,
                                  SAVE_BEST_MODEL=False,
                                  BEST_MODEL_COND=None,
                                  callbacks=None,
                                  loss='sparse_categorical_crossentropy',
                                  metrics=['accuracy'],
                                  showModel=False,
                                  CLASS_WEIGHTS=None)
```
"
Main function that trains and saves a model. This automatically builds new model for given networks/AI or reload existing AI model.
This function can be used to retrain existing models or create new models.

!!! danger "IMPORTANT"
 DO NOT PASS ANY EXTENTION TO `MODEL_SAVE_NAME` argument

USAGE:

```Python

# Set Parameters
AI_NAME = 'MobileNet_X'
MODEL_SAVE_NAME = 'testModel1'

OUTPUT_CLASSES = 10
RETRAIN_MODEL = True
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
SAVE_BEST_MODEL = False
BEST_MODEL_COND = None
callbacks = None

# Initialize Train Engine
trainer = ai.TRAIN_ENGINE()

# Train and Save Model
trainer.train_and_save_model(AI_NAME=AI_NAME,  														# AI/Network to Use
       MODEL_SAVE_NAME = MODEL_SAVE_NAME, 										# Target MODEL To Save/Load/Retrain
       trainSet=trainGen, testSet=testGen, OUTPUT_CLASSES=OUTPUT_CLASSES, 		# From Dataset Loader
       RETRAIN_MODEL= RETRAIN_MODEL, BATCH_SIZE= BATCH_SIZE, EPOCHS= EPOCHS, 	# Training Settings
       SAVE_BEST_MODEL = SAVE_BEST_MODEL, 	BEST_MODEL_COND= BEST_MODEL_COND, 	# Early Stopping Settings
       loss='categorical_crossentropy',										# Loss Function
       showModel = False,														# Show Network Summary
       callbacks = callbacks,													# Additional/Advanced Hooks
       )
```

__Arguments__

- __AI_NAME__: (Type - `string` or `NetworkInit() class`): Select Network from catalogue (string) or create your own network and pass the class.
- __MODEL_SAVE_NAME__: (Type - `filepath`): [Optional] model path/name to load existing model or create new model.
- __trainSet__: (Type - `numpy.array` or `generator`): [Optional] : Training Dataset either as generator or numpy array from `DataLoader` class.
- __testSet__: (Type - `numpy.array` or `generator`): [Optional] : Test/Validation Dataset either as generator or numpy array
      from `DataLoader` class.
- __OUTPUT_CLASSES__: (Type - `Int`): Number of unique classes in dataset.
- __RETRAIN_MODEL__: (Type - `Bool`): Whether to retrain existing model. If set to True and model does not exist,
      then it creates a new model and subsequent runs will retrain model.
- __BATCH_SIZE__: (Type - `Int`): Batch size for Training. If Training fails when using large datasets, try reducing this number.
- __EPOCHS__: (Type - `Int`): Number of Epochs to train.
- __LEARNING_RATE__: (Type - `Float`): [Optional] : Set Learning rate. If not set, optimizer default will be used.
- __convLayers__: (Type - `Int`): [Optional] Default is None. Only applicable for certain networks where convolution
         layers are reconfigurable. This parameter can be used to change the num of conv
         layers in Network. See AI_NAME Page for More Details.
- __SAVE_BEST_MODEL__: (Type - `Bool`): [Optional] : `Default: False` - Initializes Training Engine with saving best model feature.
- __BEST_MODEL_COND__: (Type - `String` or `Dict`): [Optional] : `Default: None` - Initializes Training Engine with early stopping feature.
     [Options] -> `Default` or `Dict`.
- __Dict Values Expected__:
- __'monitor'__: (Type - `String`): Which Parameter to Monitor. [Options] -> ('val_accuracy', 'val_loss', 'accuracy'),
- __'min_delta'__: (Type - `Float`): minimum change in the monitored quantity to qualify as an improvement,
       i.e. an absolute change of less than min_delta, will count as no improvement.
- __'patience'__: (Type - `Int`): number of epochs with no improvement after which training will be stopped.
- __loss__: (Type - `String`) : `Default: sparse_categorical_crossentropy`, Loss function to apply. Depends on dataprocessor.
        If dataloaders has one-hot encoded labels then use `sparse_categorical_crossentropy` else if
        labers are encoded then -> `categorical_crossentropy`.
- __metrics__: (Type - `List`): [Optional] : `Default: ['accuracy']`. Metrics to Monitor during Training.
- __showModel__: (Type - `Bool`): [Optional] : Whether to show the network summary before start of training.
- __CLASS_WEIGHTS__: (Type - `Dict`) [Optional] : Dictionary containing class weights for model.fit()
- __callbacks__: (Type - `Tensorflow Callbacks`): Tensorflow Callbacks can be attacked.

__Returns__

`None`: On successful completion saves the trained model.



### plot_train_acc_loss
```python
TRAIN_ENGINE.plot_train_acc_loss()
```

Plot training accuracy and loss graph vs epoch. Generates an interactive graph for inspection.

USAGE:

```Python

# Set Parameters
AI_NAME = 'MobileNet_X'
MODEL_SAVE_NAME = 'testModel1'

OUTPUT_CLASSES = 10
RETRAIN_MODEL = True
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
SAVE_BEST_MODEL = False
BEST_MODEL_COND = None
callbacks = None

# Initialize Train Engine
trainer = ai.TRAIN_ENGINE()

# Train and Save Model
trainer.train_and_save_model(AI_NAME=AI_NAME,  														# AI/Network to Use
       MODEL_SAVE_NAME = MODEL_SAVE_NAME, 										# Target MODEL To Save/Load/Retrain
       trainSet=trainGen, testSet=testGen, OUTPUT_CLASSES=OUTPUT_CLASSES, 		# From Dataset Loader
       RETRAIN_MODEL= RETRAIN_MODEL, BATCH_SIZE= BATCH_SIZE, EPOCHS= EPOCHS, 	# Training Settings
       SAVE_BEST_MODEL = SAVE_BEST_MODEL, 	BEST_MODEL_COND= BEST_MODEL_COND, 	# Early Stopping Settings
       loss='categorical_crossentropy',										# Loss Function
       showModel = False,														# Show Network Summary
       callbacks = callbacks,													# Additional/Advanced Hooks
       )

trainer.plot_training_metrics()
```
__Arguments__

- __None__: None

__Returns__

`None`: Opens accuracy vs loss vs epoch plot.

