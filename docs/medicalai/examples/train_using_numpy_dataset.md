# Example 1: IMAGE Recognition/Classification
## Train an AI model using medicalai's numpy dataset processor

```python
import os
import medicalai as ai
```

#### Download sample Dataset

```python
datasetDWLD = ai.getFile('https://github.com/aibharata/covid19-dataset/archive/v1.0.zip', subDir='dataset')
datasetFolderPath = datasetDWLD+'/covid19-dataset-1.0/chest-xray-pnumonia-covid19/'
```

### Define the hyperparameters of Dataset Processor

A. Specify the dimensions of image to be fed to network
```python
IMG_HEIGHT = 64
IMG_WIDTH = 64
OUTPUT_CLASSES = 3 
```
#### Process your dataset using Numpy based `datasetFromFolder` class.
The `datasetFromFolder` class takes a folder path where your dataset is located. This folder should have `test` and `train` folders. 
Each of the folder should have the class sub-folder for your classification problem.
```python
trainSet,testSet,labelNames =ai.datasetFromFolder(datasetFolderPath, targetDim = (IMG_WIDTH,IMG_WIDTH)).load_dataset()

# Print shapes of the loaded test and train data
print('TrainSet Data Shape: {:}; TrainSet Labels Shape:{:}'.format(testSet.data.shape,testSet.labels.shape))
print('TrainSet Data Shape: {:}; TrainSet Labels Shape:{:}'.format(testSet.data.shape,testSet.labels.shape))
```

### Define the hyperparameters of Training
A. Specify training hyperparamters
```
batch_size = 32
epochs = 10
learning_rate = 0.0001
```
B. Specify the model name to save/retrain
```python
MODEL_SAVE_NAME = 'medicalai_test_model_1'
```
C. Choose from the prebuilt networks from Medicalai Library. You can also pass a class with your own custom network to `AI_NAME` parameter.
```python
AI_NAME = 'tinyMedNet'
```

### Initialize TRAIN_ENGINE and Start Training
```python
trainer = ai.TRAIN_ENGINE()
trainer.train_and_save_model(AI_NAME=AI_NAME,
                             MODEL_SAVE_NAME = MODEL_SAVE_NAME, 
                             trainSet=trainGen, testSet=testGen,
                             OUTPUT_CLASSES=OUTPUT_CLASSES, 
                             RETRAIN_MODEL= True,
                             BATCH_SIZE= batch_size,
                             EPOCHS= epochs, 
                             LEARNING_RATE= learning_rate,
                             SAVE_BEST_MODEL = True,
                             showModel = True # Set this True if you want to see model summary
                             )
```
### View and Save Training Stats
Plot training accuracy and loss w.r.t to epochs

```python
trainer.plot_train_acc_loss()
```
### Generate evaluation report for Trained Model
```python
trainer.generate_evaluation_report(testSet)
```
## Explain the model for a input sample
```python
trainer.explain(testSet.data[0:1], layer_to_explain='CNN3', classNames = labelNames)
```


