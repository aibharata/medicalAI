# medicalAI
Medical-AI is a AI framework specifically for Medical Applications

# Installation
```py 
pip install medicalai
```
# Requirements
Python Version : 3.5-3.7 (Doesn't Work on 3.8 Since Tensorflow does not support 3.8 yet.

Dependencies: Numpy, Tensorflow, Seaborn, Matplotlib, Pandas

    NOTE: Dependency libraries are automatically installed. No need for user to install them manually.

# Usage

### Importing the Library
```py 
import medicalai as mai
```

## Using Templates
You can use the following templates to perform specific Tasks

### Load Dataset From Folder
Set the path of the dataset and set the target dimension of image that will be input to AI network.
```py 
trainSet,testSet,labelNames =mai.datasetFromFolder(datasetFolderPath, targetDim = (96,96)).load_dataset()
```
    - trainSet contains 'data' and 'labels' accessible by trainSet.data and trainSet.labels
    - testSet contains 'data' and 'labels' accessible by testSet.data and testSet.labels
    - labelNames contains class names/labels

### Check Loaded Dataset Size
```py 
print(trainSet.data.shape)
print(trainSet.labels.shape)
```

### Run Training and Save Model
```py
trainer = mai.TRAIN_ENGINE()
trainer.train_and_save_model(AI_NAME= 'tinyMedNet', MODEL_SAVE_NAME='PATH_WHERE_MODEL_IS_SAVED_TO', trainSet, testSet, OUTPUT_CLASSES, RETRAIN_MODEL= True, BATCH_SIZE= 32, EPOCHS= 10, LEARNING_RATE= 0.001)
```


### Plot Training Loss and Accuracy
```py
trainer.plot_train_acc_loss()
```

### Plot Confusion matrix of test data
```py
trainer.plot_confusion_matrix(labelNames,title='Confusion Matrix of Trained Model on Test Dataset')
```

### Loading Model for Prediction 
```py
model = mai.load_model_and_weights(modelName = 'PATH_WHERE_MODEL_IS_SAVED_TO')
```


### Predict With Labels 
```py
mai.predict_labels(model, testSet.data[0:2], expected_output =testSet.labels[0:2], labelNames=labels, top_preds=3)
```
### Get Just Values of Prediction without postprocessing
```py
model.predict(testSet.data[0:2])
```

## Advanced Usage

### Code snippet for Training Using Medical-AI 
```py
## Setup AI Model Manager with required AI. 
model = mai.modelManager(AI_NAME= AI_NAME, modelName = MODEL_SAVE_NAME, x_train = train_data, OUTPUT_CLASSES = OUTPUT_CLASSES, RETRAIN_MODEL= RETRAIN_MODEL)

# Start Training
result = mai.train(model, train_data, train_labels, BATCH_SIZE, EPOCHS, LEARNING_RATE, validation_data=(test_data, test_labels), callbacks=['tensorboard'])

# Evaluate Trained Model on Test Data
model.evaluate(test_data, test_labels)

# Plot Accuracy vs Loss for Training
mai.plot_training_metrics(result)

#Save the Trained Model
mai.save_model_and_weights(model, outputName= MODEL_SAVE_NAME)
```