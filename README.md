<p align="center">
  <a href="https://aibharata.github.io/medicalAI/"><img src="https://raw.githubusercontent.com/aibharata/medicalAI/master/logo/logo.png" alt="MedicalAI"></a>
</p>
<p align="center">
    <em>Medical-AI is a AI framework for rapid protyping for Medical Applications</em>
</p>


---

**Documentation**: <a href="https://aibharata.github.io/medicalAI/" target="_blank">https://aibharata.github.io/medicalAI/</a>

**Source Code**: <a href="https://github.com/aibharata/medicalai" target="_blank">https://github.com/aibharata/medicalai</a>

**Youtube Tutorial**: <a href="https://www.youtube.com/V4nCX-kLACg" target="_blank">https://www.youtube.com/V4nCX-kLACg</a>

---


[![Downloads](https://pepy.tech/badge/medicalai)](https://pepy.tech/project/medicalai) [![Downloads](https://pepy.tech/badge/medicalai/month)](https://pepy.tech/project/medicalai/month) [![Documentation Status](https://readthedocs.org/projects/medicalai/badge/?version=latest)](https://medicalai.readthedocs.io/en/latest/?badge=latest) [![Gitter](https://badges.gitter.im/aibh-medicalAI/devteam.svg)](https://gitter.im/aibh-medicalAI/devteam?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) 


Medical-AI is a AI framework  for rapid prototyping of AI for Medical Applications.

## Installation

<div class="termy">

```py
pip install medicalai
```

</div>
## Requirements
Python Version : 3.5-3.7 (Doesn't Work on 3.8 Since Tensorflow does not support 3.8 yet.

Dependencies: Numpy, Tensorflow, Seaborn, Matplotlib, Pandas

    NOTE: Dependency libraries are automatically installed. No need for user to install them manually.

## Usage
### Getting Started Tutorial: Google Colab [Google Colab Notebook Link](https://colab.research.google.com/drive/1Wma4i5f11oyYrrkz0Y-3FOyPGmIpwKdD)

### Importing the Library
```py 
import medicalai as ai
```

## Using Templates
You can use the following templates to perform specific Tasks

### Load Dataset From Folder
Set the path of the dataset and set the target dimension of image that will be input to AI network.
```py 
trainSet,testSet,labelNames =ai.datasetFromFolder(datasetFolderPath, targetDim = (96,96)).load_dataset()
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
trainer = ai.TRAIN_ENGINE()
trainer.train_and_save_model(AI_NAME= 'tinyMedNet', MODEL_SAVE_NAME='PATH_WHERE_MODEL_IS_SAVED_TO', trainSet, testSet, OUTPUT_CLASSES, RETRAIN_MODEL= True, BATCH_SIZE= 32, EPOCHS= 10, LEARNING_RATE= 0.001)
```


### Plot Training Loss and Accuracy
```py
trainer.plot_train_acc_loss()
```

### Generate a comprehensive evaluation PDF report 
```py
trainer.generate_evaluation_report()
```
PDF report will be generated with model sensitivity, specificity, accuracy, confidence intervals,
ROC Curve Plot, Precision Recall Curve Plot, and Confusion Matrix Plot for each class.
This function can be used when evaluating a model with Test or Validation Data Set.

### Explain the Model on a sample
```py
trainer.explain(testSet.data[0:1], layer_to_explain='CNN3')
```


### Loading Model for Prediction 
```py
infEngine = ai.INFERENCE_ENGINE(modelName = 'PATH_WHERE_MODEL_IS_SAVED_TO')
```


### Predict With Labels 
```py
infEngine.predict_with_labels(testSet.data[0:2], top_preds=3)
```
### Get Just Values of Prediction without postprocessing
```py
infEngine.predict(testSet.data[0:2])
```

### Alternatively, use a faster prediction method in production
```py
infEngine.predict_pipeline(testSet.data[0:1])
```
## Advanced Usage

### Code snippet for Training Using Medical-AI 
```py
## Setup AI Model Manager with required AI. 
model = ai.modelManager(AI_NAME= AI_NAME, modelName = MODEL_SAVE_NAME, x_train = train_data, OUTPUT_CLASSES = OUTPUT_CLASSES, RETRAIN_MODEL= RETRAIN_MODEL)

# Start Training
result = ai.train(model, train_data, train_labels, BATCH_SIZE, EPOCHS, LEARNING_RATE, validation_data=(test_data, test_labels), callbacks=['tensorboard'])

# Evaluate Trained Model on Test Data
model.evaluate(test_data, test_labels)

# Plot Accuracy vs Loss for Training
ai.plot_training_metrics(result)

#Save the Trained Model
ai.save_model_and_weights(model, outputName= MODEL_SAVE_NAME)
```

## Automated Tests
To Check the tests

        pytest

To See Output of Print Statements

        pytest -s 

## Author
Dr. Vinayaka Jyothi
