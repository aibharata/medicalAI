# Example 1: Train your model using medicalai generator


```python
import os
import medicalai as ai
import tensorflow as tf
```
### Define the hyperparameters
Specify the dataset folder which further contains test & train folders each with n class object folders
```python
datasetFolderPath = "../data"
```
Specify the dimensions of image to be fed to network
```python
IMG_HEIGHT = 224
IMG_WIDTH = 224
```
Specify the number of classes for classification
```python
OUTPUT_CLASSES = 3
```
Specify the name of the model to be trained on
```python
EXPT_NAME = '1'
AI_NAME = 'mobilenet' 
MODEL_SAVE_NAME = '../model/'+AI_NAME+'/Medical_RSNA_'+str(IMG_HEIGHT)+'x'+str(IMG_WIDTH)+'_'+AI_NAME+'_EXPT_'+str(EXPT_NAME)
```
Specify remaining hyperparamters
```
batch_size = 32
epochs = 10
learning_rate = 0.0001

```
### Define the augmentation for the generator
```python
augment = ai.AUGMENTATION(rotation_range = 12, 
                          fill_mode='nearest', 
                          width_shift_range=0.1, 
                          height_shift_range=0.1, 
                          brightness_range = (0.9, 1.1), 
                          zoom_range=(0.85, 1.15), 
                          rescale= 1./255,)
```
- Load your data from folder using datasetGenFromFolder if your data is in folder structured form
```python
dsHandler = ai.datasetGenFromFolder(folder=datasetFolderPath,
                                    targetDim=(IMG_HEIGHT,IMG_WIDTH), 
                                    augmentation=augment,
                                    class_mode="categorical"
                                    normalize=False,
                                    batch_size=batch_size,
                                    augmentation=True,
                                    color_mode='rgb', #if the images are of rgb channels else 'grayscale'
                                    class_mode='categorical',
                                    shuffle=True,
                                    seed=23))

trainGen, testGen = dsHandler.load_generator()
```
- Incase your data is not in folder structured form but rather details embeded in a csv file, use the datasetGenFromDataframe method to load data to generator instead of datasetGenFromFolder
```python
dsHandler = ai.datasetGenFromDataframe( folder = datasetFolderPath, #folder containg train and test folders
                                        csv_path='.', #path to train.cvs and test.csv
                                        x_col='name', 
                                        y_col='labels',
                                        targetDim=(IMG_HEIGHT,IMG_WIDTH), 
                                        normalize=False,
                                        batch_size=batch_size,
                                        augmentation=True,
                                        color_mode='rgb',
                                        class_mode='sparse',
                                        shuffle=True,
                                        seed=23
                                        )
trainGen, testGen = dsHandler.load_generator()
```
### Train model
Now our image generator is ready to be trained on our model. But first we need to define a tensorflow callback for the model
```python
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                        MODEL_SAVE_NAME+'best.h5', 
                                        verbose=0,
                                        mode='auto', 
                                        save_freq=5,
                                        save_best_only=True,
                                        )
callbacks = [model_checkpoint]
```
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
                             callbacks = callbacks,
                             convLayers= None,
                             loss='categorical_crossentropy',
							               showModel = False #mark this True if you want to see model summary
                             )
```
- Use the above model to predict
```python
test_x,test_y = dsHandler.get_numpy(testGen)
predsG = trainer.predict(test_x)
```
- Generate evaluation report
```python
trainer.generate_evaluation_report(testGen,predictions = predsG)
```


