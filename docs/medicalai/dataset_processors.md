
# medicalai.chief.dataset_prepare


## datasetFromFolder
```python
datasetFromFolder(self,
                  folder,
                  targetDim=(31, 31),
                  normalize=False,
                  name=None,
                  useCache=True,
                  forceCleanCache=False)
```

TODO: Fix samplingMethodName assignment


## datasetGenFromFolder
```python
datasetGenFromFolder(self,
                     folder,
                     targetDim=(224, 224),
                     normalize=False,
                     batch_size=16,
                     augmentation=True,
                     color_mode='rgb',
                     class_mode='sparse',
                     shuffle=True,
                     seed=23)
```

Create a dataset generator from dataset present in Folder.
The folder should consist of `test` and `train` folders and each of the folders should have `n` classes of folders.

__Arguments__

- __folder__: The directory must be set to the path where your `n` classes of folders are present.
- __targetDim__: The target_size is the size of your input images to the neural network.
- __class_mode__: Set `binary` if classifying only two classes, if not set to `categorical`, in case of an Autoencoder system, both input and the output would probably be the same image, for this case set to `input`.
- __color_mode__: `grayscale` for black and white or grayscale, `rgb` for three color channels.
- __batch_size__: Number of images to be yielded from the generator per batch. If training fails lower this number.
- __augmentation__: : [Optional] : `Default = True`: Perform augmentation on Dataset
- __shuffle__: : [Optional] : `Default = True`: Shuffle Dataset
- __seed__: : [Optional] : `Default = 23`: Initialize Random Seed

__Returns__

`None`: Initializes Test and Train Data Generators



## datasetGenFromDataframe
```python
datasetGenFromDataframe(self,
                        folder,
                        csv_path='.',
                        x_col='name',
                        y_col='labels',
                        targetDim=(224, 224),
                        normalize=False,
                        batch_size=16,
                        augmentation=True,
                        color_mode='rgb',
                        class_mode='sparse',
                        shuffle=True,
                        seed=17)
```
Creates Keras Dataset Generator for Handling Large Datasets from DataFrame.

__Arguments__

- __csv_path__: folder containing train.csv and test.csv.
- __folder__: The directory must be set to the path where your training images are present.
- __x_col__: Name of column containing image name, `default = name`.
- __y_col__: Name of column for labels, `default = labels`.
- __targetDim__: The target_size is the size of your input images to the neural network.
- __class_mode__: Set `binary` if classifying only two classes, if not set to `categorical`, in case of an Autoencoder system, both input and the output would probably be the same image, for this case set to `input`.
- __color_mode__: `grayscale` for black and white or grayscale, `rgb` for three color channels.
- __batch_size__: Number of images to be yielded from the generator per batch. If training fails lower this number.
- __augmentation__: : [Optional] : `Default = True`: Perform augmentation on Dataset
- __shuffle__: : [Optional] : `Default = True`: Shuffle Dataset
- __seed__: : [Optional] : `Default = 23`: Initialize Random Seed

__Returns__

`None`: Initializes Test and Train Data Generators

