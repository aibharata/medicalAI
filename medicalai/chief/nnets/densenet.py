import tensorflow as tf


def DenseNet121_Model(img_input=(224,224,3),classes=3):
  """
  Loaded the DenseNet121 network, ensuring the head FC layer sets are left off

  Arguments:
    img_input: optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (224, 224, 3)
                (with 'channels_last' data format) or (3, 224, 224) (with 'channels_first' data format). It should have exactly 3 inputs channels, 
                and width and height should be no smaller than 32. E.g. (200, 200, 3) would be one valid value.
    classes: Number of classes to be predicted.

    Returns : model
  """
  baseModel = tf.keras.applications.DenseNet121(weights="imagenet", include_top=False,
                                                input_tensor=tf.keras.layers.Input(shape=img_input))
  # construct the head of the model that will be placed on top of the the base model
  output = baseModel.output
  output = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(output)
  output = tf.keras.layers.Flatten(name="flatten")(output)
  output = tf.keras.layers.Dense(512, activation="relu")(output)
  output = tf.keras.layers.Dropout(0.25)(output)
  output = tf.keras.layers.Dense(classes, activation="softmax")(output)
  # place the head FC model on top of the base model (this will become the actual model we will train)
  model = tf.keras.Model(inputs=baseModel.input, outputs=output)
  # loop over all layers in the base model and freeze them so they will not be updated during the first training process
  for layer in baseModel.layers:
    layer.trainable = False
  return model

if __name__ == '__main__':
    model = DenseNet121_Model()
    model.summary()
