import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, MaxPool2D, add, Flatten, Dense


def PEPXModel(input_tensor, filters, name):
    x = Conv2D(filters=filters, kernel_size=(1, 1), activation='relu', name=name + 'FP')(input_tensor)
    x = Conv2D(filters=filters, kernel_size=(1, 1), activation='relu', name=name + 'Expansion')(x)
    x = Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same', name=name + 'DWConv3_3')(x)
    x = Conv2D(filters=filters, kernel_size=(1, 1), activation='relu', name=name + 'SP')(x)
    x = Conv2D(filters=filters, kernel_size=(1, 1), activation='relu', name=name + 'Extension')(x)
    return x


def keras_model_build(img_input=(224, 224, 3), classes =4):
    input = Input(shape=img_input, name='input')
    x = Conv2D(input_shape=img_input, filters=64, kernel_size=(7, 7), activation='relu', padding='same',
               strides=(2, 2))(input)
    x = MaxPool2D(pool_size=(2, 2))(x)
    # PEPX1_Conv1x1
    p_1_y = Conv2D(256, (1, 1), padding='same', activation='relu', name='PEPX1_Conv')(x)

    # Stage1
    y_1_1 = PEPXModel(x, 256, 'PEPX1.1')
    y_1_2 = PEPXModel(add([y_1_1, p_1_y]), 256, 'PEPX1.2')
    y_1_3 = PEPXModel(add([y_1_1, y_1_2, p_1_y]), 256, 'PEPX1.3')
    # PEPX2_Conv1x1
    p_2_y = Conv2D(512, (1, 1), padding='same', activation='relu', name='PEPX2_Conv')(add([p_1_y, y_1_1, y_1_2, y_1_3]))
    p_2_y = MaxPool2D(pool_size=(2, 2))(p_2_y)
    # Stage2
    y_2_1 = PEPXModel(add([y_1_3, y_1_2, y_1_1, p_1_y]), 512, 'PEPX2.1')
    y_2_1 = MaxPool2D(pool_size=(2, 2))(y_2_1)
    y_2_2 = PEPXModel(add([y_2_1, p_2_y]), 512, 'PEPX2.2')
    y_2_3 = PEPXModel(add([y_2_1, y_2_2, p_2_y]), 512, 'PEPX2.3')
    y_2_4 = PEPXModel(add([y_2_1, y_2_2, y_2_3, p_2_y]), 512, 'PEPX2.4')
    # PEPX3_Conv1x1
    p_3_y = Conv2D(1024, (1, 1), padding='same', activation='relu', name='PEPX3_Conv')(
        add([p_2_y, y_2_1, y_2_2, y_2_3, y_2_4])
    )
    p_3_y = MaxPool2D(pool_size=(2, 2))(p_3_y)
    # Stage3
    y_3_1 = PEPXModel(add([y_2_1, y_2_2, y_2_3, y_2_4, p_2_y]), 1024, 'PEPX3.1')
    y_3_1 = MaxPool2D(pool_size=(2, 2))(y_3_1)
    y_3_2 = PEPXModel(y_3_1, 1024, 'PEPX3.2')
    y_3_3 = PEPXModel(add([y_3_1, y_3_2]), 1024, 'PEPX3.3')
    y_3_4 = PEPXModel(add([y_3_1, y_3_2, y_3_3]), 1024, 'PEPX3.4')
    y_3_5 = PEPXModel(add([y_3_1, y_3_2, y_3_3, y_3_4]), 1024, 'PEPX3.5')
    y_3_6 = PEPXModel(add([y_3_1, y_3_2, y_3_3, y_3_4, y_3_5]), 1024, 'PEPX3.6')
    # PEPX4_Conv1x1
    p_4_y = Conv2D(2048, (1, 1), padding='same', activation='relu', name='PEPX4_Conv1')(
        add([p_3_y, y_3_1, y_3_2, y_3_3, y_3_4, y_3_5, y_3_6])
    )
    p_4_y = MaxPool2D(pool_size=(2, 2))(p_4_y)
    # Stage4
    y_4_1 = PEPXModel(add([y_3_1, y_3_2, y_3_3, y_3_4, y_3_5, y_3_6, p_3_y]), 2048, 'PEPX4.1')
    y_4_1 = MaxPool2D(pool_size=(2, 2))(y_4_1)
    y_4_2 = PEPXModel(add([y_4_1, p_4_y]), 2048, 'PEPX4.2')
    y_4_3 = PEPXModel(add([y_4_1, y_4_2, p_4_y]), 2048, 'PEPX4.3')
    # FC
    fla = Flatten()(add([y_4_1, y_4_2, y_4_3, p_4_y]))
    d1 = Dense(1024, activation='relu')(fla)
    d2 = Dense(256, activation='relu')(d1)
    output = Dense(classes, activation='softmax')(d2)

    return tf.keras.models.Model(input, output)


if __name__ == '__main__':
    model = keras_model_build()
    model.summary()
