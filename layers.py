from os import name
from warnings import filters
from tensorflow import keras


def normalized_resnet_block( x, filters ):
    for i in range(3):
        y = keras.layers.BatchNormalization()(x)

        y = keras.layers.Conv2D(
            filters,
            (3, 3),
            activation='relu',
            padding='same'
        )(y)

        y = keras.layers.Dropout(0.000)(y)

        y = keras.layers.Conv2D(
            filters,
            (3, 3),
            activation='relu',
            padding='same'
        )(y)

        x = keras.layers.BatchNormalization()(x + y)
    
    return x

def convolutional_backbone( y ):

    y = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(y)
    y = normalized_resnet_block(y, 16)      
    y = keras.layers.MaxPool2D()(y)

    y = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(y)
    y = normalized_resnet_block(y, 32)   
    y = keras.layers.MaxPool2D()(y)

    y = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(y)
    y = normalized_resnet_block(y, 64)      
    y = keras.layers.AveragePooling2D((8, 8))(y)

    return y