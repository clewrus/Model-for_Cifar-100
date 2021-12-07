from os import name
from warnings import filters
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from layers import convolutional_backbone



def model_cifar100( input_shape, batch_size=None, augmentation=False ):
    def model_input():
        if batch_size:
            return keras.Input(shape=input_shape, batch_size=batch_size)
        else:
            return keras.Input(shape=input_shape)

    def augment_image(y):
        #y = tf.keras.layers.experimental.preprocessing.RandomContrast((0.8, 1.2))(y)
        y = tf.keras.layers.experimental.preprocessing.RandomFlip()(y)
        
        #y = tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)(y)
        #y = tf.keras.layers.experimental.preprocessing.RandomTranslation(0.2, 0.2)(y)
        # y = tf.keras.layers.experimental.preprocessing.RandomContrast((0.7, 1.3))(y)
        # y = tf.keras.layers.experimental.preprocessing.RandomZoom(0.3)(y)
        return y

    def dense_head(y, num):
        y = keras.layers.Dense(
            units=num,
            activation='tanh',
            kernel_regularizer=tf.keras.regularizers.l2(0.0002),
            bias_regularizer=tf.keras.regularizers.l2(0.0002),
            activity_regularizer=tf.keras.regularizers.l2(0.0002),
        )(y)

        return y

    def classifier(y):
        y = keras.layers.Dense(
            units=100,
            activation='softmax'
        )(y)

        return y

    x = model_input()

    with tf.name_scope("augmentation"):
        y = augment_image(x) if augmentation else x
    
    with tf.name_scope("convolutional_backbone"):
        y = convolutional_backbone(y)
    
    with tf.name_scope("dense_head"):
        y = keras.layers.Flatten()(y)
        y = dense_head(y, 30)
        y = dense_head(y, 60)
    
    with tf.name_scope("classifier"):
        y = classifier(y)
    
    m = keras.Model(inputs=x, outputs=y)
    optimizer = tf.optimizers.Adam(learning_rate=0.0001)
    m.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m


def load_backbone_from_cifar10_model( cifar10_model, trainable, **params ):
    cifar100_model = model_cifar100( **params )

    for i in range( 61 ):
        cifar100_model.layers[i].set_weights(cifar10_model.layers[i].get_weights())
        cifar100_model.layers[i].trainable = trainable

    return cifar100_model
