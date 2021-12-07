from os import name
from warnings import filters
import tensorflow as tf
import numpy as np

from argparse import ArgumentParser
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.python.ops.gen_math_ops import ceil, floor


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

def model( num_classes, input_shape, batch_size=None, augmentation=False ):
    def model_input():
        if batch_size:
            return keras.Input(shape=input_shape, batch_size=batch_size)
        else:
            return keras.Input(shape=input_shape)

    def augment_image(y):
        #y = tf.keras.layers.experimental.preprocessing.RandomContrast((0.8, 1.2))(y)
        # y = tf.keras.layers.experimental.preprocessing.RandomFlip()(y)
        
        #y = tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)(y)
        #y = tf.keras.layers.experimental.preprocessing.RandomTranslation(0.2, 0.2)(y)
        # y = tf.keras.layers.experimental.preprocessing.RandomContrast((0.7, 1.3))(y)
        # y = tf.keras.layers.experimental.preprocessing.RandomZoom(0.3)(y)
        return y

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

    def dense_head(y):
        y = keras.layers.Dense(
            units=20,
            activation='tanh',
            kernel_regularizer=tf.keras.regularizers.l2(0.0002),
            bias_regularizer=tf.keras.regularizers.l2(0.0002),
            activity_regularizer=tf.keras.regularizers.l2(0.0002),
        )(y)

        #y = keras.layers.Dropout(0.1)(y)

        return y

    def classifier(y):
        y = keras.layers.Dense(
            units=num_classes,
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
        y = dense_head(y)
    
    with tf.name_scope("classifier"):
        y = classifier(y) 
    
    m = keras.Model(inputs=x, outputs=y)
    optimizer = tf.optimizers.Adam(learning_rate=0.0001)
    m.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

def load_train_validation_test(val_split=0.1, batch_size=1):
    (train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()
    train_x, test_x = train_x / 255, test_x / 255

    train_size = train_x.shape[0]
    val_size = batch_size * int(ceil(val_split * train_size / batch_size))

    (val_x, val_y) = (train_x[:val_size], train_y[:val_size])
    (train_x, train_y) = (train_x[val_size:], train_y[val_size:])

    val_x = tf.data.Dataset.from_tensor_slices(val_x).batch(batch_size)
    val_y = tf.data.Dataset.from_tensor_slices(val_y).batch(batch_size)

    train_x = tf.data.Dataset.from_tensor_slices(train_x).batch(batch_size)
    train_y = tf.data.Dataset.from_tensor_slices(train_y).batch(batch_size)

    test_x = tf.data.Dataset.from_tensor_slices(test_x).batch(batch_size)
    test_y = tf.data.Dataset.from_tensor_slices(test_y).batch(batch_size)

    val = tf.data.Dataset.zip((val_x, val_y))
    train = tf.data.Dataset.zip((train_x, train_y)).shuffle(train_size, seed=228)
    test = tf.data.Dataset.zip((test_x, test_y))
   
    return train, val, test

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument( 'experiment_name', type=str, default=None )
    parser.add_argument( '--restore_from', type=str, default=None )
    parser.add_argument( '--epochs', type=int, default=1 )
    parser.add_argument( '--batches', type=int, default=64 )
    parser.add_argument( '--metrics_only', action='store_true')
    args = parser.parse_args()

    train, validation, test = load_train_validation_test(val_split=0.1, batch_size=args.batches)

    net = model(10, input_shape=(32, 32, 3), augmentation=True)

    if args.restore_from:
        ckpt_path = "checkpoints/{}".format( args.restore_from )
        net.load_weights( ckpt_path, )

    if not args.metrics_only:
        log_dir = 'logs/fit/{}'.format( args.experiment_name )
        log_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1 )

        checkpoint_path = f"checkpoints/{args.experiment_name}/" + "{epoch:02d}.ckpt"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True)    

        
        net.fit(train, validation_data=validation, epochs=args.epochs, callbacks=[log_callback, cp_callback])

    print("train", net.evaluate(train) )
    print("validation", net.evaluate(validation) )
    print("test", net.evaluate(test) )
    
    

