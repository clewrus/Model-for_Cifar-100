from warnings import filters
import tensorflow as tf
import numpy as np

from argparse import ArgumentParser
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.python.ops.gen_math_ops import ceil, floor


def normalized_resnet_block( x, filters ):
    y = x

    for i in range(2):
        y = keras.layers.Conv2D(
            filters,
            (3, 3),
            activation='relu',
            padding='same'
        ).apply(y)

    y = keras.layers.BatchNormalization().apply(x + y)
    return y

def model( num_classes, input_shape, batch_size=None ):
    if batch_size:
        x = keras.Input(shape=input_shape, batch_size=batch_size)
    else:
        x = keras.Input(shape=input_shape)

    filters = 8
    y = x

    for i in range(5):
        filters *= 2
        y = keras.layers.Conv2D(
            filters, 
            (3,3), 
            padding='same'
        ).apply(y)
        y = normalized_resnet_block(y, filters)
        y = keras.layers.MaxPool2D().apply(y)
        y = keras.layers.Dropout(0.1 * (i / 5)).apply(y)

    y = keras.layers.Flatten().apply(y)
    y = keras.layers.Dropout(0.15).apply(y)

    y = keras.layers.Dense(
        units=15,
        activation='tanh',
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        bias_regularizer=tf.keras.regularizers.l2(0.001),
        activity_regularizer=tf.keras.regularizers.l2(0.001)
    ).apply(y)

    y = keras.layers.Dense(
        units=num_classes,
        activation='softmax'
    ).apply(y)

    m = keras.Model(inputs=x, outputs=y)
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

def load_train_validation_test(val_split=0.1, batch_size=1):
    (train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()
    val_batch_num = int(ceil(val_split * train_x.shape[0] / batch_size))

    train_perm = np.random.RandomState(228).permutation(train_x.shape[0])
    test_perm = np.random.RandomState(228).permutation(test_x.shape[0])

    train_x , train_y = train_x[train_perm], train_y[train_perm]
    test_x , test_y = test_x[test_perm], test_y[test_perm]

    train_x = tf.data.Dataset.from_tensor_slices(train_x).batch(batch_size)
    train_y = tf.data.Dataset.from_tensor_slices(train_y).batch(batch_size)

    test_x = tf.data.Dataset.from_tensor_slices(test_x).batch(batch_size)
    test_y = tf.data.Dataset.from_tensor_slices(test_y).batch(batch_size)

    val_x = train_x.take(val_batch_num)
    val_y = train_y.take(val_batch_num)

    train_x = train_x.skip(val_batch_num)
    train_y = train_y.skip(val_batch_num)

    train = tf.data.Dataset.zip((train_x, train_y))
    val = tf.data.Dataset.zip((val_x, val_y))
    test = tf.data.Dataset.zip((test_x, test_y))
   
    return train, val, test

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument( 'experiment_name', type=str, default=None )
    parser.add_argument( '--epochs', type=int, default=1 )
    parser.add_argument( '--batches', type=int, default=64 )
    args = parser.parse_args()

    train, validation, test = load_train_validation_test(val_split=0.1, batch_size=args.batches)

    net = model(10, input_shape=(32, 32, 3))

    log_dir = 'logs/fit/{}'.format( args.experiment_name )
    log_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1 )

    checkpoint_path = "checkpoints/{}".format( args.experiment_name )
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)    

    net.fit(train, validation_data=validation, epochs=args.epochs, callbacks=[log_callback, cp_callback])
    
    

