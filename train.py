from os import name
from warnings import filters
import tensorflow as tf
import numpy as np

from argparse import ArgumentParser
from tensorflow import keras
from tensorflow.python.ops.gen_math_ops import ceil

from model_cifar10 import model_cifar10
from model_cifar100 import model_cifar100, load_backbone_from_cifar10_model

def load_train_validation_test(dataset, val_split=0.1, batch_size=1):
    (train_x, train_y), (test_x, test_y) = dataset.load_data()
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

    dataset = keras.datasets.cifar100
    train, validation, test = load_train_validation_test(dataset, val_split=0.0, batch_size=args.batches)
    validation = test

    net_100 = model_cifar100(input_shape=(32, 32, 3), augmentation=True)

    if args.restore_from:
        ckpt_path = "checkpoints/{}".format( args.restore_from )
        net_100.load_weights( ckpt_path )

    print("validation", net_100.evaluate(validation) )
    #net_100 = load_backbone_from_cifar10_model( net_10, trainable=False, input_shape=(32, 32, 3), augmentation=True)

    if not args.metrics_only:
        log_dir = 'logs/fit/{}'.format( args.experiment_name )
        log_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1 )

        checkpoint_path = f"checkpoints/{args.experiment_name}/" + "{epoch:02d}.ckpt"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True)    

        
        net_100.fit(train, validation_data=validation, epochs=args.epochs, callbacks=[log_callback, cp_callback])

    print("train", net_100.evaluate(train) )
    print("validation", net_100.evaluate(validation) )
    print("test", net_100.evaluate(test) )
    
    

