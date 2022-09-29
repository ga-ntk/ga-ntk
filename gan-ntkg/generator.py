# %%
import tensorflow as tf
import numpy as np
from tensorflow._api.v2 import data

# %%
layers = tf.keras.layers
z_dim = 64
# %%
def get_generator(dataset_name):
    Generator = None
    if 'gaussian' in dataset_name:
        Generator = tf.keras.Sequential([
        layers.Dense(units=64, input_shape = (2,)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dense(units=64),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dense(units=2, activation=tf.keras.activations.sigmoid)
    ])
    elif dataset_name =='mnist':
        Generator = tf.keras.Sequential([
        layers.Dense(units=1024, input_shape = (z_dim,)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dense(units=6272),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Reshape(target_shape=(7, 7, 128)),
        layers.Conv2DTranspose(
            filters=64,
            kernel_size=4,
            strides=2,
            padding="SAME"
        ),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(
            filters=1,
            kernel_size=4,
            strides=2,
            padding="SAME",
            activation=tf.keras.activations.sigmoid
        )
    ])
    elif dataset_name == 'cifar10':
        Generator = tf.keras.Sequential([
        layers.Dense(units=1792, input_shape = (z_dim,)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Reshape(target_shape=(2, 2, 448)),
        layers.Conv2DTranspose(
            filters=256,
            kernel_size=4,
            strides=2,
            padding="SAME"
        ),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(
            filters=128,
            kernel_size=4,
            strides=2,
            padding="SAME"
        ),
        layers.ReLU(),
        layers.Conv2DTranspose(
            filters=64,
            kernel_size=4,
            strides=2,
            padding="SAME"
        ),
        layers.ReLU(),
        layers.Conv2DTranspose(
            filters=3,
            kernel_size=4,
            strides=2,
            padding="SAME",
            activation=tf.keras.activations.tanh
        )
    ])
    elif dataset_name == 'celeb_a':
        Generator = tf.keras.Sequential([
        layers.Dense(units=1792, input_shape = (z_dim,)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Reshape(target_shape=(2, 2, 448)),
        layers.Conv2DTranspose(
            filters=256,
            kernel_size=4,
            strides=2,
            padding="SAME"
        ),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(
            filters=128,
            kernel_size=4,
            strides=2,
            padding="SAME"
        ),
        layers.ReLU(),
        layers.Conv2DTranspose(
            filters=64,
            kernel_size=4,
            strides=2,
            padding="SAME"
        ),
        layers.ReLU(),
        layers.Conv2DTranspose(
            filters=32,
            kernel_size=4,
            strides=2,
            padding="SAME"
        ),
        layers.ReLU(),
        layers.Conv2DTranspose(
            filters=3,
            kernel_size=4,
            strides=2,
            padding="SAME",
            activation=tf.keras.activations.tanh
        )
    ])
    elif dataset_name == 'imagenet':
        Generator = tf.keras.Sequential([
        layers.Dense(units=16384, input_shape = (z_dim,)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Reshape(target_shape=(4, 4, 1024)),
        layers.Conv2DTranspose(
            filters=1024,
            kernel_size=4,
            strides=2,
            padding="SAME"
        ),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(
            filters=512,
            kernel_size=4,
            strides=2,
            padding="SAME"
        ),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(
            filters=256,
            kernel_size=4,
            strides=2,
            padding="SAME"
        ),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(
            filters=128,
            kernel_size=4,
            strides=2,
            padding="SAME"
        ),
        layers.ReLU(),
        layers.Conv2DTranspose(
            filters=3,
            kernel_size=4,
            strides=2,
            padding="SAME",
            activation=tf.keras.activations.tanh
        )
    ])
    return Generator