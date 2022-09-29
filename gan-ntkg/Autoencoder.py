# +
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os

dataset_name = 'celeb_a_large'
# -

path_to = None
image_size = None
if dataset_name == 'celeb_a':
    path_to = "PATH TO CELEB_A"
    image_size = (64, 64)
elif dataset_name == 'celeb_a_large':
    path_to = "PATH TO CELEB_A HQ"
    image_size = (256, 256)

dataset = keras.preprocessing.image_dataset_from_directory(
    path_to, label_mode=None, image_size=image_size, seed=123, batch_size=128
)
def change_inputs(x):
    return x, x
dataset = dataset.map(lambda x: x / 255.0)
dataset = dataset.map(lambda x: ((x*2)-1))
dataset = dataset.map(change_inputs)

if dataset_name == 'mnist':
    input_shape = (28, 28, 1)
elif dataset_name == 'cifar10':
    input_shape = (32, 32, 3)
elif dataset_name == 'celeb_a':
    input_shape = (64, 64, 3)
elif dataset_name == 'celeb_a_large':
    input_shape = (256, 256, 3)


# +
@tf.function
def ssim_loss(y_true, y_pred):
    return 1-tf.reduce_mean(tf.image.ssim(y_true, y_pred, 
                                          max_val = 1.0,filter_size=11,
                                        filter_sigma=1.5, k1=0.01, k2=0.03 ))

@tf.function
def ssim_l1_loss(y_true, y_pred, max_val=1.0):
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 
                                          max_val = 1.0,filter_size=11,
                                        filter_sigma=1.5, k1=0.01, k2=0.03 ))
    L1 = tf.reduce_mean(tf.abs(y_true - y_pred))
    return ssim_loss + L1


# +
input_img = keras.Input(shape=input_shape)
x = layers.Conv2D(16, (3, 3), strides=1, activation='selu', padding='same')(input_img)
x = layers.Conv2D(16, (3, 3), strides=1, activation='selu', padding='same')(x)
x = layers.Conv2D(16, (3, 3), strides=1, activation='selu', padding='same')(x)
x = layers.Conv2D(32, (2, 2), strides=2, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation(keras.activations.selu)(x)
x = layers.Conv2D(32, (3, 3), strides=1, activation='selu', padding='same')(x)
x = layers.Conv2D(32, (3, 3), strides=1, activation='selu', padding='same')(x)
x = layers.Conv2D(32, (3, 3), strides=1, activation='selu', padding='same')(x)
x = layers.Conv2D(64, (2, 2), strides=2, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation(keras.activations.selu)(x)
x = layers.Conv2D(64, (3, 3), strides=1, activation='selu', padding='same')(x)
x = layers.Conv2D(64, (3, 3), strides=1, activation='selu', padding='same')(x)
x = layers.Conv2D(64, (3, 3), strides=1, activation='selu', padding='same')(x)
x = layers.Conv2D(128, (2, 2), strides=2, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation(keras.activations.selu)(x)
x = layers.Conv2D(128, (3, 3), strides=1, activation='selu', padding='same')(x)
x = layers.Conv2D(128, (3, 3), strides=1, activation='selu', padding='same')(x)
x = layers.Conv2D(128, (3, 3), strides=1, activation='selu', padding='same')(x)
x = layers.Conv2D(256, (2, 2), strides=2, activation='tanh', padding='same')(x)
#bottleneck
x = layers.Conv2DTranspose(256, (2, 2), strides=2, activation='selu', padding='same')(x)
x = layers.Conv2D(128, (3, 3), strides=1, activation='selu', padding='same')(x)
x = layers.Conv2D(128, (3, 3), strides=1, activation='selu', padding='same')(x)
x = layers.Conv2D(128, (3, 3), strides=1, activation='selu', padding='same')(x)
x = layers.Conv2DTranspose(128, (2, 2), strides=2, activation='selu', padding='same')(x)
x = layers.Conv2D(64, (3, 3), strides=1, activation='selu', padding='same')(x)
x = layers.Conv2D(64, (3, 3), strides=1, activation='selu', padding='same')(x)
x = layers.Conv2D(64, (3, 3), strides=1, activation='selu', padding='same')(x)
x = layers.Conv2DTranspose(64, (2, 2), strides=2, activation='selu', padding='same')(x)
x = layers.Conv2D(32, (3, 3), strides=1, activation='selu', padding='same')(x)
x = layers.Conv2D(32, (3, 3), strides=1, activation='selu', padding='same')(x)
x = layers.Conv2D(32, (3, 3), strides=1, activation='selu', padding='same')(x)
x = layers.Conv2DTranspose(32, (2, 2), strides=2, activation='selu', padding='same')(x)
x = layers.Conv2D(16, (3, 3), strides=1, activation='selu', padding='same')(x)
x = layers.Conv2D(16, (3, 3), strides=1, activation='selu', padding='same')(x)
x = layers.Conv2D(16, (3, 3), strides=1, activation='selu', padding='same')(x)
decoded = layers.Conv2D(3, (3, 3), strides=1, activation='tanh', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss=ssim_l1_loss)
autoencoder.summary()
# -

checkpoint = ModelCheckpoint(
     'celeb_a_large_best.h5', 
     monitor='loss', 
     verbose=1, 
     save_best_only=True, 
     mode='min', 
     save_freq="epoch"
)

if dataset_name == 'mnist':
    autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=512,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder'), checkpoint])
elif dataset_name == 'cifar10':
    autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=512,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder'), checkpoint])
elif dataset_name == 'celeb_a':
    autoencoder.fit(dataset,
                epochs=100,
                batch_size=512,
                shuffle=True,
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder'), checkpoint])
elif dataset_name == 'celeb_a_large':
    autoencoder.fit(dataset,
                epochs=1000,
                batch_size=512,
                shuffle=True,
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder'), checkpoint])

autoencoder = keras.models.load_model('celeb_a_large_best.h5', compile=False)
opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
autoencoder.compile(optimizer=opt, loss=ssim_l1_loss)

# +
from matplotlib import pyplot as plt
if dataset_name == 'celeb_a':
    x_test = []
    for i, _ in dataset.take(1):
        x_test = i
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (-1, 64, 64, 3))
elif dataset_name == 'celeb_a_large':
    x_test = []
    for i, _ in dataset.take(1):
        x_test = _
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (-1, 256, 256, 3))
    
decoded_imgs = autoencoder.predict(x_test)

if dataset_name != 'mnist': 
    decoded_imgs = (decoded_imgs+1)/2
    x_test = (x_test+1)/2
    
n = 10
plt.figure(figsize=(20, 4))
# x_test = x_test[:, :, :, [2, 1, 0]]
t = None
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(input_shape))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    t= i
    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(input_shape))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
# -

autoencoder.save('celebal_encoder')
