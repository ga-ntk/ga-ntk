# %%
import tensorflow as tf
import keras
import os
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)
gpu_id = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

# %%
import argparse

import matplotlib.pyplot as plt
import numpy as np

from dataset import Dataset
from generator import get_generator

from gan.gantkgan_generator import ga_ntk_gan_training_loop
# %%
debug = False

# %%
# cmd arguments
if not debug:
    parser = argparse.ArgumentParser(description='NTK image generation.')
    parser.add_argument('--related_work', type=str, default='ga-ntk-gan',
                        help='ga-ntk-gan')
    parser.add_argument('--dataset_name', type=str,
                        help='dataset_name = mnist | cifar10 | celeb_a')
    parser.add_argument('--dataset_size', type=int,
                        help='dataset set size')
    parser.add_argument('--train_size', type=int,
                        help='training set size')
    parser.add_argument('--epoch', type=int,
                        help='iterations for training')
    parser.add_argument('--target_distribution', type=str,
                        help='target_distribution = single | all')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size for training')
    parser.add_argument('--z_dim', type=int, default=64,
                        help='z_dim of generator')
    parser.add_argument('--train_t_rate', type=float,
                        help='the training time of the NTK, multiply with 65536.0')
    parser.add_argument('--diag_reg', type=float, default=1e-4,
                        help='diag_reg for neural-tangent gradient_descent_mse function')
    parser.add_argument('--g_lr', type=float, default=2e-4,
                        help='learning rate for generator')
    parser.add_argument('--imagenet_path', type=str, default=None)
    parser.add_argument('--celeb_a_img_path', type=str, default=None)
    parser.add_argument('--celeb_a_anno_path', type=str, default=None)
    parser.add_argument('--gpu_id', type=str,
                        help='gpu_id')
    args = parser.parse_args()
# %%
if debug:
    related_work = 'ga-ntk-gan'
    dataset_name = 'celeb_a'
    dataset_size = 2048
    train_size = 2048
    epoch = 10000
    target_distribution = 'single'
    batch_size = 2048
    z_dim = 64
    train_t_rate = 1
    diag_reg = 1e-4
    g_lr = 5e-4
    imagenet_path = None
    celeb_a_img_path = None
    celeb_a_anno_path = None
    gpu_id = '0'
else:
    related_work = args.related_work
    dataset_name = args.dataset_name
    dataset_size = args.dataset_size
    train_size = args.train_size
    epoch = args.epoch
    target_distribution = args.target_distribution
    batch_size = args.batch_size
    z_dim = args.z_dim
    train_t_rate = args.train_t_rate
    diag_reg = args.diag_reg
    g_lr = args.g_lr
    imagenet_path = args.imagenet_path
    celeb_a_img_path = args.celeb_a_img_path
    celeb_a_anno_path = args.celeb_a_anno_path
    gpu_id = args.gpu_id

# %%
flatten = True
num_classes = 10
sel_features = None
target_class = None
if target_distribution == 'single':
    if dataset_name == 'celeb_a':
        sel_features = ['Male', 'Straight_Hair']
    elif dataset_name == 'imagenet':
        sel_features = ['daisy']
    elif dataset_name == 'celeb_a_large':
        sel_features = ['Male', 'Straight_Hair']
    target_class = 7
seed = 0

image_shape = None

if dataset_name == 'mnist':
    image_shape = (28, 28, 1)
    vec_size = image_shape[0] * image_shape[1] * image_shape[2]
elif dataset_name == 'cifar10':
    image_shape = (32, 32, 3)
    vec_size = image_shape[0] * image_shape[1] * image_shape[2]
elif dataset_name == 'celeb_a':
    image_shape = (64, 64, 3)
    vec_size = image_shape[0] * image_shape[1] * image_shape[2]
elif dataset_name == 'imagenet':
    image_shape = (128, 128, 3)
    vec_size = image_shape[0] * image_shape[1] * image_shape[2]
elif dataset_name == 'gaussian_8' or dataset_name == 'gaussian_25':
    image_shape = (2, )
    vec_size = image_shape[0]
elif dataset_name == 'celeb_a_large':
    image_shape = (256, 256, 3)
    vec_size = image_shape[0] * image_shape[1] * image_shape[2]

dataset_generator = Dataset(dataset_name=dataset_name, celeb_a_path=celeb_a_img_path, imagenet_path=imagenet_path, seed=1, flatten=False)
dataset_generator.set_sample_size(noise_size=None, train_size=train_size, dataset_size=dataset_size)
image_shape, vec_size = dataset_generator.get_data_shape()
x_train, x_train_all = dataset_generator.gen_data_attrs(target_class=target_class, attrs=sel_features)

if dataset_name in ['cifar10', 'celeb_a', 'celeb_a_large','imagenet']:
    x_train = (x_train * 2) - 1
    x_train_all = (x_train_all * 2) - 1
if dataset_name in ['celeb_a_large']:
    x_train = x_train[:, :, :, [2, 1, 0]]
    x_train_all = x_train_all[:, :, :, [2, 1, 0]]
# %%
G = get_generator(dataset_name)
D = None

# %%
# Data
ds_train = tf.data.Dataset.from_tensor_slices(x_train).shuffle(train_size * batch_size).batch(batch_size, drop_remainder=True).prefetch(train_size)
# %%
if related_work == 'ga-ntk-gan':
    t = train_t_rate * 65536.0
    ga_ntk_gan_training_loop(D, G, ds_train, train_size, target_distribution, ds_name=dataset_name, t=t, epoch=10, batch_size=batch_size, z_dim=z_dim, g_lr=g_lr, diag_reg=diag_reg)
else:
    raise BaseException('related work not implemented!')
