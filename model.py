# -*- coding: utf-8 -*-
from datetime import time
from typing import Callable, Tuple
from jax import random, devices, soft_pmap
from neural_tangents import stax
import numpy as onp
import jax.numpy as np
from jax.api import jit, vmap, pmap
from tqdm import tqdm
import neural_tangents as nt
from util import Perturbation, TrendRecorder
from dataset import NoiseGenerator, get_labels
from loss_fn import LossFn
from plot import Plotter

class Model():
    FNN = "fnn"
    CNN = "cnn"
    CNN_BIG = "cnn-big"
    CNN_MNIST = "cnn-mnist"
    CNN_CIFAR10 = "cnn-cifar10"
    FNN_4 = "fnn-4"
    CNN_4 = "cnn-4"
    FNN_16 = "fnn-16"
    CNN_16 = "cnn-16"

    FLATTEN_KEYWORD = 'fnn'

    def __init__(self, W_std=onp.sqrt(1.4615), b_std=onp.sqrt(0.1)):
        self.W_std = W_std
        self.b_std = b_std

    def surrogate_fn_jit(self, model_type: str):
        init_fn, apply_fn, kernel_fn = self.surrogate_fn(model_type=model_type)
        return init_fn, jit(apply_fn), jit(kernel_fn, static_argnums=(2,))

    @staticmethod
    def get_flatten(model_type: str):
        if Model.FLATTEN_KEYWORD in model_type:
            return True
        else:
            return False

    def surrogate_fn(self, model_type: str):
        """
        :param model_type: string. `fnn` or `cnn`.
        :param W_std: float. standard deviation of weights at initialization.
        :param b_std: float. standard deviation of biases at initialization.
        :return: triple of callable functions (init_fn, apply_fn, kernel_fn).
                In Neural Tangents, a network is defined by a triple of functions (init_fn, apply_fn, kernel_fn). 
                init_fn: a function which initializes the trainable parameters.
                apply_fn: a function which computes the outputs of the network.
                kernel_fn: a kernel function of the infinite network (GP) of the given architecture 
                        which computes the kernel matrix
        """
        if model_type == self.FNN:
            init_fn, apply_fn, kernel_fn = stax.serial(
                stax.Dense(64, self.W_std, self.b_std),
                stax.Relu(),
                stax.Dense(64, self.W_std, self.b_std),
                stax.Relu(),
                stax.Dense(10, self.W_std, self.b_std)
            )
        elif model_type == self.CNN:
            init_fn, apply_fn, kernel_fn = stax.serial(
                stax.Conv(
                    out_chan=64, filter_shape=(4, 4), strides=(2, 2), 
                    padding='SAME', W_std=self.W_std, b_std=self.b_std), 
                stax.Relu(),
                stax.Conv(
                    out_chan=64, filter_shape=(4, 4), strides=(2, 2), 
                    padding='SAME', W_std=self.W_std, b_std=self.b_std), 
                stax.Relu(),
                stax.Conv(
                    out_chan=64, filter_shape=(4, 4), strides=(2, 2), 
                    padding='SAME', W_std=self.W_std, b_std=self.b_std), 
                stax.Relu(),
                stax.Conv(
                    out_chan=64, filter_shape=(4, 4), strides=(2, 2), 
                    padding='SAME', W_std=self.W_std, b_std=self.b_std), 
                stax.Relu(),
                stax.Flatten(),
                stax.Dense(10, self.W_std, self.b_std)
            )
        elif model_type == self.CNN_BIG:
            # MODIFIED: 
            init_fn, apply_fn, kernel_fn = stax.serial(
                stax.Conv(
                    out_chan=64, filter_shape=(4, 4), strides=(2, 2), 
                    padding='SAME', W_std=self.W_std, b_std=self.b_std), 
                stax.Relu(),
                stax.Conv(
                    out_chan=64, filter_shape=(4, 4), strides=(2, 2), 
                    padding='SAME', W_std=self.W_std, b_std=self.b_std), 
                stax.Relu(),
                stax.Conv(
                    out_chan=64, filter_shape=(4, 4), strides=(2, 2), 
                    padding='SAME', W_std=self.W_std, b_std=self.b_std), 
                stax.Relu(),
                stax.Conv(
                    out_chan=64, filter_shape=(4, 4), strides=(2, 2), 
                    padding='SAME', W_std=self.W_std, b_std=self.b_std), 
                stax.Relu(),
                stax.Conv(
                    out_chan=64, filter_shape=(4, 4), strides=(2, 2), 
                    padding='SAME', W_std=self.W_std, b_std=self.b_std), 
                stax.Relu(),
                stax.Conv(
                    out_chan=64, filter_shape=(4, 4), strides=(2, 2), 
                    padding='SAME', W_std=self.W_std, b_std=self.b_std), 
                stax.Relu(),
                stax.Flatten(),
                stax.Dense(10, self.W_std, self.b_std)
            )
        elif model_type == self.CNN_MNIST:
            init_fn, apply_fn, kernel_fn = stax.serial(
                stax.Conv(
                    out_chan=64, filter_shape=(4, 4), strides=(2, 2), 
                    padding='SAME', W_std=self.W_std, b_std=self.b_std), 
                stax.Relu(),
                stax.Conv(
                    out_chan=128, filter_shape=(4, 4), strides=(2, 2), 
                    padding='SAME', W_std=self.W_std, b_std=self.b_std), 
                stax.Relu(),
                stax.Flatten(),
                stax.Dense(1024, self.W_std, self.b_std),
                stax.Relu(),
                stax.Dense(1, self.W_std, self.b_std)
            )
        elif model_type == self.CNN_CIFAR10:
            init_fn, apply_fn, kernel_fn = stax.serial(
                stax.Conv(
                    out_chan=64, filter_shape=(4, 4), strides=(2, 2), 
                    padding='SAME', W_std=self.W_std, b_std=self.b_std), 
                stax.Relu(),
                stax.Conv(
                    out_chan=128, filter_shape=(4, 4), strides=(2, 2), 
                    padding='SAME', W_std=self.W_std, b_std=self.b_std), 
                stax.Relu(),
                stax.Conv(
                    out_chan=256, filter_shape=(4, 4), strides=(2, 2), 
                    padding='SAME', W_std=self.W_std, b_std=self.b_std), 
                stax.Relu(),
                stax.Flatten(),
                stax.Dense(1, self.W_std, self.b_std)
            )
        elif model_type == self.FNN_4:
            NN = []
            for _ in range(3):
                NN += [
                    stax.Dense(64, self.W_std, self.b_std),
                    stax.Relu()]
            
            NN += [stax.Dense(64, self.W_std, self.b_std)]
            init_fn, apply_fn, kernel_fn = stax.serial(*NN)
            
        elif model_type == self.FNN_16:
            NN = []
            for _ in range(15):
                NN += [
                    stax.Dense(64, self.W_std, self.b_std),
                    stax.Relu()]
                
            NN += [stax.Dense(64, self.W_std, self.b_std)]
            init_fn, apply_fn, kernel_fn = stax.serial(*NN)
            
        elif model_type == self.CNN_4:
            NN = [
                stax.Conv(
                    out_chan=64, filter_shape=(7, 7), strides=(1, 1), 
                    padding='SAME', W_std=self.W_std, b_std=self.b_std), 
                stax.Relu(),
                stax.Conv(
                    out_chan=64, filter_shape=(3, 3), strides=(1, 1), 
                    padding='SAME', W_std=self.W_std, b_std=self.b_std), 
                stax.Relu(),
                stax.Conv(
                    out_chan=64, filter_shape=(3, 3), strides=(2, 2), 
                    padding='SAME', W_std=self.W_std, b_std=self.b_std), 
                stax.Relu()
            ]   
            NN += [
                stax.Flatten(),
                stax.Dense(10, self.W_std, self.b_std)]
            
            init_fn, apply_fn, kernel_fn = stax.serial(*NN)
            
        elif model_type == self.CNN_16:
            NN = [
                stax.Conv(
                    out_chan=64, filter_shape=(7, 7), strides=(1, 1), 
                    padding='SAME', W_std=self.W_std, b_std=self.b_std), 
                stax.Relu(),
                stax.Conv(
                    out_chan=64, filter_shape=(3, 3), strides=(1, 1), 
                    padding='SAME', W_std=self.W_std, b_std=self.b_std), 
                stax.Relu(),
                stax.Conv(
                    out_chan=64, filter_shape=(3, 3), strides=(2, 2), 
                    padding='SAME', W_std=self.W_std, b_std=self.b_std), 
                stax.Relu()
            ]
            for _ in range(4):
                NN += [
                    stax.Conv(
                        out_chan=64, filter_shape=(3, 3), strides=(1, 1), 
                        padding='SAME', W_std=self.W_std, b_std=self.b_std),
                    stax.Relu(),
                    stax.Conv(
                        out_chan=64, filter_shape=(3, 3), strides=(1, 1), 
                        padding='SAME', W_std=self.W_std, b_std=self.b_std), 
                    stax.Relu(),
                    stax.Conv(
                        out_chan=64, filter_shape=(3, 3), strides=(2, 2), 
                        padding='SAME', W_std=self.W_std, b_std=self.b_std),
                    stax.Relu(),
                ]
                
            NN += [
                stax.Flatten(),
                stax.Dense(10, self.W_std, self.b_std)]
            
            init_fn, apply_fn, kernel_fn = stax.serial(*NN)
        else:
            raise ValueError(f'No such model {model_type}')
        return init_fn, apply_fn, kernel_fn

class NTK_Generative():

    PREDICT_TIMES_AUTO = 100

    def __init__(self, model_type: str, loss_type: str, learning_rate: float, t: float, seed: int, perturb_method: str=None, perturb_coef: float=None, alpha: float=0):
        self.model_type = model_type
        self.loss_type = loss_type
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.perturb_method = perturb_method
        self.perturb_coef = perturb_coef
        self.t = t
        self.seed = seed

    def __process_epoch_output(self, x_noise_concat: np.ndarray, g_res: np.ndarray, p_res: np.ndarray, loss_res: np.ndarray, epoch_res: np.ndarray, ntk_prediction_res: np.ndarray, is_compute_prediction_res: bool):
        # Reshape 
        # x_noise_concat = np.reshape(x_noise_res, (parallel_num * noise_size, *image_shape))
        gs_mean = np.mean(g_res, axis=0)
        ps_mean = np.mean(p_res, axis=0)
        losses_mean = np.mean(loss_res, axis=0)
        epochs_mean = np.mean(epoch_res, axis=0, dtype=np.int32)
        ntk_predictions_mean = np.mean(ntk_prediction_res, axis=0)
        is_compute_prediction_any = np.any(is_compute_prediction_res, axis=0)

        return x_noise_concat, gs_mean, ps_mean, losses_mean, epochs_mean, ntk_predictions_mean, is_compute_prediction_any

    
    def generate_v2(self, x_train: np.ndarray, x_train_all: np.ndarray, noise_size: int, epoch: int, parallel_num: int=1, predict_per_epoch: int=-1, callback: Callable=None):
        train_size = x_train.shape[0]
        dataset_size = x_train_all.shape[0]
        image_shape = x_train_all[0].shape
        labels, target_label = get_labels(train_size=train_size, noise_size=noise_size)

        init_fn, apply_fn, kernel_fn = Model().surrogate_fn_jit(model_type=self.model_type)
        grad_fn, loss_fn, get_pred_fn = LossFn(loss_type=self.loss_type, alpha=self.alpha, target_label=target_label, seed=1).get_fns_jit()

        _predict_per_epoch_train = predict_per_epoch
        if _predict_per_epoch_train is not None:
            # Set prediction times automatically
            if _predict_per_epoch_train == -1:
                _predict_per_epoch_train = int(epoch // self.PREDICT_TIMES_AUTO)
                if _predict_per_epoch_train <= 1:
                    _predict_per_epoch_train = 1
        
        ng = NoiseGenerator(random_seed=self.seed)
        x_noises = np.stack([ng.gen_noise(noise_size=noise_size, image_shape=image_shape) for i in range(parallel_num)], axis=0)
        perturb_generator = Perturbation(coef=self.perturb_coef, method=self.perturb_method, seed=self.seed)
        
        def train(x_noise):
            # sample a batch SGD
            if train_size is None:
                x_train_batch = x_train_all
            else:
                index = onp.random.choice(dataset_size, size=train_size, replace=False)
                x_train_batch = x_train_all[index]
                
            # calculate gradient
            grads = grad_fn(x_noise, x_train_batch, labels, kernel_fn, self.t)

            # Compuet perturbation
            perturb = perturb_generator.generate(shape=x_noise.shape)
            grad_accumulator = grads * self.learning_rate + self.learning_rate * perturb
            
            # update
            x_noise_update = x_noise
            x_noise_update -= grad_accumulator
            
            # clip back to image values
            x_noise_update = np.clip(x_noise_update, a_min=0.0, a_max=1.0)

            # record L2 norm of grads
            g = grads.reshape(noise_size, -1)
            g = np.mean(np.linalg.norm(g, axis=1))
            
            # record L2 norm of perturbation
            p = perturb.reshape(noise_size, -1)
            p = np.mean(np.linalg.norm(p, axis=1))
            
            # record loss
            loss = loss_fn(x_noise_update, x_train, labels, kernel_fn, self.t)

            # NTK Predictions
            output = 0
            is_compute_prediction = False
            if _predict_per_epoch_train is not None:
                # Predict
                if (i % _predict_per_epoch_train) == 0:
                    pred_fn = get_pred_fn(x_noise=x_noise_update, x_train=x_train, labels=labels, kernel_fn=kernel_fn)
                    output = pred_fn(self.t)
                    output = output[-noise_size:]
                    output = np.mean(output[:, 1])
                    is_compute_prediction = True
                    
            return x_noise_update, g, p, loss, i, output, is_compute_prediction

        train_map = vmap(train)

        for i in tqdm(range(1, epoch+1)):
            # Parallelize
            x_noise_res, g_res, p_res, loss_res, epoch_res, ntk_prediction_res, is_compute_prediction_res, = train_map(x_noises)

            # Reshape 
            x_noise_concat = np.reshape(x_noise_res, (parallel_num * noise_size, *image_shape))

            # Callback
            callback(*self.__process_epoch_output(x_noise_concat=x_noise_concat, g_res=g_res, p_res=p_res, loss_res=loss_res, 
                    epoch_res=epoch_res, ntk_prediction_res=ntk_prediction_res, is_compute_prediction_res=is_compute_prediction_res))

            # Update x_noises
            x_noises = x_noise_res

        return x_noise_concat
