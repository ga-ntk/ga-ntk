# -*- coding: utf-8 -*-
# %%
import matplotlib.pyplot as plt
import neural_tangents as nt
import tensorflow as tf
import numpy as np
import jax.numpy as jnp
import os
import jax
import keras
from functools import partial
from neural_tangents import stax

from jax import random
from jax import grad, jit
from jax.config import config
from tqdm import tqdm

from util import utPuzzle, save_images_samples
from dataset import Mixed_Gaussian
# %%
def load_ae(ds_name):
    if ds_name == 'mnist':
        autoencoder = keras.models.load_model('PATH to trained encoder in ../Autoencoder.py', compile=False)
        lay_num = 9
    elif ds_name == 'cifar10':
        autoencoder = keras.models.load_model('PATH to trained encoder in ../Autoencoder.py', compile=False)
        lay_num = 8
    elif ds_name == 'celeb_a':
        autoencoder = keras.models.load_model('PATH to trained encoder in ../Autoencoder.py', compile=False)
        lay_num = 10
    elif ds_name == 'celeb_a_large':
        autoencoder = keras.models.load_model('PATH to trained encoder in ../Autoencoder.py', compile=False)
        lay_num = 28
    for i in range(len(autoencoder.layers)):
        autoencoder.layers[i].trainable = False
    encoder = keras.Model(inputs=autoencoder.input, outputs=autoencoder.layers[lay_num].output)
    decoder = keras.Model(inputs=autoencoder.layers[lay_num].output, outputs=autoencoder.output)
    return encoder, decoder


# %%
def as_jax(x):
    return jnp.asarray(np.asarray(x))

def as_tf(x):
    return tf.convert_to_tensor(np.asarray(x))

def _one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

def get_labels(train_size: int, noise_size: int):
    labels = jnp.concatenate([
        jnp.ones((train_size,), jnp.float32), 
        jnp.zeros((noise_size,), jnp.float32)
    ])
    labels = _one_hot(labels, 2, dtype=jnp.float32)

    target_label = jnp.ones((train_size + noise_size,), jnp.float32)
    target_label = _one_hot(target_label, 2, dtype=jnp.float32)

    return labels, target_label
# %%
class Model():
    FNN = ["fnn", "mnist", "cifar10", "celeb_a", "celeb_a_large"]
    CNN_BIG = ["cnn-big"]
    #CNN_CELEBA = ["celeb_a_large"]
    #FNN = "mnist"
    #CNN_MNIST = "mnist"
    #FNN = "cifar10"
    #CNN_CIFAR10 = "cifar10"
    CNN_4 = ["cnn-4"]
    FNN_4 = ["fnn-4"]

    FLATTEN_KEYWORD = 'fnn'

    def __init__(self, W_std=np.sqrt(1.4615), b_std=np.sqrt(0.1)):
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
        if model_type in self.FNN:
            init_fn, apply_fn, kernel_fn = stax.serial(
                stax.Flatten(),
                stax.Dense(64, self.W_std, self.b_std),
                stax.Erf(),
                stax.Dense(64, self.W_std, self.b_std),
                stax.Erf(),
                stax.Dense(64, self.W_std, self.b_std),
                stax.Erf(),
                stax.Dense(64, self.W_std, self.b_std),
                stax.Erf(),
                stax.Dense(1, self.W_std, self.b_std)
            )
        elif model_type in self.CNN_CELEBA:
            init_fn, apply_fn, kernel_fn = stax.serial(
                stax.Conv(
                    out_chan=64, filter_shape=(3, 3), strides=(1, 1), 
                    padding='SAME', W_std=self.W_std, b_std=self.b_std), 
                stax.Erf(),
                stax.Conv(
                    out_chan=64, filter_shape=(3, 3), strides=(1, 1), 
                    padding='SAME', W_std=self.W_std, b_std=self.b_std), 
                stax.Erf(),
                stax.Conv(
                    out_chan=64, filter_shape=(3, 3), strides=(1, 1), 
                    padding='SAME', W_std=self.W_std, b_std=self.b_std), 
                stax.Erf(),
                stax.Conv(
                    out_chan=64, filter_shape=(3, 3), strides=(1, 1), 
                    padding='SAME', W_std=self.W_std, b_std=self.b_std), 
                stax.Erf(),
                stax.Flatten(),
                stax.Dense(1, self.W_std, self.b_std))
            
        elif model_type in self.CNN_BIG:
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
                stax.Dense(1, self.W_std, self.b_std))
        elif model_type in self.CNN_MNIST:
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
        elif model_type in self.CNN_CIFAR10:
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
        elif model_type in self.FNN_4:
            NN = []
            for _ in range(3):
                NN += [
                    stax.Dense(64, self.W_std, self.b_std),
                    stax.Relu()]
            
            NN += [stax.Dense(64, self.W_std, self.b_std)]
            init_fn, apply_fn, kernel_fn = stax.serial(*NN)
        else:
            raise ValueError(f'No such model {model_type}')
        return init_fn, apply_fn, kernel_fn

# %%
class NTK_Generative():
    def __init__(
        self,
        train_size: int,
        model_type: str,
        t: float,
        diag_reg: float
    ):
        # assume train_size = noise_size
        self.train_size = train_size
        self.noise_size = train_size
        self.model_type = model_type
        self.t = t
        self.diag_reg = diag_reg
        labels, target_labels = get_labels(train_size=train_size, noise_size=train_size)
        self.labels = labels
        self.target_labels = target_labels

    def get_loss_fn_partial(self, ntk_learning_rate=1e-2):
        _, _, kernel_fn = Model().surrogate_fn_jit(model_type=self.model_type)
        def pred_fn(x_noise, x_train, labels, kernel_fn, t):
            return nt.predict.gradient_descent_mse(
                    kernel_fn(jnp.concatenate([x_train, x_noise]), None, 'ntk'),
                    labels,
                    learning_rate=ntk_learning_rate,
                    diag_reg=self.diag_reg)(t)
        
        def cross_entropy(fx, y_hat): 
            return -jnp.mean(jnp.sum(jax.nn.log_softmax(fx) * y_hat, axis=1))
    
        def loss_fn(x_noise, x_train, labels, target_labels, kernel_fn, t):
            output = pred_fn(x_noise, x_train, labels, kernel_fn, t)
            return cross_entropy(output, target_labels)
        
        pred_fn_partial = partial(pred_fn, labels=self.labels, kernel_fn=kernel_fn, t=self.t)
        loss_fn_partial = partial(loss_fn, labels=self.labels, target_labels=self.target_labels, kernel_fn=kernel_fn, t=self.t)
        
        return pred_fn_partial, loss_fn_partial

# %%
def wrap_ga_ntk_in_tf(ga_ntk_fn):
    """
    :param ga_ntk_fn_partial: jax GA-NTK functions implement with Neural Tangents.
    :return: tf version of GA-NTK functions. 
    
    NOTE: we only calculate gradients of x_noise and ignore gradient of x_train 
    for performance reasons.
    """
    ga_ntk_fn = jit(ga_ntk_fn)
    grad_ga_ntk_fn = jit(grad(ga_ntk_fn, argnums=0))
    @tf.custom_gradient
    def tf_ga_ntk_fn(x_noise, x_train):
        jax_x_noise = as_jax(x_noise)
        jax_x_train = as_jax(x_train)
        loss = ga_ntk_fn(jax_x_noise, jax_x_train)

        def tf_grad_ga_ntk_fn(upstream):
            """
            NOTE: this only calculate gradients of x_noise.
            Ignore gradient of x_train for performance reasons.
            """
            jax_x_noise_grad = grad_ga_ntk_fn(jax_x_noise, jax_x_train)
            return as_tf(jax_x_noise_grad), None
        
        return as_tf(loss), tf_grad_ga_ntk_fn
    return tf_ga_ntk_fn


# %%
def G_Train(
    c1,
    tf_ntk_loss_fn_partial,
    G,
    optimizer_d,
    optimizer_g,
    loss_fn,
    encoder,
    batch_size=64,
    z_dim=64
    ):
    
    c1 = encoder(c1)
    
    z = tf.random.normal(shape=(batch_size, z_dim))
    with tf.GradientTape() as tp:
        c0 = G(z, training=True)
        #c0 = encoder(c0)
        lg = tf_ntk_loss_fn_partial(c0, c1)
        #ld = -tf_ntk_loss_fn_partial(c0, c1)
        ld = -lg
    print(lg)
    gradient_g = tp.gradient(lg, G.trainable_variables)
    gradient_g , _ = tf.clip_by_global_norm(gradient_g, 5.0)
    for i in range(len(gradient_g)):
        gradient_g[i] = tf.where(tf.math.is_nan(gradient_g[i]),  tf.zeros_like(gradient_g[i]), gradient_g[i])
    optimizer_g.apply_gradients(zip(gradient_g, G.trainable_variables))
    
    return lg, ld
# %%
ga_ntk_gan_training_step = [
    G_Train,
]
ga_ntk_step = len(ga_ntk_gan_training_step)
# %%
def ga_ntk_gan_training_loop(
    D, 
    G, 
    ds_train, 
    ds_size,
    target_distribution,
    ds_name,
    t,
    epoch=10000, 
    batch_size=64,
    z_dim=64,
    g_lr=2e-4,
    diag_reg=1e-4
    ):
    
    encoder, decoder = load_ae(ds_name)
    print(encoder.summary())
    s = tf.random.normal([batch_size*3, z_dim])

    ga_ntk_lg = [None] * epoch #record loss of g for each epoch
    ga_ntk_ld = [None] * epoch #record loss of d for each epoch
    ga_ntk_sp = [None] * epoch #record sample images for each epoch

    save_path = "./imgs/ga_ntk-%s-t=%.2e-ds_size=%d-bt_size=%d-class=%s-diag_reg=%.2e-g_lr=%.2e/"%(
        ds_name, t, ds_size, batch_size, target_distribution, diag_reg, g_lr
    )
    dirname = os.path.dirname(save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        raise BaseException("save path existed !")
    
    ntk_pred_fn_partial, ntk_loss_fn_partial = NTK_Generative(
        train_size=batch_size,
        model_type=ds_name,
        t=t,
        diag_reg=diag_reg
    ).get_loss_fn_partial()
    
    tf_ntk_pred_fn_partial = wrap_ga_ntk_in_tf(ntk_pred_fn_partial)
    tf_ntk_loss_fn_partial = wrap_ga_ntk_in_tf(ntk_loss_fn_partial)
    
    optimizer_g = tf.keras.optimizers.Adam(g_lr, beta_1=0.5)

    ctr = 0
    loss_g_t = 0.0
    loss_d_t = 0.0
    avg_constant = float(batch_size) / float(ds_size)

    #tf.debugging.enable_check_numerics()
    
    los_x = []
    los_y = []
    for ep in tqdm(range(epoch)):
        for batch in ds_train:
            loss_g, loss_d = ga_ntk_gan_training_step[ctr](
                c1=batch,
                tf_ntk_loss_fn_partial=tf_ntk_loss_fn_partial,
                G=G,
                optimizer_d=None,
                optimizer_g=optimizer_g,
                loss_fn=None,
                encoder = encoder,
                batch_size=batch_size,
                z_dim=z_dim,
            )
            
            ctr += 1
            loss_g_t += loss_g.numpy()
            loss_d_t += loss_d.numpy()
            if ctr == ga_ntk_step:
                ctr = 0
        ga_ntk_lg[ep] = loss_g_t * avg_constant
        ga_ntk_ld[ep] = loss_d_t * avg_constant
        # save snapshots
        
        if (ep) % 100 == 0:
            print("G loss: %.5f, D Loss: %.5f"%(ga_ntk_lg[ep], ga_ntk_ld[ep]))
            out = decoder(G(s, training=False))
            if 'gaussian' in ds_name:
                Mixed_Gaussian.visualize(out.numpy(), width=64, height=64, plot_type='scatter',
                                        fig_path=os.path.join(save_path, "ga_ntk_%04d.png"%(ep)))
            else:
                row = 8
                col = 8
                imgs = out.numpy()[:row*col]
                if ds_name in ['cifar10', 'celeb_a', 'imagenet', 'celeb_a_large']:
                    imgs = (imgs + 1) / 2
                imgs *= 255
                img = utPuzzle(
                    imgs=imgs.astype(np.uint8),
                    row=row, col=col, path=os.path.join(save_path, "ga_ntk_%04d.png"%(ep))
                )
            los_x.append(ep)
            los_y.append(loss_g.numpy())
        # ga_ntk_sp[ep] = img

    # save image samples
    Generater = keras.Sequential([G,decoder])
    if 'gaussian' in ds_name:
        save_images_samples(G, save_path, z_dim, ds_name, save_format='npy')
    else:
        save_images_samples(Generater, save_path, z_dim, ds_name, save_format='png')
    
    plt.plot(los_x, los_y)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.show()

    return
