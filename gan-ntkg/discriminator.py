# %%
import tensorflow as tf
import numpy as np

# %%
layers = tf.keras.layers

# %%
def get_discriminator(dataset_name, related_work):
    Discriminator = None
    if 'gaussian' in dataset_name:
        Discriminator = tf.keras.Sequential([])
        Discriminator.add(
            layers.Dense(
                units=64, input_shape=(2,),
                kernel_constraint=SpectralNorm1D(64) if related_work=='sngan' else None
        ))
        if related_work != 'wgan-gp':
            Discriminator.add(layers.BatchNormalization())
        Discriminator.add(layers.ReLU())
        Discriminator.add(
            layers.Dense(
                units=64,
                kernel_constraint=SpectralNorm1D(64) if related_work=='sngan' else None
        ))
        if related_work != 'wgan-gp':
            Discriminator.add(layers.BatchNormalization())
        Discriminator.add(layers.ReLU())
        Discriminator.add(
            layers.Dense(
                units=1,
                kernel_constraint=SpectralNorm1D(1) if related_work=='sngan' else None
        ))
        
    elif dataset_name == 'mnist':
        Discriminator = tf.keras.Sequential([
        layers.Conv2D(
            filters=64,
            kernel_size=4,
            strides=(2, 2),
            padding="SAME",
            input_shape=(28, 28, 1),
            kernel_constraint=SpectralNorm2D(64) if related_work=='sngan' else None
        ),
        layers.LeakyReLU(),
        layers.Conv2D(
            filters=64,
            kernel_size=4,
            strides=(2, 2),
            padding="SAME",
            kernel_constraint=SpectralNorm2D(64) if related_work=='sngan' else None
        )])
        if related_work != 'wgan-gp':
            Discriminator.add(layers.BatchNormalization())
        Discriminator.add(layers.LeakyReLU())
        Discriminator.add(layers.Flatten())
        Discriminator.add(
            layers.Dense(
                1024,
                kernel_constraint=SpectralNorm1D(1024) if related_work=='sngan' else None
        ))
        if related_work != 'wgan-gp':
            Discriminator.add(layers.BatchNormalization())
        Discriminator.add(layers.LeakyReLU())
        Discriminator.add(
            layers.Dense(
                1,
                kernel_constraint=SpectralNorm1D(1) if related_work=='sngan' else None
            )
        )

    elif dataset_name == 'cifar10':
        Discriminator = tf.keras.Sequential([
        layers.Conv2D(
            filters=64,
            kernel_size=4,
            strides=(2, 2),
            padding="SAME",
            input_shape=(32, 32, 3),
            kernel_constraint=SpectralNorm2D(64) if related_work=='sngan' else None
        ),
        layers.LeakyReLU(),
        layers.Conv2D(
            filters=128,
            kernel_size=4,
            strides=(2, 2),
            padding="SAME",
            kernel_constraint=SpectralNorm2D(128) if related_work=='sngan' else None
        )])
        if related_work != 'wgan-gp':
            Discriminator.add(layers.BatchNormalization())
        Discriminator.add(layers.LeakyReLU())
        Discriminator.add(layers.Conv2D(
            filters=256,
            kernel_size=4,
            strides=(2, 2),
            padding="SAME",
            kernel_constraint=SpectralNorm2D(256) if related_work=='sngan' else None
        ))
        if related_work != 'wgan-gp':
            Discriminator.add(layers.BatchNormalization())
        Discriminator.add(layers.LeakyReLU())
        Discriminator.add(layers.Flatten())
        Discriminator.add(
            layers.Dense(
                1,
                kernel_constraint=SpectralNorm1D(1) if related_work=='sngan' else None
            )
        )
        
    elif dataset_name == 'celeb_a':
        Discriminator = tf.keras.Sequential([
        layers.Conv2D(
            filters=64,
            kernel_size=4,
            strides=(2, 2),
            padding="SAME",
            input_shape=(64, 64, 3),
            kernel_constraint=SpectralNorm2D(64) if related_work=='sngan' else None
        ),
        layers.LeakyReLU(),
        layers.Conv2D(
            filters=128,
            kernel_size=4,
            strides=(2, 2),
            padding="SAME",
            kernel_constraint=SpectralNorm2D(128) if related_work=='sngan' else None
        )])
        if related_work != 'wgan-gp':
            Discriminator.add(layers.BatchNormalization())
        Discriminator.add(layers.LeakyReLU())
        Discriminator.add(layers.Conv2D(
            filters=256,
            kernel_size=4,
            strides=(2, 2),
            padding="SAME",
            kernel_constraint=SpectralNorm2D(256) if related_work=='sngan' else None
        ))
        if related_work != 'wgan-gp':
            Discriminator.add(layers.BatchNormalization())
        Discriminator.add(layers.LeakyReLU())
        Discriminator.add(layers.Conv2D(
            filters=256,
            kernel_size=4,
            strides=(2, 2),
            padding="SAME",
            kernel_constraint=SpectralNorm2D(256) if related_work=='sngan' else None
        ))
        if related_work != 'wgan-gp':
            Discriminator.add(layers.BatchNormalization())
        Discriminator.add(layers.LeakyReLU())
        Discriminator.add(layers.Flatten())
        Discriminator.add(
            layers.Dense(
                1,
                kernel_constraint=SpectralNorm1D(1) if related_work=='sngan' else None
            )
        )
    elif dataset_name == 'imagenet':
        Discriminator = tf.keras.Sequential([
        layers.Conv2D(
            filters=64,
            kernel_size=4,
            strides=(2, 2),
            padding="SAME",
            input_shape=(128, 128, 3),
            kernel_constraint=SpectralNorm2D(64) if related_work=='sngan' else None
        ),
        layers.LeakyReLU(),
        layers.Conv2D(
            filters=128,
            kernel_size=4,
            strides=(2, 2),
            padding="SAME",
            kernel_constraint=SpectralNorm2D(128) if related_work=='sngan' else None
        )])
        if related_work != 'wgan-gp':
            Discriminator.add(layers.BatchNormalization())
        Discriminator.add(layers.LeakyReLU())
        Discriminator.add(layers.Conv2D(
            filters=256,
            kernel_size=4,
            strides=(2, 2),
            padding="SAME",
            kernel_constraint=SpectralNorm2D(256) if related_work=='sngan' else None
        ))
        if related_work != 'wgan-gp':
            Discriminator.add(layers.BatchNormalization())
        Discriminator.add(layers.LeakyReLU())
        Discriminator.add(layers.Conv2D(
            filters=512,
            kernel_size=4,
            strides=(2, 2),
            padding="SAME",
            kernel_constraint=SpectralNorm2D(512) if related_work=='sngan' else None
        ))
        if related_work != 'wgan-gp':
            Discriminator.add(layers.BatchNormalization())
        Discriminator.add(layers.LeakyReLU())
        Discriminator.add(layers.Conv2D(
            filters=512,
            kernel_size=4,
            strides=(2, 2),
            padding="SAME",
            kernel_constraint=SpectralNorm2D(512) if related_work=='sngan' else None
        ))
        if related_work != 'wgan-gp':
            Discriminator.add(layers.BatchNormalization())
        Discriminator.add(layers.LeakyReLU())
        Discriminator.add(layers.Conv2D(
            filters=512,
            kernel_size=4,
            strides=(2, 2),
            padding="SAME",
            kernel_constraint=SpectralNorm2D(512) if related_work=='sngan' else None
        ))
        if related_work != 'wgan-gp':
            Discriminator.add(layers.BatchNormalization())
        Discriminator.add(layers.LeakyReLU())
        Discriminator.add(layers.Flatten())
        Discriminator.add(
            layers.Dense(
                1,
                kernel_constraint=SpectralNorm1D(1) if related_work=='sngan' else None
            )
        )
        
    return Discriminator
# %%
class SpectralNorm1D(tf.keras.constraints.Constraint):
    def __init__(self, output_neurons, power_iterations=1):

        assert power_iterations>=1, "The number of power iterations should be positive integer"
        self.Ip = power_iterations
        u_init = tf.random_uniform_initializer()
        self.u = tf.Variable(initial_value = u_init(shape=(1, output_neurons), dtype='float32'),
                             trainable=False)

    def __call__(self, w):

        W_mat = tf.transpose(w, (1, 0))  # (i, o) => (o, i)

        _u = self.u
        _v = None

        for _ in range(self.Ip):
            _v = l2_norm(tf.matmul(_u, W_mat))
            _u = l2_norm(tf.matmul(_v, W_mat, transpose_b=True))

        sigma = tf.reduce_sum(tf.matmul(_u, W_mat) * _v)
        sigma = tf.cond(sigma==0, lambda: 1e-8, lambda: sigma)

        self.u.assign(tf.keras.backend.in_train_phase(_u, self.u))
        return w / sigma

class SpectralNorm2D(tf.keras.constraints.Constraint):
    def __init__(self, output_neurons, power_iterations=1):

        assert power_iterations>=1, "The number of power iterations should be positive integer"
        self.Ip = power_iterations
        u_init = tf.random_uniform_initializer()
        self.u = tf.Variable(initial_value = u_init(shape=(1, output_neurons), dtype='float32'),
                             trainable=False)

    def __call__(self, w):

        W_mat = tf.transpose(w, (3, 2, 0, 1))  # (h, w, i, o) => (o, i, h, w)
        W_mat = tf.reshape(W_mat, [tf.shape(W_mat)[0], -1])  # (o, i * h * w)

        _u = self.u
        _v = None

        for _ in range(self.Ip):
            _v = l2_norm(tf.matmul(_u, W_mat))
            _u = l2_norm(tf.matmul(_v, W_mat, transpose_b=True))

        sigma = tf.reduce_sum(tf.matmul(_u, W_mat) * _v)
        sigma = tf.cond(sigma==0, lambda: 1e-8, lambda: sigma)

        self.u.assign(tf.keras.backend.in_train_phase(_u, self.u))
        return w / sigma

def l2_norm(x):
    return x / tf.sqrt(tf.reduce_sum(tf.square(x)) + 1e-8)