import jax
import jax.numpy as np
from jax.api import jit, grad
import neural_tangents as nt

from util import avg_max_similarity

class LossFn():
    NTK_LEARNING_RATE = 1e-2
    DIAG_REG = 1e-5

    # Loss types
    ORIGIN_LOSS = 'origin'
    CROSS_LOSS = 'corss'
    MSE_LOSS = 'mse'
    SIMILARITY_LOSS = 'similarity'

    def __init__(self, loss_type: str, target_label: np.ndarray, seed: int, alpha: float=1e-1):
        self.loss_type = loss_type
        self.target_label = target_label
        self.alpha = alpha
        # self.augmentation = augmentation
        self.is_augment = False
        self.seed = seed
        self.loss_fn, self.get_pred_fn  = self.__get_loss_pred_fn(loss_type=loss_type)
        
    # NOTE this only works for hard labels (y_hat)
    def __cross_entropy(self, fx, y_hat): 
        return -np.mean(np.sum(jax.nn.log_softmax(fx) * y_hat, axis=1))
    
    # MODIFIED: MSE loss
    def __mse(self, y, y_hat): 
        return np.mean(np.sum(np.square(y - y_hat), axis=1))
    
    # MODIFIED: Return cross entropy prediction function
    def __get_pred_fn_cross(self, x_noise, x_train, labels, kernel_fn):
        return nt.predict.gradient_descent(
                self.__cross_entropy,
                kernel_fn(np.concatenate([x_train, x_noise]), None, 'ntk', diag_reg=self.DIAG_REG),
                labels,
                learning_rate=self.NTK_LEARNING_RATE,
            )
        
    # MODIFIED: Return MSE prediction function
    def __get_pred_fn_mse(self, x_noise, x_train, labels, kernel_fn):
        return nt.predict.gradient_descent_mse(
                kernel_fn(np.concatenate([x_train, x_noise]), None, 'ntk', diag_reg=self.DIAG_REG),
                labels,
                learning_rate=self.NTK_LEARNING_RATE,
                diag_reg=self.DIAG_REG
            )
    
    # def __augment(self, x):
    #     if self.is_augment:
    #         return self.aug.augment(x)
    #     else:
    #         return x

    # MODIFIED: 
    def __loss_fn_origin(self, x_noise, x_train, labels, kernel_fn, t):
        # x_noise_aug = self.__augment(x_noise)
        # x_train_aug = self.__augment(x_train)
        # pred_fn = self.__get_pred_fn_mse(x_noise_aug, x_train_aug, labels, kernel_fn)
        pred_fn = self.__get_pred_fn_mse(x_noise, x_train, labels, kernel_fn)
        output = pred_fn(t)
        return self.__cross_entropy(output, self.target_label)
    
    # MODIFIED: pure cross entropy loss
    def __loss_fn_cross_entropy(self, x_noise, x_train, labels, kernel_fn, t):
        # x_noise_aug = self.__augment(x_noise)
        # x_train_aug = self.__augment(x_train)
        # pred_fn = self.__get_pred_fn_cross(x_noise_aug, x_train_aug, labels, kernel_fn)
        pred_fn = self.__get_pred_fn_cross(x_noise, x_train, labels, kernel_fn)
        output = pred_fn(t)
        return self.__cross_entropy(output, self.target_label)
    
    # MODIFIED: pure MSE loss
    def __loss_fn_mse(self, x_noise, x_train, labels, kernel_fn, t):
        # x_noise_aug = self.__augment(x_noise)
        # x_train_aug = self.__augment(x_train)
        # pred_fn = self.__get_pred_fn_mse(x_noise_aug, x_train_aug, labels, kernel_fn)
        pred_fn = self.__get_pred_fn_mse(x_noise, x_train, labels, kernel_fn)
        output = pred_fn(t)
        return self.__mse(output, self.target_label)
    
    # MODIFIED: 
    def __loss_fn_similarity_regular(self, x_noise, x_train, labels, kernel_fn, t):
        # x_noise_aug = self.__augment(x_noise)
        # x_train_aug = self.__augment(x_train)
        # pred_fn = self.__get_pred_fn_mse(x_noise_aug, x_train_aug, labels, kernel_fn)
        pred_fn = self.__get_pred_fn_mse(x_noise, x_train, labels, kernel_fn)
        output = pred_fn(t)
        return self.__cross_entropy(output, self.target_label) + self.alpha * avg_max_similarity(x_noise=x_noise, x_train_all=x_train)
    
    def __get_loss_pred_fn(self, loss_type):
        # MODIFIED: switch loss types
        if loss_type == self.ORIGIN_LOSS:
            loss_fn = self.__loss_fn_origin
            get_pred_fn = self.__get_pred_fn_mse
        elif loss_type == self.CROSS_LOSS:
            loss_fn = self.__loss_fn_cross_entropy
            get_pred_fn = self.__get_pred_fn_cross
        elif loss_type == self.MSE_LOSS:
            loss_fn = self.__loss_fn_mse
            get_pred_fn = self.__get_pred_fn_mse
        elif loss_type == self.SIMILARITY_LOSS:
            loss_fn = self.__loss_fn_similarity_regular
            get_pred_fn = self.__get_pred_fn_mse
        else:
            raise ValueError(f'No such loss function: {loss_type}')
        
        return loss_fn, get_pred_fn
    
    def get_fns(self):
        return self.loss_fn, self.get_pred_fn

    def get_fns_jit(self):
        grad_fn = jit(grad(self.loss_fn, 0), static_argnums=3)
        return grad_fn, self.loss_fn, self.get_pred_fn