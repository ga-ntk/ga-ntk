import os
import imageio
import configparser as Parser
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as onp

from datetime import datetime
from textwrap import wrap
from enum import Enum, auto
from typing import Callable, Dict, Tuple

np = onp

def utPuzzle(imgs, row, col, path=None):
    h, w, c = imgs[0].shape
    out = np.zeros((h * row, w * col, c), np.uint8)
    for n, img in enumerate(imgs):
        j, i = divmod(n, col)
        out[j * h : (j + 1) * h, i * w : (i + 1) * w, :] = img
    if path is not None : imageio.imwrite(path, out)
    return out

def save_images_samples(
    G: tf.keras.Model, 
    path: str,
    z_dim: int,
    ds_name: str,
    save_format: str="png",
    sample_size: int=4096,
    batch_size: int=64,
):
    # create path
    save_folder = os.path.join(path, save_format)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    
    # save samples by batch
    for i in range(0, sample_size, batch_size):
        z_sample = tf.random.normal((batch_size, z_dim))
        sample_images = G(z_sample).numpy()
        
        # pre-processing for image data
        if 'gaussian' not in ds_name:
            # from float32 to uint8
            if ds_name == 'cifar10' or ds_name == 'celeb_a' or ds_name == 'imagenet':
                sample_images = (sample_images + 1) / 2
            sample_images *= 255
            sample_images = sample_images.astype(np.uint8)
        
        # save samples
        for idx, img in enumerate(sample_images):
            path_for_img = os.path.join(save_folder, f'{i+idx}.{save_format}')
            if save_format in ['jpg', 'png']:
                imageio.imwrite(path_for_img, img)
            else:
                np.save(path_for_img, img)
    return

class Args():
    def __init__(self, model_type, celeb_a_path):
        self.model_type = model_type        
        self.celeb_a_path = celeb_a_path
        
    def read_config(self, config_file: str):
        self.parser = Parser.RawConfigParser()
        self.parser.read(config_file)
        
class Dataset(Enum):
    mnist = auto()
    celeb_a = auto()
    cifar10 = auto()

def init_env(gpu_id: str):
    from jax.config import config
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
    os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
    # os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    config.update('jax_enable_x64', True)

# MODIFIED: 
def get_result_path() -> str:
    cwd = os.getcwd()
    print(f'Current Directory: {cwd}')
    result_path = os.path.join(cwd, "results", f"{datetime.today().strftime('%Y-%m-%d')}", f"{datetime.today().strftime('%H-%M-%S')}")
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    return result_path

def check_make_dir(path: str, is_delete_if_exist: bool=False):
    """
    Check whether the directory exist or not and make the directory if it doesn't exist
    """
    if os.path.exists(path):
        if is_delete_if_exist:
            os.rmdir(path)
            os.makedirs(path)
    else:
        os.makedirs(path)

def get_exp_path(exp_name: str, info_list: str, base_path: str="results") -> str:
    cwd = os.getcwd()
    # infos = '_'.join(info_list).replace(".", "")
    infos = '_'.join(info_list)
    
    print(f'Current Directory: {cwd}')
    result_path = os.path.join(cwd, base_path, exp_name, infos)
    if os.path.exists(result_path):
        os.rmdir(result_path)
    os.makedirs(result_path)
    
    return result_path

# MODIFIED: Add `is_save_fig` to decide whether save the figure or not.
def plt_samples(arr, iteration, dataset_name, image_shape, batch_size, target_distribution, is_save_fig, result_path, infos='', postfix=''):
    """
    :param is_save_fig: bool. Decide whether save the figure or not.
    """
    fig, axs = plt.subplots(2, 5, figsize=(3*(5/2), 3), sharex=True)
    fig.suptitle("\n".join(wrap(f"{dataset_name} Samples, {infos}", 60)), horizontalalignment='center', wrap=True)
    
    if dataset_name == 'mnist':    
        for row, ax in enumerate(axs):
            for idx, a in enumerate(ax):
                img = arr[idx + row*5].reshape(image_shape[:2])
                a.axis('off')
                a.xaxis.set_visible(False)
                a.yaxis.set_visible(False)
                a.imshow(img, cmap='gray', vmin=0, vmax=1)
    elif dataset_name == 'cifar10' or dataset_name == 'celeb_a':
        for row, ax in enumerate(axs):
            for idx, a in enumerate(ax):
                img = arr[idx + row*5].reshape(image_shape)
                a.axis('off')
                a.xaxis.set_visible(False)
                a.yaxis.set_visible(False)
                a.imshow(img, vmin=0, vmax=1)

    plt.tight_layout()
    if is_save_fig:
        fig_name = '%s-%s-batch=%d=iter=%d-%s.png'%(dataset_name, target_distribution, batch_size, iteration, postfix)
        fig_path = os.path.join(result_path, fig_name)
        plt.savefig(fig_path)
        plt.show()
    else:
        plt.close('%s-%s-batch=%d=iter=%d-%s.png'%(dataset_name, target_distribution, batch_size, iteration, postfix))

def show_imgs(img, is_grey=False):
    # plt.figure()
    plt.axis('off')
    if is_grey:
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    else:
        plt.imshow(img, vmin=0, vmax=1, aspect='equal')
    # plt.close()

def norm(x: np.array) -> float:
    """
    :param x: Compute Euclidean norm for `x`. If `x` is a one dimension vector, function would return ||x||. 
              If `x` has more than one dimension, it returns the average of norm along axis 1.
    """
    if x.ndim <= 1:
        return np.linalg.norm(x)
    elif x.ndim == 2:
        x_new = np.reshape(x, (x.shape[0], -1))
        return np.mean(np.linalg.norm(x_new, axis=1))
    else:
        raise ValueError("Cannot handle more than 2 dimension array.")

def cos_similarity_score(x1, x2):
    """
    Cosine similarity score
    """
    return (onp.dot(x1, x2) / (onp.linalg.norm(x1) * onp.linalg.norm(x2)))

def find_best(num, x_noise_flatten, x_train_all_flatten):
    score = onp.zeros((x_train_all_flatten.shape[0]))
    for idx, x2 in enumerate(x_train_all_flatten):
        # cosine score
        score[idx] = cos_similarity_score(x_noise_flatten[num], x2)
    best_list = onp.argsort(score)[::-1]
    return best_list

def avg_max_similarity(x_noise, x_train_all):
    x_noise_flatten = np.asarray(x_noise.reshape(x_noise.shape[0], -1))
    x_train_all_flatten = np.asarray(x_train_all.reshape(x_train_all.shape[0], -1))
    
    max_score = np.zeros((x_noise_flatten.shape[0]))
    for num, x1 in enumerate(x_noise_flatten):
        score = np.zeros((x_train_all_flatten.shape[0]))
        for idx, x2 in enumerate(x_train_all_flatten):
            # cosine score
            score[idx] = cos_similarity_score(x1, x2)
        max_score[num] = np.max(score)
        
    return np.mean(max_score)
    
def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan(x, y)
    return r, theta

def pol2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

class PerturbationMethod(Enum):
    exp = auto()
    anneal = auto()
    static = auto()
class TrendRecorder():
    def __init__(self, fig_path: str) -> None:
        self.recorder = {}
        self.time_recorder = {}
        self.fig_path = fig_path
        
    def record(self, name: str, value: float, time: int=None) -> None:
        if self.recorder.get(name, None) == None:
            self.recorder[name] = [value]
            if time != None:
                self.time_recorder[name] = [time]
            else:
                self.time_recorder[name] = None
        else:
            self.recorder[name].append(value)
            if time != None:
                if self.time_recorder.get(name, None) != None:
                    self.time_recorder[name].append(time)
                else:
                    raise ValueError(f"Missing some value of time, the length of value and time should be equal.")
            else:
                if self.time_recorder.get(name, None) != None:
                    raise ValueError(f"Missing the time for the corresponding value, the length of value and time should be equal.")
                else:
                    self.time_recorder[name] = None
            
    def clear(self) -> None:
        self.recorder.clear()
        self.time_recorder.clear()
        
    def __auto_y_scale(self, name: str) -> Dict[str, float]:
        record = self.recorder.get(name, None)
        if record != None:
            set_ylim = {'bottom': min(record), 'top': max(record)}
            return set_ylim
        else:
            raise ValueError(f"No coresponding record named '{name}' in the Trend_Recorder")
        
    def visualize(self, name:str, is_save_fig:bool, is_show_fig:bool, 
                  set_xlim: object=None, set_ylim: object=None, y_log_scale: bool=False, auto_y_scale: bool=False, 
                  dpi: int=90, format: str='png', postfix: str='') -> None:
        record = self.recorder.get(name, None)
        if  record != None:
            # xs = [i*100 for i in range(len(record))]
            plt.figure(dpi=dpi)
            xs = self.time_recorder.get(name, None)
            if xs != None:
                plt.plot(xs, record)
            else:
                plt.plot(record)
            plt.title(name)

            # Set the range of x-axis and y-axis
            axes = plt.gca()
            if set_xlim != None:
                axes.set_xlim(**set_xlim)
            if set_ylim != None:
                axes.set_ylim(**set_ylim)
            elif auto_y_scale:
                axes.set_ylim(**self.__auto_y_scale(name=name))
                
            if y_log_scale:
                axes.set_yscale('log')
            
            if is_save_fig:
                fig_path = os.path.join(self.fig_path, f'{name}{postfix}.{format}')
                plt.savefig(fig_path, dpi=dpi)
            
            # Show figure or close it
            if is_show_fig:
                plt.show()
            else:
                plt.close()
        else:
            raise ValueError(f"No coresponding record named '{name}' in the Trend_Recorder")
    
if __name__ == '__main__':
    arg = Args() 
    arg.read_config(config_file='run.config')