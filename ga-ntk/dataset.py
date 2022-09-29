import os
from typing import List, Tuple, Union
import json
from joblib import Parallel, delayed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as onp
import glob
import jax.numpy as np
import cv2
import tensorflow_datasets as tfds
from jax import random

from util import pol2cart, check_make_dir, cos_similarity_score

def shuffle_data(images, labels, seed=None):
    perm = onp.random.RandomState(seed).permutation(images.shape[0])
    images = images[perm]
    labels = labels[perm]
    return images, labels

# MODIFIED: Sample data for specific number of samples
def sample_data(images, sample_num, labels=None, seed=None):
    sample_range = images.shape[0]
    if sample_num != -1:
        indices = onp.random.RandomState(seed).choice(sample_range, sample_num, replace=False)
        images = images[indices]
        
        if labels is not None:
            labels = labels[indices]
            return images, labels
        return images
    else:
        if labels != None:
            return images, labels
        return images

def _partial_flatten_and_normalize(x):
    """Flatten all but the first dimension of an `np.ndarray`."""
    x = onp.reshape(x, (x.shape[0], -1))
    return (x - onp.mean(x)) / onp.std(x)

def _flatten(x):
    """Flatten all but the first dimension of an `np.ndarray`."""
    return onp.reshape(x, (x.shape[0], -1))/255

def _normalize(x):
    """Flatten all but the first dimension of an `np.ndarray`."""
    return x / 255


def _one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return onp.array(x[:, None] == onp.arange(k), dtype)

def rgb2gray(rgb):
    return onp.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def get_labels(train_size: int, noise_size: int):
    labels = onp.concatenate([
        onp.ones((train_size,), onp.float64), 
        onp.zeros((noise_size,), onp.float64)
    ])
    labels = _one_hot(labels, 2, dtype=np.float64)

    target_label = onp.ones((train_size + noise_size,), onp.float64)
    target_label = _one_hot(target_label, 2, dtype=np.float64)

    return labels, target_label

class BaseDataset():
    def __init__(self, data_path: str, shape: Tuple[int, ...]):
        self.data_path = data_path
        self.shape = shape
        
    def _crop_to_center(self, image: np.ndarray):
        shape = image.shape
        up_len = self.shape[0] // 2
        low_len = self.shape[0] - up_len
        left_len = self.shape[1] // 2
        right_len = self.shape[1] - left_len
        
        left = shape[1] // 2 - left_len
        top = shape[0] // 2 - up_len
        right = shape[1] // 2 + right_len
        bottom = shape[0] // 2 + low_len
        
        if len(shape) == 2:
            center_cropped_img = image[top:bottom, left:right]
        else:
            center_cropped_img = image[top:bottom, left:right, ...]
        
        return center_cropped_img
    
    def _read_img(self, path: str):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        # img = self._crop_to_center(img)
        img = cv2.resize(img, (self.shape[0], self.shape[1]), interpolation=cv2.INTER_AREA)
        return img

    def _read_images(self, img_path_list: List[str], n_jobs: int=-1):
        # print(f"Images List{img_path_list}")
        images = Parallel(n_jobs=n_jobs)(
            delayed(self._read_img)(f) for f in img_path_list
        )
        # print(f"images[0] shape: {images[0].shape}")
        for idx, img in enumerate(images):
            if img.shape != self.shape:
                images.pop(idx)
                print(f"Removed: {img_path_list[idx]} images[{idx}] due to wrong shape: {img.shape}")
        if len(images) > 0:
            return np.flip(np.stack(images, axis=0), axis=3)
        else:
            raise ValueError(f"Empty dataset")
            # return np.array(images)
    
    def _get_img_path(self, dir_paths: Tuple[List[str], str]):
        img_list = []
        # print(f"dir_paths: {dir_paths}")
        if isinstance(dir_paths, list):
            for d in dir_paths:
                for (dirpath, dirnames, filenames) in os.walk(d):
                    for f in filenames:
                        # print(f"filenames: {f}")
                        img_list.append(os.path.join(dir_paths, f))
            return img_list
        elif isinstance(dir_paths, str):
            for (dirpath, dirnames, filenames) in os.walk(dir_paths):
                for f in filenames:
                    # print(f"filenames: {f}")
                    img_list.append(os.path.join(dir_paths, f))
            return img_list
        else:
            raise TypeError(f"Argument classes should be a string, List of string or None, but not {type(dir_paths)}")

class FewShotDataset(BaseDataset):
    OBAMA_100 = "obama"
    GRUMPY_CAT_100 = "grumpy_cat"
    BRIDGE_100 = "bridge"
    MEDICI_100 = "medici"
    PANDA_100 = "panda"
    WUZHEN_100 = "wuzhen"
    TEMPLE_100 = "temple"
    ANIMALFACE_DOG = "animalface_dog"
    ANIMALFACE_CAT = 'animalface_cat'
    
    OBAMA_100_DIR = "100-shot-obama"
    GRUMPY_CAT_100_DIR = "100-shot-grumpy_cat"
    BRIDGE_100_DIR = "100-shot-bridge_of_sighs"
    MEDICI_100_DIR = "100-shot-medici_fountain"
    PANDA_100_DIR = "100-shot-panda"
    WUZHEN_100_DIR = "100-shot-wuzhen"
    TEMPLE_100_DIR = "100-shot-temple_of_heaven"
    ANIMALFACE_DOG_DIR = "AnimalFace-dog"
    ANIMALFACE_CAT_DIR = "AnimalFace-cat"
    
    def __init__(self, data_path: str, shape: Tuple[int, ...]=(128, 128, 3)):
        super().__init__(data_path=data_path, shape=shape)

    def get_data(self, catergory: str):
        if catergory == self.OBAMA_100:
            data_path = os.path.join(self.data_path, self.OBAMA_100_DIR)
        elif catergory == self.GRUMPY_CAT_100:
            data_path = os.path.join(self.data_path, self.GRUMPY_CAT_100_DIR)
        elif catergory == self.BRIDGE_100:
            data_path = os.path.join(self.data_path, self.BRIDGE_100_DIR)
        elif catergory == self.MEDICI_100:
            data_path = os.path.join(self.data_path, self.MEDICI_100_DIR)
        elif catergory == self.PANDA_100:
            data_path = os.path.join(self.data_path, self.PANDA_100_DIR)
        elif catergory == self.WUZHEN_100:
            data_path = os.path.join(self.data_path, self.WUZHEN_100_DIR)
        elif catergory == self.TEMPLE_100:
            data_path = os.path.join(self.data_path, self.TEMPLE_100_DIR)
        elif catergory == self.ANIMALFACE_DOG:
            data_path = os.path.join(self.data_path, self.ANIMALFACE_DOG_DIR)
        elif catergory == self.ANIMALFACE_CAT:
            data_path = os.path.join(self.data_path, self.ANIMALFACE_CAT_DIR)
        else:
            raise ValueError(f"No such catergory named '{catergory}'")
        images = self._read_images(self._get_img_path(data_path))
        return images

class ImageNet(BaseDataset):
    TRAINING_DATA = 'train'
    TESTING_DATA = 'test'
    VALIDATING_DATA = 'val'
    MID_PATH = 'ILSVRC/Data/CLS-LOC'
    CLASS_INDEX = 'imagenet_class_index.json'

    def __init__(self, data_path: str, shape: Tuple[int, ...]=(128, 128, 3)):
        super().__init__(data_path=data_path, shape=shape)

    def __get_class_dir(self, classes: Tuple[List[str], str]=None):
        with open(os.path.join(self.data_path, self.CLASS_INDEX)) as json_file:
            data = json.load(json_file)

        class_dict = {}
        for k in data.keys():
            folder, class_name = data[k][0], data[k][1]
            class_dict[class_name] = [k, folder]

        if isinstance(classes, list):
            return [class_dict.get(c, [None, None])[1] for c in classes]
        elif isinstance(classes, str):
            return [class_dict.get(classes, None)]
        elif classes == None:
            return [class_dict.get(k, [None, None])[1] for k in class_dict.keys()]
        else:
            raise TypeError(f"Argument classes should be a string, List of string or None, but not {type(classes)}")

    def __get_type_path(self, types: Tuple[List[str], str]=None):
        if isinstance(types, list):
            return [os.path.join(self.data_path, self.MID_PATH, t) for t in types]
        elif isinstance(types, str):
            return [os.path.join(self.data_path, self.MID_PATH, types)]
        elif types is None:
            return [os.path.join(self.data_path, self.MID_PATH, self.TRAINING_DATA)]
        else:
            raise TypeError(f"Argument classes should be a string, List of string or None, but not {type(types)}")

    def _get_img_path(self, dir_paths: Tuple[List[str], str]):
        img_list = []
        # print(f"dir_paths: {dir_paths}")
        if isinstance(dir_paths, list):
            for d in dir_paths:
                for (dirpath, dirnames, filenames) in os.walk(d):
                    for f in filenames:
                        # print(f"filenames: {f}")
                        img_list.append(os.path.join(dir_paths, f))
            return img_list
        elif isinstance(dir_paths, str):
            for (dirpath, dirnames, filenames) in os.walk(dir_paths):
                for f in filenames:
                    # print(f"filenames: {f}")
                    img_list.append(os.path.join(dir_paths, f))
            return img_list
        else:
            raise TypeError(f"Argument classes should be a string, List of string or None, but not {type(dir_paths)}")
    
    def __get_imgs(self, data_paths: List[str], class_folders: List[str]):
        img_path_list = []
        for d in data_paths:
            for c in class_folders:
                # print(f"Join d, c: {os.path.join(d, c)}")
                if (d is not None) and (c is not None):
                    img_path_list += self._get_img_path(os.path.join(d, c))
        images = self._read_images(img_path_list)
        return images

    def get_data(self, types: Tuple[List[str], str]=None, classes: Tuple[List[str], str]=None):
        class_folders = self.__get_class_dir(classes=classes)
        # print(f"class_folders: {class_folders}")
        data_paths = self.__get_type_path(types=types)
        # print(f"data_paths: {data_paths}")
        images = self.__get_imgs(data_paths=data_paths, class_folders=class_folders)
        return images

class Similiarity():
    SSIM = 'ssim'
    COSINE = 'cosine'
    
    def __init__(self):
        pass
    
    def __SSIM(self, x_noise, x_train):
        score = tf.image.ssim(x_noise , x_train, max_val=1.0, filter_size=4, filter_sigma=1.5, k1=0.01, k2=0.03)
        return score.numpy()
        
    def __cos_similarity(self, x_noise, x_train):
        score = []
        for ds in x_train:
            score.append(cos_similarity_score(np.ravel(x_noise), np.ravel(ds)))
        return onp.array(score)
    
    def compute(self, x_noise, x_train, image_shape, metric: str='ssim'):
        """get average max ssim similarity"""
        max_score = []
        max_score_idx = []
        def ssim_max_sim(x_noise, x_train):
            """get max ssim similarity for one image"""
            x_noise = onp.asarray(x_noise).reshape(-1, *image_shape)
            x_train = onp.asarray(x_train).reshape(-1, *image_shape)
            
            # x_noise = tf.image.decode_png(x_noise, channels=1, dtype=tf.float32)
            # x_train = tf.image.decode_png(x_train, channels=1, dtype=tf.float32)
            
            x_noise = tf.image.convert_image_dtype(x_noise, tf.float32).numpy()
            x_train = tf.image.convert_image_dtype(x_train, tf.float32).numpy()
            
            if metric == self.SSIM:
                score = self.__SSIM(x_noise=x_noise, x_train=x_train)
            elif metric == self.COSINE:
                score = self.__cos_similarity(x_noise=x_noise, x_train=x_train)
                
            # max_score_idx = tf.math.argmax(score)
            max_score_idx = onp.argsort(score)[::-1]
            return score[np.ix_(max_score_idx)], max_score_idx
        
        for x1 in tqdm(x_noise):
            sc, sc_idx = ssim_max_sim(x1, x_train)
            max_score.append(sc)
            max_score_idx.append(sc_idx)
        
        return max_score, max_score_idx
    
class Dataset():
    MNIST = 'mnist'
    CELEB_A = 'celeb_a'
    CELEB_A_LARGE = 'celeb_a_large'
    CIFAR10 = 'cifar10'
    CIFAR100 = 'cifar100'
    IMAGENET = 'imagenet'
    FEWSHOT = 'fewshot'
    GAUSSIAN_8 = 'gaussian_8'
    GAUSSIAN_25 = 'gaussian_25'

    MNIST_SHAPE = (28, 28, 1)
    CIFAR10_SHAPE = (32, 32, 3)
    IMAGENET_SHAPE = (128, 128, 3)
    FEWSHOT_SHAPE = (128, 128, 3)
    CELEB_A_SHAPE = (64, 64, 3)
    CELEB_A_LARGE_SHAPE = (256, 256, 3)
    GAUSSIAN_8_SHAPE = (2, )
    GAUSSIAN_25_SHAPE = (2, )

    CELEB_A_ANNO_DIR = 'Anno'
    CELEB_A_IMG64_DIR = 'img_align_celeba_64'
    CELEB_A_ATTRS_FILE = 'list_attr_celeba.txt'
    
    CELEB_A_LARGE_ANNO_DIR = ''
    CELEB_A_LARGE_IMG256_DIR = 'CelebA-HQ-img-256'
    CELEB_A_LARGE_ATTRS_FILE = 'CelebAMask-HQ-attribute-anno.txt'

    def __init__(self, dataset_name: str, seed: int, celeb_a_path: str=None, celeb_a_large_path: str=None,imagenet_path: str=None, fewshot_path: str=None, flatten: bool=False) -> None:
        self.dataset_name = dataset_name
        self.seed = seed
        self.celeb_a_path = celeb_a_path
        self.celeb_a_large_path = celeb_a_large_path
        self.imagenet_path = imagenet_path
        self.fewshot_path = fewshot_path
        self.flatten = flatten
        self.ng = NoiseGenerator(random_seed=seed)

        self.image_shape, self.vec_size = Dataset.get_dataset_shape(dataset_name=dataset_name)
    
    def __get_dataset(self, n_train: int=None, n_test: int=None, permute_train: bool=False, 
                      normalize: bool=False):
        """Download, parse and process a dataset to unit scale and one-hot labels."""

        ds_builder = tfds.builder(self.dataset_name)
        ds_train, ds_test = tfds.as_numpy(
            tfds.load(
                self.dataset_name + ':3.*.*',
                split=['train' + ('[:%d]' % n_train if n_train is not None else ''),
                    'test' + ('[:%d]' % n_test if n_test is not None else '')],
                batch_size=-1,
                as_dataset_kwargs={'shuffle_files': False}))

        train_images, train_labels, test_images, test_labels = (ds_train['image'],
                                                                ds_train['label'],
                                                                ds_test['image'],
                                                                ds_test['label'])
        num_classes = ds_builder.info.features['label'].num_classes
        
        if self.flatten and normalize:
            train_images = _partial_flatten_and_normalize(train_images)
            test_images = _partial_flatten_and_normalize(test_images)
        elif self.flatten:
            train_images = _flatten(train_images)
            test_images = _flatten(test_images)
        else:
            train_images = _normalize(train_images)
            test_images = _normalize(test_images)
            
        train_labels = _one_hot(train_labels, num_classes)
        test_labels = _one_hot(test_labels, num_classes)

        if permute_train:
            perm = onp.random.RandomState(0).permutation(train_images.shape[0])
            train_images = train_images[perm]
            train_labels = train_labels[perm]

        return train_images, train_labels, test_images, test_labels

    @staticmethod
    def get_dataset_shape(dataset_name: str):
        if dataset_name == Dataset.MNIST:
            image_shape = Dataset.MNIST_SHAPE
            vec_size = image_shape[0] * image_shape[1] * image_shape[2]
        elif dataset_name == Dataset.CIFAR10:
            image_shape = Dataset.CIFAR10_SHAPE
            vec_size = image_shape[0] * image_shape[1] * image_shape[2]
        elif dataset_name == Dataset.IMAGENET:
            image_shape = Dataset.IMAGENET_SHAPE
            vec_size = image_shape[0] * image_shape[1] * image_shape[2]
        elif dataset_name == Dataset.FEWSHOT:
            image_shape = Dataset.FEWSHOT_SHAPE
            vec_size = image_shape[0] * image_shape[1] * image_shape[2]
        elif dataset_name == Dataset.CELEB_A:
            image_shape = Dataset.CELEB_A_SHAPE
            vec_size = image_shape[0] * image_shape[1] * image_shape[2]
        elif dataset_name == Dataset.CELEB_A_LARGE:
            image_shape = Dataset.CELEB_A_LARGE_SHAPE
            vec_size = image_shape[0] * image_shape[1] * image_shape[2]
        elif dataset_name == Dataset.GAUSSIAN_8:
            image_shape = Dataset.GAUSSIAN_8_SHAPE
            vec_size = image_shape[0]
        elif dataset_name == Dataset.GAUSSIAN_25:
            image_shape = Dataset.GAUSSIAN_25_SHAPE
            vec_size = image_shape[0]
        else:
            raise ValueError(f"No such dataset named {dataset_name}")
        return image_shape, vec_size

    def get_data_shape(self):
        return self.image_shape, self.vec_size

    def reset_seed(self, seed: int):
        self.seed = seed
        self.ng = NoiseGenerator(random_seed=seed)
        
    def __process_sample_size_param(self, num: Tuple[int, str]):
        if num is None:
            # Select All images
            return None
        elif isinstance(num, str):
            if num.lower() == 'none':
                # Select All images
                return None
            else:
                # Select specific number of images
                return int(num)
        elif isinstance(num, int):
            # Select specific number of images
            if num >= 0:
                return num
            else:
                return None
        else:
            return None
        
    def __sample_dataset(self, X: np.ndarray, y: np.ndarray=None):
        if self.dataset_size is not None:
            if self.train_size is None:
                self.train_size = self.dataset_size
                
            if y is not None:
                X_all, y_all = sample_data(images=X, labels=y, sample_num=self.dataset_size, seed=self.seed)
                x_train = X_all[:self.train_size]
                y_train = y_all[:self.train_size]
                return X_all, x_train, y_all, y_train
            else:
                X_all = sample_data(images=X, sample_num=self.dataset_size, seed=self.seed)
                x_train = X_all[:self.train_size]
                return X_all, x_train
        else:
            self.dataset_size = X.shape[0]
            if self.train_size is None:
                self.train_size = self.dataset_size
                
            if y is not None:
                x_train = X[:self.train_size]
                y_train = y[:self.train_size]
                return X, x_train, y, y_train
            else:
                x_train = X[:self.train_size]
                return X, x_train

    def set_sample_size(self, noise_size: int=None, train_size: int=None, dataset_size: int=None) -> 'Dataset':
        """
        If dataset_size is None(default), switch to GD. Otherwise, use SGD with batch size = dataset_size
        """
        self.noise_size = self.__process_sample_size_param(num=noise_size)
        self.train_size = self.__process_sample_size_param(num=train_size)
        self.dataset_size = self.__process_sample_size_param(num=dataset_size)

        return self

    def gen_noise(self):
        if self.noise_size is not None:
            x_noise = self.ng.gen_noise(noise_size=self.noise_size, image_shape=self.image_shape, flatten=self.flatten)
            return x_noise
        else:
            raise ValueError(f"Please call method set_sample_size and set noise_size first")

    def gen_labels(self):
        if (self.train_size is not None) and (self.noise_size is not None):
            return get_labels(self.train_size, self.noise_size)
        else:
            raise ValueError(f"Please call method set_sample_size and set train_size, noise_size first")

    def _read_img(self, path: str):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        return img

    def _read_images(self, img_path_list: List[str], n_jobs: int=-1):
        images = Parallel(n_jobs=n_jobs)(
            delayed(self._read_img)(f) for f in img_path_list
        )
        return images
    
    def gen_data_attrs(self, target_class: int=None, attrs: List[str]=None) -> Tuple[np.array, np.array]:
        # read data
        if self.dataset_name == self.MNIST or self.dataset_name == self.CIFAR10:
            x_train_all, y_train_all, x_test_all, y_test_all = tuple(
                onp.array(x) for x in self.__get_dataset(None, None)
            )

            _target_class = None
            if isinstance(target_class, str):
                if target_class.lower() == 'none':
                    _target_class = None
                else:
                    _target_class = int(target_class)
            elif isinstance(target_class, float) or isinstance(target_class, int):
                _target_class = int(target_class)
            else:
                if target_class is None:
                    _target_class = None

            if _target_class is None:
                # shuffle
                x_train_all, y_train_all = shuffle_data(x_train_all, y_train_all, self.seed)
            else:
                # get target class images
                x_train_all = x_train_all[onp.argmax(y_train_all, axis=1)==_target_class]
                y_train_all = y_train_all[onp.argmax(y_train_all, axis=1)==_target_class]
            
            x_train_all, x_train, y_train_all, y_train = self.__sample_dataset(X=x_train_all, y=y_train_all)
                
        elif self.dataset_name == self.CELEB_A:
            if self.celeb_a_path is None:
                raise BaseException("Please specify the path of CELEB_A dataset")

            # parse attribute file
            file_list = []
            attr_list_file_path = os.path.join(self.celeb_a_path, self.CELEB_A_ANNO_DIR, self.CELEB_A_ATTRS_FILE)
            f = open(attr_list_file_path, 'r')
            # print(f"Anno - {self.CELEB_A_ATTRS_FILE}: {attr_list_file_path}")
            file_list = f.readlines()
            f.close()
            attr_to_num = {file_list[1].split(' ')[:-1][idx]: idx for idx in range(len(file_list[1].split(' ')[:-1]))}
            file_name_len = 10
            
            # select images with attributes
            x_train_all = []
            img_path_list = []
            for i, line in enumerate(file_list[2:]):
                # attriutes
                cond_list = [True]
                if (attrs != None) and (attrs != []):
                    for attr in attrs:
                        offset_1 = file_name_len+3*(attr_to_num[attr])
                        offset_2 = file_name_len+3*(attr_to_num[attr]+1)
                        
                        cond_n = (int(line[offset_1:offset_2]) == 1)
                        cond_list.append(cond_n)
                
                if np.all(np.array(cond_list)):
                    file_path = os.path.join(self.celeb_a_path, self.CELEB_A_IMG64_DIR, line[:10])
                    img_path_list.append(file_path)
                    
            x_train_all = self._read_images(img_path_list=img_path_list)
            
            # to numpy array
            x_train_all = onp.stack(x_train_all, axis=0)
            # print(f"x_train_all after stack: {x_train_all.shape}")
            x_train_all = x_train_all.astype(onp.float64)
            x_train_all /= 255
            # MODIFIED: Sample data
            if self.flatten:
                x_train_all = onp.reshape(x_train_all, (x_train_all.shape[0], -1))
                
            x_train_all, x_train = self.__sample_dataset(X=x_train_all)
        elif self.dataset_name == self.CELEB_A_LARGE:
            if self.celeb_a_large_path is None:
                raise BaseException("Please specify the path of CELEB_A_LARGE dataset")

            # parse attribute file
            file_list = []
            attr_list_file_path = os.path.join(self.celeb_a_large_path, self.CELEB_A_LARGE_ANNO_DIR, self.CELEB_A_LARGE_ATTRS_FILE)
            f = open(attr_list_file_path, 'r')
            # print(f"Anno - {self.CELEB_A_ATTRS_FILE}: {attr_list_file_path}")
            file_list = f.readlines()
            f.close()
            attr_to_num = {file_list[1].split(' ')[:-1][idx]: idx for idx in range(len(file_list[1].split(' ')[:-1]))}
            
            # select images with attributes
            x_train_all = []
            img_path_list = []
            for i, line in enumerate(file_list[2:]):
                # attriutes
                cond_list = [True]
                attr_array = line[:-1].split(' ')
                file_name = attr_array[0]
                file_name_len = len(file_name)
                if (attrs != None) and (attrs != []):
                    for attr in attrs:
                        cond = attr_array[attr_to_num[attr] + 2]
                        cond_n = (int(cond) == 1)
                        cond_list.append(cond_n)
                
                if onp.all(onp.array(cond_list)):
                    file_path = os.path.join(self.celeb_a_large_path, self.CELEB_A_LARGE_IMG256_DIR, file_name)
                    img_path_list.append(file_path)
                    
            x_train_all = self._read_images(img_path_list=img_path_list)
            
            # to numpy array
            x_train_all = onp.stack(x_train_all, axis=0)
            x_train_all = x_train_all.astype(onp.float64)
            x_train_all /= 255
            # MODIFIED: Sample data
            if self.flatten:
                x_train_all = onp.reshape(x_train_all, (x_train_all.shape[0], -1))
                # print(f"x_train_all after flatten: {x_train_all.shape}")
                
            x_train_all, x_train = self.__sample_dataset(X=x_train_all)
        elif self.dataset_name == self.IMAGENET:
            if self.imagenet_path is None:
                raise BaseException("Please specify the path of ImageNet dataset")
            
            imagenet = ImageNet(data_path=self.imagenet_path)
            x_train_all = imagenet.get_data('train', classes=attrs)
            # print(f"x_train_all: {x_train_all.shape}")
            
            x_train_all = x_train_all.astype(onp.float64)
            x_train_all /= 255
            # MODIFIED: Sample data
            if self.flatten:
                x_train_all = np.reshape(x_train_all, (x_train_all.shape[0], -1))
                
            x_train_all, x_train = self.__sample_dataset(X=x_train_all)
            
        elif self.dataset_name == self.FEWSHOT:
            if self.fewshot_path is None:
                raise BaseException("Please specify the path of FewShot dataset")
            
            fewshot = FewShotDataset(data_path=self.fewshot_path, shape=self.FEWSHOT_SHAPE)
            if isinstance(attrs, list):
                x_train_all = fewshot.get_data(catergory=str(attrs[0]))
            elif isinstance(attrs, str):
                x_train_all = fewshot.get_data(catergory=str(attrs))
            else:
                raise TypeError(f"Argument 'attrs' can only be either a string or list of string, but a {type(attrs)}")
            # print(f"x_train_all: {x_train_all.shape}")
            
            x_train_all = x_train_all.astype(np.float64)
            x_train_all /= 255
            # MODIFIED: Sample data
            if self.flatten:
                x_train_all = np.reshape(x_train_all, (x_train_all.shape[0], -1))
                
            x_train_all, x_train = self.__sample_dataset(X=x_train_all)
            
        elif self.dataset_name == self.GAUSSIAN_8:
            width = 1
            height = 1
            r = 0.4
            n_mode = 8
            scale = 0.002
            gauss_gen = Mixed_Gaussian()
            x_train_all = gauss_gen.circle(n_sample=self.dataset_size, n_mode=n_mode, r=r, scale=scale, width=width, height=height)
            # x_train = x_train_all[:self.train_size]
            print(f"x_train_all: {x_train_all.shape}")
            x_train_all, x_train = self.__sample_dataset(X=x_train_all)

        elif self.dataset_name == self.GAUSSIAN_25:
            width = 1
            height = 1
            sqrt_mode = 5
            scale = 0.0005
            gauss_gen = Mixed_Gaussian()
            x_train_all = gauss_gen.square(n_sample=self.dataset_size, sqrt_mode=sqrt_mode, scale=scale, width=width, height=height)
            # x_train = x_train_all[:self.train_size]
            x_train_all, x_train = self.__sample_dataset(X=x_train_all)
            
        return onp.array(x_train), onp.array(x_train_all)
    
class Mixed_Gaussian():
    CIRCLE_GEN_TYPE = 'CIRCLE'
    SQUARE_GEN_TYPE = 'SQUARE'
    HEATMAP = 'heatmap'
    SCATTER = 'scatter'

    def __init__(self, seed: int=0):
        self.rnd_key = random.PRNGKey(seed)
    
    def __get_key(self) -> np.ndarray:
        key, subkey = random.split(self.rnd_key)
        self.rnd_key = subkey
        return subkey

    def __circle_gauss(self, n_mode: int, r: float, scale: float, width: float, height: float):
        mean_vecs = []
        cov_mats = []
        for i in range(n_mode):
            theta = 2 * np.pi / n_mode * i
            x, y = pol2cart(r, theta)
            mean_vec = np.array([x + width/2, y + height/2])
            cov_mat = np.eye(2) * scale
            mean_vecs.append(mean_vec)
            cov_mats.append(cov_mat)

        return np.array(mean_vecs), np.array(cov_mats)

    def __square_gauss(self, sqrt_mode: int, scale: float, width: float, height: float):
        mean_vecs = []
        cov_mats = []
        x_start = width * 0.15
        y_start = height * 0.15
        x_gap = (width - 2 * x_start) / (sqrt_mode - 1)
        y_gap = (height - 2 * y_start) / (sqrt_mode - 1)
        # print(f"X_start: {x_start} Y_start: {y_start} X_gap: {x_gap} Y_gap: {y_gap}")
        for i in range(sqrt_mode):
            for j in range(sqrt_mode):
                mean_vec = np.array([x_start + x_gap * i, y_start + y_gap * j ])
                cov_mat = np.eye(2) * scale
                mean_vecs.append(mean_vec)
                cov_mats.append(cov_mat)

        return np.array(mean_vecs), np.array(cov_mats)
    
    def __mode_count(self, n_sample: int, n_mode: int):
        mode_choices = random.randint(key=self.__get_key(), shape=(n_sample,), minval=0, maxval=n_mode)
        uniques, counts = np.unique(mode_choices, return_counts=True)
        return counts

    def __sample_mixed_gauss(self, n_mode: int, mean_vecs: np.ndarray, cov_mats: np.ndarray, counts: np.array):
        samples = []
        for i in range(n_mode):
            xy = random.multivariate_normal(key=self.__get_key(), mean=mean_vecs[i], cov=cov_mats[i], shape=(counts[i], ))
            samples.append(xy)
        samples = np.concatenate(samples, axis=0)
        return random.permutation(key=self.__get_key(), x=samples)

    def circle(self, n_sample: int, n_mode: int, r: Union[int, float], scale: Union[int, float], width: float, height: float) -> np.ndarray:
        counts = self.__mode_count(n_sample=n_sample, n_mode=n_mode)
        mean_vecs, cov_mats = self.__circle_gauss(n_mode=n_mode, r=r, scale=scale, width=width, height=height)

        return self.__sample_mixed_gauss(n_mode=n_mode, mean_vecs=mean_vecs, cov_mats=cov_mats, counts=counts)

    def square(self, n_sample: int, sqrt_mode: int, scale: Union[int, float], width: float, height: float) -> np.ndarray:
        n_mode = sqrt_mode * sqrt_mode
        counts = self.__mode_count(n_sample=n_sample, n_mode=n_mode)
        mean_vecs, cov_mats = self.__square_gauss(sqrt_mode=sqrt_mode, scale=scale, width=width, height=height)

        return self.__sample_mixed_gauss(n_mode=n_mode, mean_vecs=mean_vecs, cov_mats=cov_mats, counts=counts)

    @staticmethod
    def visualize(datas: np.ndarray, width: int, height: int, title=None, plot_type: str=None, xlim: object={'left': 0, 'right': 1}, ylim: object={'bottom': 0, 'top': 1}, 
                  is_axis: bool=True, fig_path: str=None, is_show_fig: bool=False, dpi=300, fig_size: Tuple=(9, 9), background_color: str=None):
        if fig_path is not None:
            fig_dir = os.path.dirname(fig_path)
            check_make_dir(fig_dir)

        plt.clf()
        fig = plt.figure(figsize=fig_size, dpi=dpi, facecolor=background_color)
        if not is_axis:
            plt.axis('off')

        if title is not None:
            plt.title(title)
        plt.tight_layout()
        
        if xlim != None:
            plt.xlim(**xlim)
        if ylim != None:
            plt.ylim(**ylim)

        if plot_type is Mixed_Gaussian.SCATTER:
            xs = datas[:, 0]
            ys = datas[:, 1]
            plt.scatter(xs, ys)
        else:
            xs = datas[:, 0]
            ys = datas[:, 1]
            heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=(width, height))
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            plt.imshow(heatmap.T, extent=extent, origin='lower')

        if fig_path is not None:
            plt.savefig(fig_path)

        if is_show_fig:
            plt.show()
        plt.close()


class NoiseGenerator():
    def __init__(self, random_seed: int):
        self.rnd_key = random.PRNGKey(random_seed)

    def __get_key(self) -> np.ndarray:
        key, subkey = random.split(self.rnd_key)
        self.rnd_key = subkey
        return subkey

    def gen_noise(self, noise_size: int, image_shape: Tuple, flatten: bool=False):
        #Generate noise
        x_noise = random.uniform(self.__get_key(), shape=(noise_size, *image_shape), minval=0, maxval=1.0)

        if flatten:
            x_noise = np.reshape(x_noise, (noise_size, -1))
        return x_noise
