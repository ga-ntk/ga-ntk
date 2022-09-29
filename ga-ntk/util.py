import configparser as Parser
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Tuple, Type
from datetime import datetime, time
from enum import Enum
import os
from textwrap import wrap
import json
import pickle
import math
from PIL.Image import NONE

import matplotlib.pyplot as plt
import numpy as onp
import jax.numpy as np
from jax import random
from jax.config import config

def init_env(gpu_id: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
    os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
    # os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
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
    infos = '_'.join(info_list)
    
    print(f'Current Directory: {cwd}')
    result_path = os.path.join(cwd, base_path, exp_name, infos)
    if os.path.exists(result_path):
        os.rmdir(result_path)
    os.makedirs(result_path)
    
    return result_path

def parse_dataset_features(features):
    feats_list = []
    if isinstance(features, str):
        feats_list = features.split(',')
    return feats_list

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

def row_col(item_num: int):
    if not isinstance(item_num, int):
        raise TypeError(f"The parameter item_num should be an integer")
    item_num_sqrt = int(math.sqrt(item_num))
    row = 1
    idx = 1
    while idx <= item_num_sqrt:
        if (item_num % idx) == 0:
            row = idx
        idx += 1
    
    return row, int(item_num // row)

class PerturbationMethod(Enum):
    exp = auto()
    anneal = auto()
    static = auto()

class Perturbation():
    def __init__(self, coef: float, method: str, seed: int, is_manual_decay: bool=False) -> None:
        self.rnd_key = random.PRNGKey(seed)
        self.counter = 0
        self.coef = coef
        self.method = method
        self.is_manual_decay = is_manual_decay
        
        self.__gen = self.__get_generate_fn()
        
    def __exp_decay(self) -> Tuple[float, float]:
        self.counter += 1
        shift = 0
        scale = self.coef ** self.counter
        return shift, scale
    
    def __anneal_decay(self) -> Tuple[float, float]:
        self.counter += 1
        shift = 0
        scale = (1 + self.counter) ** self.coef
        return shift, scale
    
    def __linear_decay(self) -> Tuple[float, float]:
        self.counter += 1
        shift = 0
        scale = (1 + self.counter) ** self.coef
        return shift, scale
    
    def __get_key(self) -> np.ndarray:
        key, subkey = random.split(self.rnd_key)
        return subkey
    
    def __add_counter(self) -> None:
        self.counter += 1
    
    def __clear_counter(self) -> None:
        self.counter = 0
        
    def __get_generate_fn(self) -> Callable[[Tuple[int, ...]], np.ndarray]:
        switch = {
            PerturbationMethod.exp.name: self.__gen_decay_normal,
            PerturbationMethod.anneal.name: self.__gen_anneal_normal,
            PerturbationMethod.static.name: self.__gen_static_normal
        }
        
        if self.method is None:
            return self.__gen_none
        if isinstance(self.method, str):
            if self.method.lower() == 'none':
                return self.__gen_none
        
        if switch.get(self.method, None) != None:
            return switch[self.method]
        else:
            raise ValueError(f'No such method {self.method} in class Perturbation')
    
    def __gen_static_normal(self, shape: Tuple[int, ...]) -> np.ndarray:
        """
        A stateless standard normal random variable generator
        """
        return random.normal(key=self.__get_key(), shape=shape) * self.coef
    
    def __gen_decay_normal(self, shape: Tuple[int, ...]) -> np.ndarray:
        """
        A stateful normal random variable generator, follow exponential decay
        """
        # if not self.is_manual_decay:
        self.__add_counter()
        shift, scale = self.__exp_decay()
        
        return shift + random.normal(key=self.__get_key(), shape=shape) * scale
    
    def __gen_anneal_normal(self, shape: Tuple[int, ...]) -> np.ndarray:
        """
        A stateful normal random variable generator, follow anneal decay
        """
        # if not self.is_manual_decay:
        self.__add_counter()
        shift, scale = self.__anneal_decay()
        
        return shift + random.normal(key=self.__get_key(), shape=shape) * scale
    
    def __gen_none(self, shape: Tuple[int, ...]) -> np.ndarray:
        return np.zeros(shape=shape)
    
    def reset_seeed(self, seed: int) -> None:
        self.rnd_key = random.PRNGKey(seed)

    def generate(self, shape: Tuple[int, ...]) -> np.ndarray:
        return self.__gen(shape=shape)

    def decay(self):
        if self.is_manual_decay:
            self.__add_counter()
        else:
            print(f"Warning: You shouldn't use method decay() unless you set 'is_manual_decay' as True")

    def reset_decay(self) -> None:
        self.__clear_counter()

def save_infos(data_path: str, training_infos: dict):
    if training_infos != None:
        json_path = os.path.join(data_path, 'training_infos.json')
        with open(json_path, 'w') as fp:
            json.dump(training_infos, fp)
    else:
        raise ValueError(f"training_infos shouldn't be None")

class Recorder():
    PICKLE_NAME_ENTRY = 'name'
    PICKLE_DATA_ENTRY = 'data'
    PICKLE_VALUE_ENTRY = 'value'
    PICKLE_TIME_ENTRY = 'time'

    def __init__(self, data_path: str=None) -> None:
        self.recorder = {}
        self.time_recorder = {}
        self.data_path = data_path

    def set_data_path(self, data_path: str) -> None:
        self.data_path = data_path

    def get_record(self, name: str, idx: int) -> Tuple[object, int]:
        old_values, old_times = self.get_records(name=name)
        
        if isinstance(old_values, list):
            query_value = old_values[idx]
        else:
            raise ValueError(f"There is no corresponding record named {name}")

        if isinstance(old_times, list):
            query_time = old_times[idx]
        return query_value, query_time

    def get_seq(self, name: str) -> Tuple[List[object], List[int]]:
        return self.recorder.get(name, None), self.time_recorder.get(name, None)

    def set_record(self, name: str, idx: int, value: object, time: int) -> None:
        old_values, old_times = self.get_records(name=name)
        if old_values is not None:
            if idx < len(old_values):
                self.recorder[name][idx] = value
            else:
                raise ValueError(f"The idx exceed the length of the records of '{name}'")
        else:
            raise ValueError(f"There is no corresponding record named {name}")

        if old_times is not None:
            if idx < len(old_times):
                self.time_recorder[name][idx] = time
            else:
                raise ValueError(f"The idx exceed the length of the records of '{name}'")

    def set_seq(self, name: str, values: List[object], times: List[int]=None) -> None:
        self.__create_seq(name=name, values=values, times=times)

    def __create_record(self, name: str, value: object, time: int=None):
        self.recorder[name] = [value]
        if time != None:
            self.time_recorder[name] = [time]
        else:
            self.time_recorder[name] = None
    
    def __create_seq(self, name: str, values: List[object], times: int=None):
        self.recorder[name] = values
        if times != None:
            if len(values) == len(times):
                self.time_recorder[name] = times
            else:
                raise ValueError(f"The length of values and times should be equal, but {len(values)} and {len(times)} respectively")
        else:
            self.time_recorder[name] = None

    def __append_record(self, name: str, value: object, time: int=None):
        if self.recorder.get(name, None) == None:
            raise ValueError(f"There is no corresponding record named {name}")
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

    def __append_seq(self, name: str, values: List[object], times: List[int]=None):
        if self.recorder.get(name, None) == None:
            raise ValueError(f"There is no corresponding record named {name}")
        else:
            self.recorder[name] += values
            if times != None:
                if len(values) == len(times):
                    self.time_recorder[name] += times
                else:
                    raise ValueError(f"The length of values and times should be equal, but {len(values)} and {len(times)} respectively")
            else:
                self.time_recorder[name] = None

    def record(self, name: str, value: object, time: int=None) -> None:
        if self.recorder.get(name, None) == None:
            self.__create_record(name=name, value=value, time=time)
        else:
            self.__append_record(name=name, value=value, time=time)

    def record_seq(self, name: str, values: List[object], times: List[int]=None) -> None:
        if isinstance(values, list):
            if self.recorder.get(name, None) == None:
                self.__create_seq(name=name, values=values, times=times)
            else:
                self.__append_seq(name=name, values=values, times=times)
        else:
            raise TypeError(f"Values should be a List")
            
    def clear(self) -> None:
        self.recorder.clear()
        self.time_recorder.clear()

    def __update_recorder(self, value_dict: dict, time_dict: dict) -> None:
        old_value_dict = self.recorder
        old_time_dict = self.time_recorder
        self.recorder = dict(old_value_dict, **value_dict)
        self.time_recorder = dict(old_time_dict, **time_dict)

    def _make_data_path(self, file_name: str):
        if file_name is None:
            return None

        check_make_dir(path=self.data_path)
        data_path = os.path.join(self.data_path, file_name)
        return data_path

    def __dump_pickle_dict(self, name: str, values: List[object], times: List[int]) -> dict:
        value_dict = {name: values}
        time_dict = {name: times}
        return value_dict, time_dict

    def __dump_pickle(self, value_dict: dict, time_dict: dict) -> dict:
        return {self.PICKLE_VALUE_ENTRY: value_dict, self.PICKLE_TIME_ENTRY: time_dict}

    def __load_pickle(self, data: dict) -> Tuple[dict, dict]:
        if not isinstance(data, dict):
            raise TypeError(f"The data should be a dictinoary, but a {type(data)}")

        value_dict = data.get(self.PICKLE_VALUE_ENTRY, None)
        if value_dict is None:
            raise TypeError(f"The field '{self.PICKLE_VALUE_ENTRY}' is empty or None")

        time_dict  = data.get(self.PICKLE_TIME_ENTRY, None)

        return value_dict, time_dict

    def save(self, data: object, file_name: str) -> None:
        data_path = self._make_data_path(file_name=file_name)
        with open(data_path, 'wb') as f:
            if file_name.split('.')[-1] == 'pkl':
                pickle.dump(data, f)
            elif file_name.split('.')[-1] == 'npy':
                onp.save(f, data)
            else:
                pickle.dump(data, f)

    def read(self, file_name: str, is_clip: bool=False) -> object:
        data_path = self._make_data_path(file_name=file_name)
        with open(data_path, 'rb') as f:
            if file_name.split('.')[-1] == 'pkl':
                data = pickle.load(f)
            elif file_name.split('.')[-1] == 'npy':
                data = onp.load(f)
                if is_clip:
                    data = (data + 1) / 2
                # print(f"{onp.max(data)} {onp.min(data)}")
            else:
                data = pickle.load(f)
                
        return data

    def __save(self, value_dict: dict, time_dict: dict, file_name: str) -> None:
        self.save(data=self.__dump_pickle(value_dict=value_dict, time_dict=time_dict), file_name=file_name)

    def __read(self, file_name: str) -> Tuple[dict, dict]:
        value_dict, time_dict = self.__load_pickle(data=self.read(file_name=file_name))
        return value_dict, time_dict

    def save_seq(self, name: str, file_name: str) -> None:
        """
        Save raw data sequence with pickle format
        """
        values = self.recorder.get(name, None)
        times = self.time_recorder.get(name, None)
        if values != None:
            value_dict, time_dict = self.__dump_pickle_dict(name=name, values=values, times=times)
            self.__save(value_dict=value_dict, time_dict=time_dict, file_name=file_name)
        else:
            raise ValueError(f"No coresponding record named '{name}' in the Trend_Recorder")

    def read_seq(self, name: str, file_name: str, is_load_to_recorder: bool=True, load_to_recorder_name: str=None) -> Tuple[List[np.ndarray], List[int]]:
        """
        Read raw data sequence in pickle format
        """
        data_path = self._make_data_path(file_name=file_name)
        value_dict, time_dict = self.__read(data_path)
        values = value_dict.get(name, None)
        times = time_dict.get(name, None)

        if values is not None:
            if is_load_to_recorder:
                # self.__update_recorder(value_dict=value_dict, time_dict=time_dict)
                new_name = load_to_recorder_name
                if load_to_recorder_name is None:
                    new_name = name
                self.__create_seq(name=new_name, values=values, times=times)
            
        return values, times

    def save_all(self, file_name: str) -> None:
        self.__save(value_dict=self.recorder, time_dict=self.time_recorder, file_name=file_name)

    def read_all(self, file_name: str, is_load_to_recorder: bool=True) -> Tuple[dict, dict]:
        value_dict, time_dict = self.__read(file_name=file_name)
        if is_load_to_recorder:
            self.__update_recorder(value_dict=value_dict, time_dict=time_dict)

        return value_dict, time_dict

class TrendRecorder(Recorder):
    PICKLE_RECORD_ENTRY = 'record'
    PICKLE_TIME_ENTRY = 'time'
    
    LOSS_CURVE_NAME = 'loss_curve'
    MEAN_OF_GRAD_CURVE_NAME = 'mean_of_grad_curve'
    NTK_PREDICTION_CURVE_NAME = 'ntk_prediction_curve'
    PERTURB_CURVE_NAME = 'perturb_curve'
    
    TREND_DATA_NAME = 'trend.pkl'

    def __init__(self, data_path: str=None, training_infos: dict=None) -> None:
        self.recorder = {}
        self.time_recorder = {}
        self.data_path = data_path
        self.training_infos = training_infos
        
    def __auto_y_scale(self, name: str) -> Dict[str, float]:
        record = self.recorder.get(name, None)
        if record != None:
            ylim = {'bottom': min(record), 'top': max(record)}
            return ylim
        else:
            raise ValueError(f"No coresponding record named '{name}' in the Trend_Recorder")

    def visualize(self, name: str, is_save_fig: bool, is_show_fig: bool, indice: slice=None, title: str=None, xlabel: str=None, ylabel: str=None,
                  xlim: object=None, ylim: object=None, y_log_scale: bool=False, auto_y_scale: bool=False, text_size: int=12, tick_size: int=10,
                  dpi: int=300, format: str='png', postfix: str='', background_color: str=None) -> None:
        record = self.recorder.get(name, None)    
        if  record != None:
            # xs = [i*100 for i in range(len(record))]
            plt.figure(dpi=dpi, facecolor=background_color)
            # plt.rcParams.update({'font.size': font_size})
            xs = self.time_recorder.get(name, None)
            if xs != None:
                if indice is not None:
                    plt.plot(xs[indice], record[indice])
                else:
                    plt.plot(xs, record)
            else:
                if indice is not None:
                    plt.plot(record[indice])
                else:
                    plt.plot(record)
                
            if title is not None:
                plt.title(title, fontsize=text_size)
            else:
                plt.title(name, fontsize=text_size)

            # Set the range of x-axis and y-axis
            axes = plt.gca()
            if xlabel != None:
                axes.set_xlabel(xlabel, fontsize=text_size)
            if ylabel != None:
                axes.set_ylabel(ylabel, fontsize=text_size)
            if xlim != None:
                axes.set_xlim(**xlim)
            if ylim != None:
                axes.set_ylim(**ylim)
            elif auto_y_scale:
                axes.ylim(**self.__auto_y_scale(name=name))
                
            if y_log_scale:
                axes.set_yscale('log')
                
            plt.xticks(fontsize=tick_size)
            plt.yticks(fontsize=tick_size)
            
            if is_save_fig:
                data_path = os.path.join(self.data_path, f'{name}{postfix}.{format}')
                plt.savefig(data_path, dpi=dpi)
            
            # Show figure or close it
            if is_show_fig:
                plt.show()
            else:
                plt.close()
        else:
            raise ValueError(f"No coresponding record named '{name}' in the Trend_Recorder")