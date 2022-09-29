import os
import pickle
from textwrap import wrap
from typing import Callable, Dict, List, Tuple, Union
from PIL.Image import NONE
from jax._src.numpy.lax_numpy import isin

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

from util import check_make_dir, row_col, Recorder
from dataset import Mixed_Gaussian, Dataset

class Plotter(Recorder):
    PICKLE_RECORD_ENTRY = 'record'
    PICKLE_TIME_ENTRY = 'time'
    
    GENERATE_IMAGE_PROGRESS_NAME = 'gen_imgs_progress'
    
    FINAL_SCATTER_IMG_NAME = 'final_scatter.png'
    FINAL_HEATMAP_IMG_NAME = 'final_heatmap.png'
    FINAL_IMG_NAME = 'final.png'
    DATA_RECORD_NAME = 'data_records.pkl'
    GENERATIVE_DATA_NAME = 'generative_data.pkl'
    
    DEFAULT_PADDING_INCH = 0.02
    
    DEFAULT_BACKGROUND_COLOR = 'white'
    WHITE_BACKGROUND_COLOR = 'white'
    BLACK_BACKGROUND_COLOR = 'black'

    def __init__(self, dataset_name: str=None, data_path: str=None):
        self.recorder = {}
        self.time_recorder = {}
        self.dataset_name = dataset_name
        self.data_path = data_path

    def set_dataset_name(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name
    
    @staticmethod
    def epoch_img_name(epoch: int):
        return f'epoch-{epoch}.png'

    def __row_col_parser(self, item_num: int, row: int, col: int):
        if (row is None) and (col is not None):
            return int(item_num // col), col
        elif (row is not None) and (col is None):
            return row, int(item_num // row)
        elif (row is None) and (col is None):
            return row_col(item_num=item_num)
        else:
            return row, col
        
    def __process_space(self, hspace: float, wspace: float):
        if (hspace is None) and (wspace is None):
            axes_pad = self.DEFAULT_PADDING_INCH
        elif hspace is None:
            axes_pad = wspace
        elif wspace is None:
            axes_pad = hspace
        else:
            axes_pad = (wspace, hspace)
        return axes_pad
    
    def plot(self, data: np.ndarray, title: str, indice: Union[slice, List[int]]=None, row_num: int=None, col_num: int=None, hspace: float=None, wspace: float=None, dpi: int=300, fig_data_path: str=None, fig_name: str=None, plot_type: str=Mixed_Gaussian.SCATTER, is_show_fig: bool=False, is_show_img_idx: bool=False, background_color: str=None, borders_color: Tuple[List[str], str]=None) -> None:
        image_shape, vsc_size = Dataset.get_dataset_shape(dataset_name=self.dataset_name)

        if self.dataset_name == Dataset.MNIST or self.dataset_name == Dataset.CIFAR10 or self.dataset_name == Dataset.CELEB_A or self.dataset_name == Dataset.CELEB_A_LARGE or self.dataset_name == Dataset.IMAGENET or self.dataset_name == Dataset.FEWSHOT:    
            if indice is not None:
                if isinstance(indice, list) or isinstance(indice, tuple):
                    data = data[np.ix_(indice)]
                elif isinstance(indice, slice):
                    data = data[indice]
            item_num = data.shape[0]
            row_num_parsed, col_num_parsed = self.__row_col_parser(item_num=item_num, row=row_num, col=col_num)
            # print(f"Row: {row_num_parsed}, Col: {col_num_parsed}")
            w_size_unit = h_size_unit = 0.8
            if is_show_img_idx:
                h_size_unit += 0.1
                if hspace < 0.3:
                    hspace = 0.3
            
            axes_pad = self.__process_space(hspace=hspace, wspace=wspace)
            
            fig = plt.figure(figsize=(w_size_unit*col_num_parsed, h_size_unit*row_num_parsed), dpi=dpi, facecolor=background_color)
            grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(row_num_parsed, col_num_parsed),  # creates 2x2 grid of axes
                 axes_pad=axes_pad,  # pad between axes in inch.
                 )
            
            fig.subplots_adjust(hspace=hspace, wspace=wspace)
            fig.suptitle(title, horizontalalignment='center', wrap=True)
            for i, a in enumerate(grid):
                if is_show_img_idx:
                    a.set_title(i, fontdict={'fontsize': 10})
                a.axis('off')
                a.xaxis.set_visible(False)
                a.yaxis.set_visible(False)
                
                if borders_color is not None:
                    color = None
                    if isinstance(borders_color, list):
                        color = borders_color[i]
                    elif isinstance(borders_color, str):
                        color = borders_color
                    if color is not None:
                        a.axis('on')
                        a.xaxis.set_visible(True)
                        a.yaxis.set_visible(True)
                        a.set_yticks([])
                        a.set_xticks([])
                        for spine in a.spines.values():
                            spine.set_edgecolor(color)
                    
                if self.dataset_name == Dataset.MNIST:
                    img = data[i].reshape(image_shape[:2])
                    a.imshow(img, cmap='gray', vmin=0, vmax=1)
                else:
                    # img = data[idx + row*col_num_parsed].reshape(image_shape)
                    img = data[i].reshape(image_shape)
                    a.imshow(img, vmin=0, vmax=1)

            # plt.tight_layout()
            
            if fig_name is not None:
                if fig_data_path is not None:
                    plt.savefig(os.path.join(fig_data_path, fig_name))
                else:
                    plt.savefig(self._make_data_path(file_name=fig_name))
                
            if is_show_fig:
                plt.show()
            plt.close()
        elif self.dataset_name == Dataset.GAUSSIAN_8 or self.dataset_name == Dataset.GAUSSIAN_25:
            Mixed_Gaussian.visualize(data, title=title, width=256, height=256, plot_type=plot_type, fig_path=self._make_data_path(file_name=fig_name), is_show_fig=is_show_fig, dpi=dpi, background_color=background_color)
    
    def plot_record(self, indice: slice, name: str, title_fn: Callable[[int], str]=None, img_indice: Union[slice, List[int]]=None, row_num: int=None, col_num: int=None, hspace: float=None, wspace: float=None, is_show_fig: bool=False, fig_name: str=None, plot_type: str=Mixed_Gaussian.SCATTER, dpi: int=300, background_color: str=None) -> None:
        record, time = self.get_seq(name=name)
        
        if time is None:
            time = [i for i in range(len(record))]
            
        sliced_records = record[indice]
        sliced_times = time[indice]
            
        for r, t in zip(sliced_records, sliced_times):
            if title_fn is not None:
                self.plot(data=r, indice=img_indice, title=title_fn(t), row_num=row_num, col_num=col_num, hspace=hspace, wspace=wspace, fig_name=fig_name, is_show_fig=is_show_fig, plot_type=plot_type, dpi=dpi, background_color=background_color)
            else:
                self.plot(data=r, indice=img_indice, title="", row_num=row_num, col_num=col_num, hspace=hspace, wspace=wspace, fig_name=fig_name, is_show_fig=is_show_fig, plot_type=plot_type, dpi=dpi, background_color=background_color)
        
    def plot_seq(self, name: str, title_fn: Callable[[int], str], row_num: int=None, col_num: int=None, hspace: float=None, wspace: float=None, plot_per_epoch: int=-1, is_show_fig: bool=False, plot_type: str=Mixed_Gaussian.SCATTER, dpi: int=300, background_color: str=None) -> None:
        record = self.recorder.get(name, None)
        time = self.time_recorder.get(name, None)
        epoch = len(record)
        if plot_per_epoch == -1:
            plot_per_epoch = int(epoch // 100)
            if plot_per_epoch <= 1:
                plot_per_epoch = 1

        if record is not None:
            if time is None:
                time = [i for i in range(len(record))]
            if self.data_path is None:
                for r, t in zip(record, time):
                    self.plot(data=r, title=title_fn(t), row_num=row_num, col_num=col_num, hspace=hspace, wspace=wspace, fig_name=None, is_show_fig=is_show_fig, plot_type=plot_type, dpi=dpi, background_color=background_color)
            else:
                check_make_dir(self.data_path)

                for i, (r, t) in enumerate(zip(record, time)):
                    idx = i + 1
                    if (idx % plot_per_epoch) == 0 or idx == 1:
                        self.plot(data=r, title=title_fn(t), row_num=row_num, col_num=col_num, hspace=hspace, wspace=wspace, fig_name=f"epoch-{int(t)}.png", is_show_fig=is_show_fig, plot_type=plot_type, dpi=dpi, background_color=background_color)
        else:
            raise ValueError(f"There is no corresponding record named {name}")