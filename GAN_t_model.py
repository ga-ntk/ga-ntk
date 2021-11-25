#!/usr/bin/env python
# coding: utf-8
# %%
import os
import argparse
from typing import List
import jax.numpy as np

# MODIFIED: Customize modules
from util import TrendRecorder, get_exp_path, find_best, init_env, save_infos, parse_dataset_features
from plot import Plotter
from model import Model, NTK_Generative
from dataset import Dataset, Mixed_Gaussian
from loss_fn import LossFn

# %%
def ntk_generative(dataset_name: str, noise_size: int, dataset_size: int, train_t_rate: float, epoch: int, model_type: str, dir_name: str, training_seed: int, dataset_seed: int, gpu_id: str, base_path: str='results',
                   dataset_features: List[str]=None, target_class: int=None, train_size: int=None, parallel_num: int=1, learning_rate: float=1000, loss_type: str=LossFn.ORIGIN_LOSS, 
                   alpha: float=0, perturb_coef: float=0, perturb_method: str=None, augmentation: str=None, celeb_a_path: str=None, imagenet_path: str=None, fewshot_path: str=None,
                   save_raw_data: bool=False, save_fig: bool=False, show_fig: bool=False, save_cmp: bool=False):
    init_env(gpu_id=gpu_id)
    print("Inited")

    plot_per_epoch = -1
    predict_per_epoch = -1
    row_num, col_num = None, None

    train_t_unit = 65536.0 
    t = train_t_unit * train_t_rate

    # Result directory
    result_path = get_exp_path(base_path=base_path, exp_name=dir_name, info_list=[dataset_name, f"{model_type}", f"t{train_t_rate}", f"{perturb_method}", f"nc{perturb_coef}", f"aug-{'-'.join(str(augmentation or '').split(','))}", f"bs{train_size}", f"ds{dataset_size}", f"tseed{training_seed}"])
    save_infos(data_path=result_path, training_infos=args.__dict__)
    print("Directory")

    # Generate Dataset
    dataset_generator = Dataset(dataset_name=dataset_name, celeb_a_path=celeb_a_path, imagenet_path=imagenet_path, fewshot_path=fewshot_path, 
                                seed=dataset_seed, flatten=Model.get_flatten(model_type=model_type))
    dataset_generator.set_sample_size(noise_size=noise_size, train_size=train_size, dataset_size=dataset_size)
    image_shape, vec_size = dataset_generator.get_data_shape()
    print("Dataset Generator")

    # Generate Label
    x_train, x_train_all = dataset_generator.gen_data_attrs(target_class=target_class, attrs=dataset_features)
    print("Generate Dataset")

    # Utility class
    trend_recorder = TrendRecorder(data_path=result_path)
    plotter = Plotter(dataset_name=dataset_name, data_path=result_path)
    print("Utility class")
    
    # Training model
    _plot_per_epoch = plot_per_epoch
    if _plot_per_epoch == -1:
        _plot_per_epoch = int(epoch // 100)
        if _plot_per_epoch <= 1:
            _plot_per_epoch = 1
            
    def callback(x_noise_concat, gs_mean, ps_mean, losses_mean, epochs_mean, ntk_predictions_mean, is_compute_prediction_any):
        # Record
        trend_recorder.record(name=TrendRecorder.MEAN_OF_GRAD_CURVE_NAME, value=gs_mean, time=epochs_mean)
        trend_recorder.record(name=TrendRecorder.PERTURB_CURVE_NAME, value=ps_mean, time=epochs_mean)
        trend_recorder.record(name=TrendRecorder.LOSS_CURVE_NAME, value=losses_mean, time=epochs_mean)
        if is_compute_prediction_any:
            trend_recorder.record(name=TrendRecorder.NTK_PREDICTION_CURVE_NAME, value=ntk_predictions_mean, time=epochs_mean)

        # Plot
        if epochs_mean % _plot_per_epoch == 0:
            plotter.record(name=Plotter.GENERATE_IMAGE_PROGRESS_NAME, value=x_noise_concat, time=epochs_mean)
            
            if save_raw_data:
                trend_recorder.save_all(file_name=TrendRecorder.TREND_DATA_NAME)
                plotter.save_seq(Plotter.GENERATE_IMAGE_PROGRESS_NAME, file_name=Plotter.DATA_RECORD_NAME)
                plotter.save(data=x_noise_concat, file_name=Plotter.GENERATIVE_DATA_NAME)    
            
            if dataset_name == Dataset.GAUSSIAN_8 or dataset_name == Dataset.GAUSSIAN_25:
                plotter.plot(data=x_noise_concat, title=f"Epoch {epochs_mean}", row_num=None, col_num=None, hspace=None, wspace=0.02, fig_name=Plotter.epoch_img_name(epoch=epochs_mean), plot_type=Mixed_Gaussian.SCATTER)
            else:
                plotter.plot(data=x_noise_concat, title=f"Epoch {epochs_mean}", row_num=row_num, col_num=col_num, hspace=None, wspace=0.02, fig_name=Plotter.epoch_img_name(epoch=epochs_mean))
    
    ntk_gen = NTK_Generative(model_type=model_type, loss_type=loss_type, alpha=alpha, learning_rate=learning_rate, t=t, seed=training_seed, perturb_method=perturb_method, perturb_coef=perturb_coef)
    x_noise  = ntk_gen.generate_v2(x_train=x_train, x_train_all=x_train_all, noise_size=noise_size, epoch=epoch, parallel_num=parallel_num, predict_per_epoch=predict_per_epoch, callback=callback)
    
    trend_recorder.visualize(name=TrendRecorder.LOSS_CURVE_NAME, is_save_fig=save_fig, is_show_fig=show_fig)
    trend_recorder.visualize(name=TrendRecorder.NTK_PREDICTION_CURVE_NAME, is_save_fig=save_fig, is_show_fig=show_fig, ylim={'bottom': 0, 'top': 1})
    trend_recorder.visualize(name=TrendRecorder.MEAN_OF_GRAD_CURVE_NAME, is_save_fig=save_fig, is_show_fig=show_fig)
    trend_recorder.visualize(name=TrendRecorder.PERTURB_CURVE_NAME, is_save_fig=save_fig, is_show_fig=show_fig)
    trend_recorder.visualize(name=TrendRecorder.PERTURB_CURVE_NAME, is_save_fig=save_fig, is_show_fig=show_fig, y_log_scale=True, postfix='-log')

    def title_fn(t):
        return f"Epoch {t}"

    # Plot final reaults
    if dataset_name == Dataset.GAUSSIAN_8 or dataset_name == Dataset.GAUSSIAN_25:
        plotter.plot(data=x_noise, title=f"Scatter", row_num=None, col_num=None, hspace=0, wspace=0, fig_name=Plotter.FINAL_SCATTER_IMG_NAME, plot_type=Mixed_Gaussian.SCATTER)
        plotter.plot(data=x_noise, title=f"Heat Map", row_num=None, col_num=None, hspace=0, wspace=0, fig_name=Plotter.FINAL_HEATMAP_IMG_NAME, plot_type=Mixed_Gaussian.HEATMAP)
        plotter.plot_seq(name=Plotter.GENERATE_IMAGE_PROGRESS_NAME, row_num=None, col_num=None, title_fn=title_fn, hspace=0, wspace=0, plot_per_epoch=plot_per_epoch, plot_type=Mixed_Gaussian.SCATTER)
    else:
        plotter.plot(data=x_noise, title=f"Final Result", row_num=row_num, col_num=col_num, hspace=0, wspace=0, fig_name=Plotter.FINAL_IMG_NAME)
    
    # Save raw image
    if save_raw_data:
        trend_recorder.save_all(file_name=TrendRecorder.TREND_DATA_NAME)
        plotter.save_seq(Plotter.GENERATE_IMAGE_PROGRESS_NAME, file_name=Plotter.DATA_RECORD_NAME)
        plotter.save(data=x_noise, file_name=Plotter.GENERATIVE_DATA_NAME)

    # get closest sample by cosine similarity
    if save_cmp:
        x_noise_flatten = np.asarray(x_noise.reshape(-1, vec_size))
        x_train_all_flatten = np.asarray(x_train_all.reshape(-1, vec_size))
        num = 0 # the generated image you want to compare
        for targetIdx in range(0, noise_size):
            best_list = find_best(targetIdx, x_noise_flatten, x_train_all_flatten)
            best_images = np.asarray([x_train_all[best_list[i]] for i in range(9)])
            compared_images = np.concatenate([np.expand_dims(x_noise[targetIdx], 0), best_images])
            plotter.plot(data=compared_images, title=f"Similarity Result", row_num=row_num, col_num=col_num, hspace=0, wspace=0, fig_name=f'similarity-{targetIdx}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTK image generation.')
    parser.add_argument('--dataset_name', type=str,
                        help='dataset_name = mnist | cifar10 | celeb_a')
    parser.add_argument('--dataset_features', type=str, default=None,
                        help='filter the CelebA images with features')
    parser.add_argument('--celeb_a_path', type=str, default=None,
                        help='path of CelebA dataset')
    parser.add_argument('--imagenet_path', type=str, default=None,
                        help='path of ImageNet dataset')
    parser.add_argument('--fewshot_path', type=str, default=None,
                        help='path of FewShot dataset')
    parser.add_argument('--target_class', type=int, default=None,
                        help='specify the class/catergory of the training images')
    parser.add_argument('--train_size', type=int, default=None,
                        help='training set size')
    parser.add_argument('--noise_size', type=int,
                        help='noise set size')
    parser.add_argument('--parallel_num', type=int, default=1,
                        help='times of parallel generating')
    parser.add_argument('--dataset_size', type=int,
                        help='dataset set size')
    parser.add_argument('--model_type', type=str,
                        help='model architecture = fnn | cnn | cnn-big')
    parser.add_argument('--learning_rate', type=float, default=1000,
                        help='the update scaling on the generative image')
    parser.add_argument('--epoch', type=int,
                        help='iterations for training')
    parser.add_argument('--train_t_rate', type=float,
                        help='the training time of the NTK, multiply with 65536.0')
    parser.add_argument('--loss_type', type=str, default='origin',
                        help='the type of loss function = origin | mse | cross | similarity')
    parser.add_argument('--alpha', type=float, default=0,
                        help='the weight ot the regularizer of similarity-regularization loss')
    parser.add_argument('--perturb_coef', type=float, default=0,
                        help='perturbation coefficient')
    parser.add_argument('--perturb_method', type=str, default=None,
                        help='the method of perturbation = none | exp | anneal | static')
    parser.add_argument('--augmentation', type=str, default=None,
                        help='the method of augmentation = color, translation, cutout')
    parser.add_argument('--base_path', type=str, default='results', 
                        help='The path that put the experiment results')
    parser.add_argument('--dir_name', type=str,
                        help='the folder name of the results')
    parser.add_argument('--training_seed', type=int, default=1,
                        help='random seed of the training progress')
    parser.add_argument('--dataset_seed', type=int, default=1,
                        help='random eed of the Dataset')
    parser.add_argument('--save_raw_data', action='store_true',
                        help='save the raw image')
    parser.add_argument('--save_fig', action='store_true',
                        help='save the plotting figure')
    parser.add_argument('--show_fig', action='store_true',
                        help='show the plotting figure')
    parser.add_argument('--save_cmp', action='store_true',
                        help='save the comparison results')
    parser.add_argument('--gpu_id', type=str,
                        help='gpu_id')
    args = parser.parse_args()
    
    args.dataset_features = parse_dataset_features(features=args.dataset_features)
    ntk_generative(**args.__dict__)
