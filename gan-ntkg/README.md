# GA-NTKg

Implementation of the paper "Single-level Adversarial Data Synthesis based on Neural Tangent Kernels"

## Quick Run

### pretraining Autoencoder
You should pretrain an autoencoder to support larger dataset.
This repo only demos GA-NTKg with autoencoders.

### training GA-NTKg
For MNIST dataset:
```python
python main_generator.py --dataset_name mnist --train_size 256 --dataset_size 256 --batch_size 64 --epoch 100000 --target_distribution all --training_seed 1 --train_t_rate 2 --gpu_id 0
```


For CIFAR-10 dataset:
```python
python main_generator.py --dataset_name cifar10 --train_size 256 --dataset_size 256 --batch_size 64 --epoch 100000 --target_distribution single --training_seed 1 --train_t_rate 2 --gpu_id 0
```

For CelebA dataset, one should specify the `celeb_a_img_path` and `celeb_a_anno_path`

Below is an example:
```python
python main_generator.py --dataset_name celeb_a --celeb_a_img_path ./myCelebA --celeb_a_anno_path ./myCelebA/Anno/list_attr_celeba.txt --train_size 256 --dataset_size 256 --batch_size 64 --epoch 100000 --target_distribution single --training_seed 1 --train_t_rate 2 --gpu_id 0
```
