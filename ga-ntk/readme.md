# GA-NTK

Implementation of the paper "Single-level Adversarial Data Synthesis based on Neural Tangent Kernels"

## Quick Run

For MNIST dataset:
```python
python GAN_t_model.py --dataset_name mnist  --train_size 256 --noise_size 256 --dataset_size 256 --model_type cnn-mnist --learning_rate 1000 --epoch 100000 --loss_type origin --training_seed 1 --dataset_seed 1 --dir_name mnist --train_t_rate 2 --save_fig  --gpu_id 0
```

For CIFAR-10 dataset:
```python
python GAN_t_model.py --dataset_name cifar10  --train_size 256 --noise_size 256 --dataset_size 256 --model_type cnn-cifar10 --learning_rate 1000 --epoch 100000 --loss_type origin --training_seed 1 --dataset_seed 1 --dir_name cifar10 --target_class 7 --train_t_rate 2 --save_fig  --gpu_id 0
```

For CelebA dataset, one should specify the `celeb_a_path` and put the annotation file in `[celeb_a_path]/Anno/list_attr_celeba.txt`

Below is an example:
```python
python GAN_t_model.py --dataset_name celeb_a  --celeb_a_path ./myCelebA  --dataset_features Male,Straight_Hair  --train_size 256 --noise_size 256 --dataset_size 256 --model_type cnn --learning_rate 1000 --epoch 100000 --loss_type origin --training_seed 1 --dataset_seed 1 --dir_name  celeb_a --train_t_rate 4 --save_fig  --gpu_id 0
```

### Notes
* If you encounter `Out-Of-Memory` error, please reduce the `train_size`, which decreases batch size for GA-NTK.