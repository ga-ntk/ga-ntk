from scalablerunner.taskrunner import TaskRunner

def run(config: dict) -> None: 
    """
    A simple function for running specific config.
    """
    tr = TaskRunner(config=config)
    tr.output_log(file_name='logs/taskrunner.log')
    tr.run()

def search_t_exp():
    """
    
    """
    config = {
        'TWCC adam CELEBA 256': {
            'Group - twcc CELEBA': {
                'Call': 'source /home/$USER/.bashrc; python GAN_t_model.py',
                'Param': {
                    '--dataset_name': ['mnist'],
                    '--celeb_a_path': ['/work/nthudatalab1/CelebA'],
                    # '--dataset_features': ['Male,Straight_Hair'],
                    '--dataset_features': [None],
                    '--train_size': [256],
                    '--noise_size': [256],
                    '--parallel_num': [1],
                    '--dataset_size': [2048],
                    '--model_type': ['fnn'],
                    '--learning_rate': [1000],
                    '--epoch': [10000],
                    '--loss_type': ['origin'],
                    '--training_seed': [1],
                    '--dataset_seed': [1],
                    '--dir_name': ['test'],
                    '--train_t_rate': [1],
                    '--save_fig': [''],
                    '--save_raw_data': [''],
                },
                'Async': {
                    '--gpu_id': ['0']
                }
            },
        },
    }
    run(config=config)

if __name__ == '__main__':
    search_t_exp()

