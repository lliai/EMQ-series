# mixed precision quantization
from .base_models import *  # noqa: F403,F401
from .mobilenet_imagenet import *  # noqa: F403,F401
from .resnet_imagenet import *  # noqa: F403,F401
from .run_manager import RunConfig


class TrainRunConfig(RunConfig):

    def __init__(self,
                 n_epochs=150,
                 init_lr=0.05,
                 lr_schedule_type='cosine',
                 lr_schedule_param=None,
                 dataset='imagenet',
                 train_batch_size=256,
                 test_batch_size=500,
                 opt_type='sgd',
                 opt_param=None,
                 weight_decay=4e-5,
                 label_smoothing=0.1,
                 no_decay_keys='bn',
                 model_init='he_fout',
                 init_div_groups=False,
                 validation_frequency=1,
                 print_frequency=10,
                 n_worker=32,
                 local_rank=0,
                 world_size=1,
                 sync_bn=True,
                 warm_epoch=5,
                 save_path=None,
                 base_path=None,
                 **kwargs):
        super(TrainRunConfig,
              self).__init__(n_epochs, init_lr, lr_schedule_type,
                             lr_schedule_param, dataset, train_batch_size,
                             test_batch_size, opt_type, opt_param,
                             weight_decay, label_smoothing, no_decay_keys,
                             model_init, init_div_groups, validation_frequency,
                             print_frequency, local_rank, world_size, sync_bn,
                             warm_epoch)

        self.n_worker = n_worker
        self.save_path = save_path
        self.base_path = base_path

        print(kwargs.keys())

    @property
    def data_config(self):
        return {
            'train_batch_size': self.train_batch_size,
            'test_batch_size': self.test_batch_size,
            'n_worker': self.n_worker,
            'local_rank': self.local_rank,
            'world_size': self.world_size,
            'save_path': self.save_path,
        }


class SearchRunConfig(RunConfig):

    def __init__(self,
                 n_epochs=150,
                 init_lr=0.05,
                 lr_schedule_type='cosine',
                 lr_schedule_param=None,
                 dataset='imagenet',
                 train_batch_size=256,
                 test_batch_size=500,
                 opt_type='sgd',
                 opt_param=None,
                 weight_decay=4e-5,
                 label_smoothing=0.1,
                 no_decay_keys='bn',
                 model_init='he_fout',
                 init_div_groups=False,
                 validation_frequency=1,
                 print_frequency=10,
                 n_worker=32,
                 local_rank=0,
                 world_size=1,
                 sync_bn=True,
                 warm_epoch=5,
                 save_path=None,
                 search_epoch=10,
                 target_flops=1000,
                 n_remove=2,
                 n_best=3,
                 n_populations=8,
                 n_generations=25,
                 div=8,
                 **kwargs):
        super(SearchRunConfig,
              self).__init__(n_epochs, init_lr, lr_schedule_type,
                             lr_schedule_param, dataset, train_batch_size,
                             test_batch_size, opt_type, opt_param,
                             weight_decay, label_smoothing, no_decay_keys,
                             model_init, init_div_groups, validation_frequency,
                             print_frequency, local_rank, world_size, sync_bn,
                             warm_epoch)

        self.div = div
        self.n_worker = n_worker
        self.save_path = save_path
        self.search_epoch = search_epoch
        self.target_flops = target_flops
        self.n_remove = n_remove
        self.n_best = n_best
        self.n_populations = n_populations
        self.n_generations = n_generations

        print(kwargs.keys())

    @property
    def data_config(self):
        return {
            'train_batch_size': self.train_batch_size,
            'test_batch_size': self.test_batch_size,
            'n_worker': self.n_worker,
            'local_rank': self.local_rank,
            'world_size': self.world_size,
            'save_path': self.save_path
        }
