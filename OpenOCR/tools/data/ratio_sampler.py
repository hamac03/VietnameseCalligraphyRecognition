import math
import os
import random
import numpy as np
import torch
from torch.utils.data import Sampler

class RatioSampler(Sampler):
    def __init__(self,
                 data_source,
                 scales,
                 first_bs=1,
                 fix_bs=True,
                 divided_factor=[1, 1],
                 is_training=True,
                 max_ratio=1,
                 max_bs=1,
                 seed=None):
        self.data_source = data_source
        self.ds_width = data_source.ds_width
        self.seed = data_source.seed
        self.n_data_samples = len(self.data_source)
        self.max_ratio = max_ratio
        self.max_bs = max_bs
        self.base_batch_size = first_bs
        self.base_im_h = scales[0][1] if isinstance(scales[0], list) else scales[0]
        self.base_im_w = scales[0][0] if isinstance(scales[0], list) else scales[0]

        num_replicas = torch.cuda.device_count()
        rank = (int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0)
        num_samples_per_replica = int(math.ceil(self.n_data_samples * 1.0 / num_replicas))

        self.img_indices = list(range(self.n_data_samples))
        self.shuffle = is_training
        self.is_training = is_training
        
        if is_training:
            indices_rank_i = self.img_indices[rank:len(self.img_indices):num_replicas]
        else:
            indices_rank_i = self.img_indices
            
        self.indices_rank_i_ori = np.array(indices_rank_i)
        self.batch_list = [[self.base_im_w, self.base_im_h, idx, 1] for idx in indices_rank_i]
        self.length = len(self.batch_list)
        self.batchs_in_one_epoch_id = list(range(self.length))
        self.epoch = 0
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def __iter__(self):
        if self.shuffle:
            self.epoch += 1
            random.seed(self.epoch) 
            random.shuffle(self.batchs_in_one_epoch_id)
        for batch_id in self.batchs_in_one_epoch_id:
            yield [self.batch_list[batch_id]]

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.length