import os
import sys
import argparse
import time
import numpy as np
from tqdm import tqdm
import random
import logging
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn.init as init
import torch.utils.data
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from fastai.vision import *

from Dino.model.dino_vision import DINO_Finetune
from Dino.utils.utils import Config, Logger, MyConcatDataset
from Dino.utils.util import Averager
from Dino.dataset.dataset_pretrain import ImageDataset, collate_fn_filter_none
from Dino.dataset.datasetsupervised_kmeans import ImageDatasetSelfSupervisedKmeans
from Dino.metric.eval_acc import TextAccuracy
from Dino.modules import utils
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def _set_random_seed(seed):
    cudnn.deterministic = True
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        logging.warning('You have chosen to seed training. '
                        'This will slow down your training!')


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='path to config file')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='batch size')
    parser.add_argument('--test_root', type=str, default=None,
                        help='path to test datasets')
    parser.add_argument('--save_dir', type=str, default="saved_data",
                        help='directory to save GT images and labels')
    args = parser.parse_args()
    config = Config(args.config)
    if args.batch_size is not None:
        config.dataset_test_batch_size = args.batch_size
    if args.test_root is not None:
        config.dataset_test_roots = [args.test_root]
    return config, args.save_dir


def save_batch_images_and_labels(batch, labels, batch_idx, save_dir="saved_data"):
    """Saves each image in the batch and writes corresponding labels to a text file."""
    # Create directories and paths
    images_dir = os.path.join(save_dir, "images_gt")
    os.makedirs(images_dir, exist_ok=True)
    labels_filepath = os.path.join(save_dir, "label_gt.txt")

    # Save each image and write its label
    with open(labels_filepath, "a") as label_file:
        for idx, (img_tensor, label) in enumerate(zip(batch, labels)):
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy
            img_np = (img_np * 255).astype(np.uint8)  # Scale to [0, 255] for saving
            image_filename = f"{batch_idx}_{idx}_{label}.jpg"
            image_filepath = os.path.join(images_dir, image_filename)
            cv2.imwrite(image_filepath, img_np)
            logging.info(f"Saved image: {image_filepath}")

            # Save label in the text file
            label_file.write(f"{image_filename}\t{label}\n")
            logging.info(f"Saved label: {image_filename}\t{label}")


def _get_databaunch(config):
    # Modify this function if you have different dataset structure
    def _get_dataset(ds_type, paths, is_training, config, **kwargs):
        kwargs.update({
            'img_h': config.dataset_image_height,
            'img_w': config.dataset_image_width,
            'case_sensitive': config.dataset_case_sensitive,
            'charset_path': config.dataset_charset_path,
        })
        datasets = []
        for p in paths:
            subfolders = [f.path for f in os.scandir(p) if f.is_dir()]
            if subfolders:  # Concat all subfolders
                datasets.append(_get_dataset(ds_type, subfolders, is_training, config, **kwargs))
            else:
                datasets.append(ds_type(path=p, is_training=is_training, **kwargs))
        return datasets[0]

    dataset_class = ImageDataset  # Modify as needed based on your dataset class
    test_dataloaders = []
    for eval_root in config.dataset_test_roots:
        test_ds = _get_dataset(dataset_class, [eval_root], False, config)
        print(f"batch size: {config.dataset_test_batch_size}")
        test_dataloader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=config.dataset_test_batch_size,
            shuffle=False,
            num_workers=config.dataset_num_workers,
            collate_fn=collate_fn_filter_none,
            pin_memory=False,
            drop_last=False,
        )
        test_dataloaders.append(test_dataloader)
    return test_dataloaders


if __name__ == "__main__":
    config, save_dir = _parse_arguments()
    Logger.init(config.global_workdir, config.global_name, config.global_phase)
    Logger.enable_file()
    _set_random_seed(config.global_seed)
    logging.info(config)

    # Prepare dataset
    logging.info('Constructing dataset.')
    test_dataloaders = _get_databaunch(config)

    # Save each batch of images and labels
    logging.info('Saving ground truth images and labels.')
    for test_dataloader in test_dataloaders:
        for batch_idx, (images, labels) in enumerate(tqdm(test_dataloader, desc="Processing batches")):
            save_batch_images_and_labels(images, labels, batch_idx, save_dir=save_dir)

    logging.info("All ground truth images and labels have been saved.")
