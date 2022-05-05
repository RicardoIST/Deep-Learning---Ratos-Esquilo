from os import remove
from ssd.modeling import AnchorBoxes
from ssd.modeling.retinanet import RetinaNet
from tops.config import LazyCall as L
from ssd.modeling.backbones.fpn import FPN
from ssd.data.transforms import (
    ToTensor, RandomHorizontalFlip, RandomSampleCrop, Normalize, Resize,
    GroundTruthBoxesToAnchors)
import torch
import torchvision
import getpass
import pathlib
from configs.utils import get_dataset_dir, get_output_dir
from ssd.data import MNISTDetectionDataset
from torch.optim.lr_scheduler import MultiStepLR, LinearLR
from ssd.modeling import SSD300
from ssd.modeling.FocalLoss import FocalLoss
##


# The line belows inherits the configuration set for the tdt4265 dataset
from .task2_3_2 import(
    train,
    optimizer,
    schedulers,
    loss_objective,
   # model,
    backbone,
    data_train,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map,
    anchors
)

# The config below is copied from the ssd300.py model trained on images of size 300*300.
# The images in the tdt4265 dataset are of size 128 * 1024, so resizing to 300*300 is probably a bad idea
# Change the imshape to (128, 1024) and experiment with better prior boxes

model = L(RetinaNet)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=10 + 1 , # Add 1 for background
    boolea=True #weight initialization
)