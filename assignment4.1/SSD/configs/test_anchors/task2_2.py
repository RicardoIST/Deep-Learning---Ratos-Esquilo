
# Inherit configs from the default ssd300
import torchvision
from ssd.data import TDT4265Dataset
from tops.config import LazyCall as L
from ssd.data.transforms import (
    ToTensor, Normalize, Resize,
    GroundTruthBoxesToAnchors, RandomSampleCrop, RandomHorizontalFlip)
from ..ssd300 import backbones
from ..ssd300 import train, anchors, optimizer, schedulers, backbone, model, data_train, data_val, loss_objective
from ..utils import get_dataset_dir
from ssd.modeling import AnchorBoxes

# The line belows inherits the configuration set for the tdt4265 dataset
from ..tdt4265 import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    backbone,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map
)

# The config below is copied from the ssd300.py model trained on images of size 300*300.
# The images in the tdt4265 dataset are of size 128 * 1024, so resizing to 300*300 is probably a bad idea
# Change the imshape to (128, 1024) and experiment with better prior boxes
train.imshape = (128, 1024)
train.epoch=40
model.num_classes = 8 + 1  # Add 1 for background class


anchors = L(AnchorBoxes)(
    feature_sizes= [[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    strides= [[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes=[[16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [128, 400]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    # aspect ratio is defined per feature map (first index is largest feature map (38x38))
    # aspect ratio is used to define two boxes per element in the list.
    # if ratio=[2], boxes will be created with ratio 1:2 and 2:1
    # Number of boxes per location is in total 2 + 2 per aspect ratio
    aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)


train_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(RandomSampleCrop)(),
    L(ToTensor)(),
    L(RandomHorizontalFlip)(),
    L(Resize)(imshape="${train.imshape}"),
    L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),
])

val_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(ToTensor)(),
    L(Resize)(imshape="${train.imshape}"),
])

data_train.dataset = L(TDT4265Dataset)(
    img_folder=get_dataset_dir("tdt4265_2022"),
    transform="${train_cpu_transform}",
    annotation_file=get_dataset_dir("tdt4265_2022/train_annotations.json"))


data_val.dataset = L(TDT4265Dataset)(
    img_folder=get_dataset_dir("tdt4265_2022"),
    transform="${val_cpu_transform}",
    annotation_file=get_dataset_dir("tdt4265_2022/val_annotations.json"))
