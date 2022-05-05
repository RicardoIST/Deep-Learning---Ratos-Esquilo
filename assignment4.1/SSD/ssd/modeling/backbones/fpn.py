import torch
from typing import OrderedDict,Tuple, List
from torch import nn
import torchvision

class FPN(torch.nn.Module):
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        self.model=torchvision.models.resnet34(pretrained=True)
        self.fpn= torchvision.ops.FeaturePyramidNetwork(in_channels_list=[64, 128, 256, 512, 256, 64], out_channels=256)
        
        #2 More Features
        self.layer5 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU()
        )
        

        self.layer6= nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU()
        )
        
        
        
        
    def forward(self, x):

        out_features_res = {}
        
        x =self.model.conv1(x)
        x =self.model.bn1(x)
        x =self.model.relu(x)
        x =self.model.maxpool(x)
        
        x =self.model.layer1(x)
        out_features_res['map1'] =x
        
        x =self.model.layer2(x)
        out_features_res['map2'] =x
        
        x =self.model.layer3(x)
        out_features_res['map3'] =x
        
        x =self.model.layer4(x)
        out_features_res['map4'] =x
        
        x =self.layer5(x) #extra layer
        out_features_res['map5'] =x
        
        x =self.layer6(x) #extra layer
        out_features_res['map6'] =x
        
        #FPN
        out_features_fpn = self.fpn(out_features_res)
        
        out_features = []
        for feat in out_features_fpn.values():
            out_features.append(feat)
            
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)
            
            
        
        