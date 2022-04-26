import torch
from torch import nn
from typing import Tuple, List


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()


        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2), 
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            # Use ceil mode to get 75/2 to ouput 38#sol
            nn.Conv2d(
                in_channels=64,
                out_channels=output_channels[0],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),
        ) #sol

        self.additional_layers = nn.ModuleList([#sol
            nn.Sequential( # 19 x 19 out#sol
                nn.Conv2d(
                    in_channels=output_channels[0],
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=128,
                    out_channels=output_channels[1],
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.ReLU(),#sol
            ),#sol
            nn.Sequential( 
                nn.Conv2d(
                    in_channels=output_channels[1],
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=256,
                    out_channels=output_channels[2],
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.ReLU(),#sol
            ),#sol
            nn.Sequential( # 5 x 5 out#sol
                nn.Conv2d(
                    in_channels=output_channels[2],
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(),#sol
                nn.Conv2d(
                    in_channels=128,
                    out_channels=output_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.ReLU(),
            ),#sol
            nn.Sequential( 
                nn.Conv2d(
                    in_channels=output_channels[3],
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=128,
                    out_channels=output_channels[4],
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.ReLU(),
                
            ),
            
            nn.Sequential(
                nn.Conv2d(
                    in_channels=output_channels[4],
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=128,
                    out_channels=output_channels[5],
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.ReLU(),
            ),
        ])

        

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """

        out_features = []
        x = self.feature_extractor(x)
        out_features.append(x)
        for additional_layer in self.additional_layers.children():
            x = additional_layer(x)
            out_features.append(x)
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            print("The expected shape: "+ str(expected_shape)+" got "+str(feature.shape[1:]))
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

