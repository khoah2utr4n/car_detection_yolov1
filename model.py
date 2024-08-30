import torch.nn as nn
from torchsummary import summary


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
    
    def forward(self, x):
        return self.activation(self.batchnorm(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.darknet = self._create_darknet(in_channels)
        self.fcn = self._create_fcn_layers(**kwargs)
    
    def forward(self, x):
        logits = self.darknet(x)
        out = self.fcn(logits)
        return out
    
    def _create_darknet(self, in_channels):
        layers = [
            ConvBlock(in_channels=in_channels, out_channels=64, kernel_size=7, padding=3, stride=2),
            nn.MaxPool2d(kernel_size=2),
            
            ConvBlock(in_channels=64, out_channels=192, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2),
            
            ConvBlock(in_channels=192, out_channels=128, kernel_size=1, padding=0, stride=1),
            ConvBlock(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=1, padding=0, stride=1),
            ConvBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2)
        ]
        
        for _ in range(4):
            layers += [
                ConvBlock(in_channels=512, out_channels=256, kernel_size=1, padding=0, stride=1),
                ConvBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1)
            ]
        
        layers += [
            ConvBlock(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1),
            ConvBlock(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2)
        ]
        
        for _ in range(2):
            layers += [
                ConvBlock(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1),
                ConvBlock(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
            ]
            
        layers += [
            ConvBlock(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            ConvBlock(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=2),
            ConvBlock(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            ConvBlock(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
        ]

        return nn.Sequential(*layers)
    
    def _create_fcn_layers(self, num_grids, num_boxes, num_classes):
        S, B, C = num_grids, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=S*S*512, out_features=512), # 4096 in paper
            # nn.Dropout(0.0),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(512, S*S*(5*B + C))
        )