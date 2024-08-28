import torch
import torch.nn as nn
from torchsummary import summary
import config


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
            ConvBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2)
        ]
        
        for _ in range(2):
            layers += [
                ConvBlock(in_channels=1024, out_channels=512, kernel_size=1, padding=0, stride=1),
                ConvBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=1)
            ]
            
        layers += [
            ConvBlock(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=1),
            ConvBlock(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=2),
            ConvBlock(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=1),
            ConvBlock(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=1),
        ]

        return nn.Sequential(*layers)
    
    def _create_fcn_layers(self, num_grids, num_boxes):
        S, B = num_grids, num_boxes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=S*S*1024, out_features=496), # 4096 in paper
            # nn.Dropout(0.0),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(496, S*S*(5*B))
        )


if __name__ == '__main__':
    model = Yolov1(in_channels=3, num_grids=7, num_boxes=2)
    x = torch.rand(2, 3, 448, 448)
    out = model(x)
    print(out.shape)
    summary(model.to(config.DEVICE), (3, 448, 448))