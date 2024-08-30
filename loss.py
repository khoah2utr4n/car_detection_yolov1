import torch
import torch.nn as nn
from utils import compute_IOU


class YoloLoss(nn.Module):
    def __init__(self, num_grids, num_boxes, num_classes):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = num_grids
        self.B = num_boxes
        self.C = num_classes
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        
    def forward(self, predictions, target):
        # x,y,w,h,conf
        predictions = predictions.reshape(-1, self.S, self.S, 5*self.B + self.C)
        iou_b1 = compute_IOU(predictions[..., 0:4], target[..., 0:4])
        iou_b2 = compute_IOU(predictions[..., 5:9], target[..., 0:4])
        ious = torch.cat((iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)), dim=0)
        iou_maxes, bestbox_idx = torch.max(ious, dim=0)
        exists_box = target[..., 4].unsqueeze(-1)
        
        # box responsible for prediction
        box_predictions = exists_box * (
            (1-bestbox_idx) * predictions[..., 0:5] + 
            bestbox_idx * predictions[..., 5:10]
        )

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            abs(box_predictions[..., 2:4] + 1e-6) # add 1e-6 to avoid sqrt = 0 (derivate: -inf)
        )

        box_targets = exists_box * target[..., 0:5]
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        
        # Box coords loss
        box_loss = self.mse(
            box_predictions[..., 0:4], box_targets[..., 0:4]
        )

        # Object loss
        object_loss = self.mse(
            box_predictions[..., 4:5], box_targets[..., 4:5]
        )
        
        # No object loss
        no_object_loss = self.mse(
            (1 - exists_box) * predictions[..., 4:5], 
            (1 - exists_box) * box_targets[..., 4:5]
        )
        no_object_loss += self.mse(
            (1 - exists_box) * predictions[..., 9:10], 
            (1 - exists_box) * box_targets[..., 4:5]
        )
        
        class_loss = self.mse(
            exists_box * predictions[..., (5*self.B): (5*self.B + self.C)],
            exists_box * target[..., 5: 5 + self.C]
        )
        
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
        return loss