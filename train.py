import cv2
import numpy as np
import config
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import compute_mAP, get_boxes, save_checkpoint


def train_one_epoch(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True) # leave=True to update the same line instead printing new line
    mean_loss = []
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update the progress bar 
        loop.set_postfix({'loss': loss})
    
    mean_loss = sum(mean_loss) / len(mean_loss)
    print(f"Mean loss was {mean_loss}")
    return mean_loss


def train_fn(model, train_loader, val_loader, optimizer, loss_fn, cal_mAP_freq=20):
    list_best_mAP = []
    mean_losses = []
    
    best_mAP = 0.0
    for epoch in range(config.EPOCHS):
        print('Start epoch: ', epoch)
        mean_loss = train_one_epoch(train_loader, model, optimizer, loss_fn)

        
        if (epoch+1) % cal_mAP_freq == 0:
            print('Get boxes....')
            predicted_boxes, target_boxes = get_boxes(
                val_loader, model, threshold=0.5, iou_threshold=0.5
            )
            print('Compute mAP....')
            mean_average_prec = compute_mAP(predicted_boxes, target_boxes, iou_threshold=0.5)
            print(f"Val mAP: {mean_average_prec}")

            if mean_average_prec > best_mAP:
                print('Save model')
                best_mAP = mean_average_prec.item()
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }
                save_checkpoint(checkpoint, filename="checkpoint.pth.tar")
            
        mean_losses.append(mean_loss)
        list_best_mAP.append(best_mAP)
    print('Done!!!, best_mAP: ', best_mAP)
    return mean_losses, list_best_mAP