import os
import torch
import shutil
import config
from collections import Counter


def realbox_to_yolobox(bndbox, width, height):
    """Convert real bounding box to YOLO bounding box.
    
        Args:
            bndbox (list | np.darray): A xml bounding box with format [xmin, ymin, xmax, ymax]
            width (int): A width of entire image
            height (int): A height of entire image
        Return:
            yolo_bndbox (list): The bounding box in YOLO format [x_center, y_center, bnd_width, bndbox_height]
    """
    x_center = ((bndbox[0] + bndbox[2]) / 2.) / width
    y_center = ((bndbox[1] + bndbox[3]) / 2.) / height
    bnd_width = (bndbox[2] - bndbox[0]) / width
    bnd_height = (bndbox[3] - bndbox[1]) / height
    yolo_bndbox = [x_center, y_center, bnd_width, bnd_height]
    return yolo_bndbox


def yolobox_to_realbox(bndbox, width, height):
    """Convert YOLO bounding box to xml bounding box.
    
        Args:
            bndbox (list | np.darray): A YOLO bounding box with format [x_center, y_center, bnd_width, bndbox_height]
            width (int): A width of entire image
            height (int): A height of entire image
        Return:
            xml_bndbox (list): The bounding box in xml format [xmin, ymin, xmax, ymax]
    """
    xmin = (bndbox[0] - bndbox[2] / 2.) * width
    ymin = (bndbox[1] - bndbox[3] / 2.) * height
    xmax = (bndbox[0] + bndbox[2] / 2.) * width
    ymax = (bndbox[1] + bndbox[3] / 2.) * height
    xml_bndbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
    return xml_bndbox


def move_file(filenames, images_dir, images_dest_dir, labels_dir, labels_dest_dir):
    os.makedirs(images_dest_dir, exist_ok=True)    
    os.makedirs(labels_dest_dir, exist_ok=True)

    for filename in filenames:
        image_filename = filename[:-4] + '.jpg'
        label_filename = filename[:-4] + '.txt'
        img_src = os.path.join(images_dir, image_filename)
        label_src = os.path.join(labels_dir, label_filename)
        if os.path.exists(img_src) and not os.path.exists(f'{images_dest_dir}/{image_filename}'):
            shutil.move(img_src, images_dest_dir)
        if os.path.exists(label_src) and not os.path.exists(f'{labels_dest_dir}/{label_filename}'):
            shutil.move(label_src, labels_dest_dir)

def compute_IOU(predicted_boxes, true_boxes):
    # x, y, w, h
    box1_x1 = predicted_boxes[..., 0:1] - predicted_boxes[..., 2:3] / 2
    box1_y1 = predicted_boxes[..., 1:2] - predicted_boxes[..., 3:4] / 2
    box1_x2 = predicted_boxes[..., 0:1] + predicted_boxes[..., 2:3] / 2
    box1_y2 = predicted_boxes[..., 1:2] + predicted_boxes[..., 3:4] / 2
    box2_x1 = true_boxes[..., 0:1] - true_boxes[..., 2:3] / 2
    box2_y1 = true_boxes[..., 1:2] - true_boxes[..., 3:4] / 2
    box2_x2 = true_boxes[..., 0:1] + true_boxes[..., 2:3] / 2
    box2_y2 = true_boxes[..., 1:2] + true_boxes[..., 3:4] / 2
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union = box1_area + box2_area - intersection
    return intersection / union


def compute_mAP(predicted_boxes, true_boxes, iou_threshold):
    """
    train_idx, x, y, w, h, conf, class
    """
    epsilon = 1e-6
    average_precisions = []
    detections = [box for box in predicted_boxes]
    ground_truths = [box for box in true_boxes]

    # sort by box probabilities which is index 5
    detections.sort(key=lambda x: x[5], reverse=True)
    TP = torch.zeros(len(detections))
    FP = torch.zeros(len(detections))
    total_true_boxes = len(ground_truths)
    if total_true_boxes > 0:
        amount_boxes = Counter([gt[0] for gt in ground_truths])
        for key, value in amount_boxes.items():
            amount_boxes[key] = torch.zeros(value)

        for detection_idx, detection in enumerate(detections):

            best_iou = 0
            for gt_idx, gt in enumerate(ground_truths):
                iou = compute_IOU(torch.tensor(detection), torch.tensor(gt))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gt_idx

            if best_iou > iou_threshold:
                if amount_boxes[detection[0], best_idx] == 0:
                    TP[detection_idx] = 1
                    amount_boxes[detection[0], best_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)
            precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
            recalls = TP_cumsum / (total_true_boxes + epsilon) 
            precisions = torch.cat((torch.tensor([1]), precisions))
            recalls = torch.cat((torch.tensor([0]), recalls))
            # torch.trapz for numerical integration
            average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / (len(average_precisions) + epsilon)
    

def do_NMS(boxes, threshold, iou_threshold):
    """
    x,y,w,h,conf
    """
    boxes = [box for box in boxes if box[4] > threshold]
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    boxes_after_nms = []
    while boxes:
        chosen_box = boxes.pop(0)
        boxes = [
            box for box in boxes
            if compute_IOU(
                torch.tensor(box[0:4]), 
                torch.tensor(chosen_box[0:4])
            ) < iou_threshold
        ]
        boxes_after_nms.append(chosen_box)
    return boxes_after_nms


def get_bestbox(boxes, B, S=7):
    """
    x,y,w,h,conf
    (N, S*S, 5B) -> (N, S*S, 5)
    """
    boxes = boxes.reshape(-1, S, S, 5*B)
    box1 = boxes[..., 0:5]
    box2 = boxes[..., 5:10]
    confs = torch.cat((box1[..., 4:5].unsqueeze(0), box2[..., 4:5].unsqueeze(0)), dim=0)
    best_conf, bestbox_idx = torch.max(confs, dim=0)
    bestbox = (1-bestbox_idx) * box1 + bestbox_idx * box2
    cell_indices = torch.arange(7).repeat(boxes.shape[0], 7, 1).unsqueeze(-1).to(config.DEVICE)
    bestbox[..., :1] = 1/S * (bestbox[..., :1] + cell_indices)
    bestbox[..., 1:2] = 1/S * (bestbox[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    return bestbox


def get_list_of_boxes(boxes, S=7):
    batch_size = boxes.shape[0] 
    boxes = get_bestbox(boxes, B=config.NUM_BOXES).reshape(batch_size, S*S, -1)
    boxes_list = []
    for ex_idx in range(batch_size):
        bboxes = []
        for cell_idx in range(S*S):
            bboxes.append([x.item() for x in boxes[ex_idx, cell_idx, :]])
        boxes_list.append(bboxes)
    return boxes_list


def get_boxes(train_loader, model, threshold, iou_threshold):
    model.eval()
    box_predictions = []
    box_targets = []
    train_idx = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        with torch.no_grad():
            pred_boxes = model(x)
        batch_size = x.shape[0]
        pred_boxes = get_list_of_boxes(pred_boxes, S=config.NUM_GRIDS)
        true_boxes = get_list_of_boxes(y, S=config.NUM_GRIDS)

        for i in range(batch_size):
            nms_boxes = do_NMS(pred_boxes[i], threshold=threshold, iou_threshold=iou_threshold)
            for box in nms_boxes:
                box_predictions.append([train_idx] + box)
            
            for box in true_boxes[i]:
                if box[4] > 0: # box exists
                    box_targets.append([train_idx] + box)
            train_idx += 1
    model.train()
    return box_predictions, box_targets


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])