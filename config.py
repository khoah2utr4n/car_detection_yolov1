import os
import torch


DATASET_DIR = 'data'
TRAINING_IMAGES_DIR = 'data/training_images'
VAL_IMAGES_DIR = 'data/validation_images'
TESTING_IMAGES_DIR = 'data/testing_images'

TRAINING_LABELS_DIR = 'data/training_labels'
VAL_LABELS_DIR = 'data/validation_labels'

CSV_FILEPATH = 'data/train_solution_bounding_boxes (1).csv'
ALL_FILENAMES = os.listdir(TRAINING_IMAGES_DIR)

NUM_GRIDS = 7
NUM_BOXES = 2
CLASS_INDEXS = {'car': 0}
CLASS_NAMES = {0: 'car'}
NUM_CLASSES = len(CLASS_NAMES)

PIN_MEMORY = True
BATCH_SIZE = 8
EPOCHS = 200
LR = 1e-4
WEIGHT_DECAY = 1e-6
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'