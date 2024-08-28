import os
import pandas as pd
import numpy as np
import config
from sklearn.model_selection import train_test_split
from PIL import Image
from utils import realbox_to_yolobox, move_file



def generate_csv(filenames, csv_file_name):
    """Generate a csv file with 2 columns ['img','label'] contains image and label file name of given filenames.
    File will be save at DATASET_DIR
    
        Args:
            filenames (list): List of filename to write to csv file.
            csv_file_name (str): Name to save the csv file.
            
    """
    csv_data = {'img': [], 'label': []}
    for filename in filenames:
        img_name = filename
        label_name = filename[:-4] + '.txt'
        csv_data['img'].append(img_name)
        csv_data['label'].append(label_name)
    pd.DataFrame(csv_data).to_csv(f'{config.DATASET_DIR}/{csv_file_name}')

    
def get_box_from_row(row):
    image_name = row.iloc[0]
    image_path = os.path.join(config.TRAINING_IMAGES_DIR, image_name)
    image = Image.open(image_path)
    image_array = np.array(image)
    image_height, image_width, _ = image_array.shape
    box = [row.iloc[1], row.iloc[2], row.iloc[3], row.iloc[4]]
    box = realbox_to_yolobox(box, image_width, image_height)
    return image_name, box


if __name__ == '__main__':
    #  Create label directory and convert annotations
    print("Creating training labels ...")
    os.makedirs(config.TRAINING_LABELS_DIR, exist_ok=True)    
    annotation = pd.read_csv(config.CSV_FILEPATH)
    for i in range(len(annotation)):
        row = annotation.iloc[i]
        image_name, (x, y, w, h) = get_box_from_row(row)
        txt_filepath = os.path.join(config.TRAINING_LABELS_DIR, image_name[:-4]+'.txt')
        with open(txt_filepath, 'a') as f:           
            f.write(f"{x} {y} {w} {h}\n")
            f.close() 
        
            
    #  Split data to train/val/test sets
    random_state = 1
    train, val = train_test_split(config.ALL_FILENAMES, test_size=0.3, 
                                    random_state=random_state, shuffle=True) 

    print('Generate csv file ...')
    generate_csv(filenames=train, csv_file_name='train.csv')
    generate_csv(filenames=val, csv_file_name='val.csv')
    generate_csv(filenames=os.listdir(config.TESTING_IMAGES_DIR), csv_file_name='test.csv')

    # Move image and label files to corresponding train/val/test directories
    os.makedirs(config.VAL_IMAGES_DIR, exist_ok=True)
    os.makedirs(config.VAL_LABELS_DIR, exist_ok=True)
    move_file(val, config.TRAINING_IMAGES_DIR, config.VAL_IMAGES_DIR, 
            config.TRAINING_LABELS_DIR, config.VAL_LABELS_DIR)


    print('Done!!')
    