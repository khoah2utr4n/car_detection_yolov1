# Re-Implementation YOLOv1 model using Pytorch for Car Object Detection
This repository provides a Pytorch implementation of the YOLOv1 (You Only Look Once) object detection model for Car Object Detection task. Building upon the [YOLOv1 paper](https://ieeexplore.ieee.org/document/7780460/) and this [aladdinpersson repository](https://github.com/aladdinpersson/Machine-Learning-Collection). The dataset used in this repository is the [Car Detection Dataset](https://www.kaggle.com/datasets/sshikamaru/car-object-detection) on Kaggle.

![example](https://github.com/user-attachments/assets/db850f1f-e25b-46bc-a763-473fda753042)


## Setup
### 1. Create a virtual environment 
  ```
  conda create --name myenv python==3.11.2
  conda activate myenv
  ```
### 2. Clone this repository and install packages
  * Clone this repository:
  ```
  git clone https://github.com/khoah2utr4n/car_detection_yolov1.git
  ```
  * Install [PyTorch GPU/CPU](https://pytorch.org/get-started/locally/).
  * Install packages
  ```
  pip install -r requirements.txt
  ```
### 3. Dataset
  * Download the [Car Detection Dataset](https://www.kaggle.com/datasets/sshikamaru/car-object-detection) and uncompress it to get the `data` folder.
  * Preprocess the dataset:
  ```
  python preprocessing_data.py
  ```
## Usage
 * Configure Hyperparameters: Modify the `config.py` file to adjust training parameters like learning rate, batch size, and epochs.
 * The `notebook.ipynb` notebook file shows how to use the model for training and making predictions. It also provides the code for the visualization training process and the model's predictions.
