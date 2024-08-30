# Re-implement YOLOv1 model for car detection
In this project, I re-implement a YOLOv1 ("you only look once") model for Car Object Detection task. 
 * 
 * The [Car Detection Dataset](https://www.kaggle.com/datasets/sshikamaru/car-object-detection) on Kaggle.

**Image**

## Setup
### 1. Create a virtual environment 
  ```
  conda create --name myenv python==3.11.2
  conda activate myenv
  ```
### 2. Clone this repository and install packages
  * Clone this repository:
  ```
  git clone https://github.com/khoah2utr4n/face_mask_detection_yolo.git
  ```
  * Install [PyTorch GPU/CPU](https://pytorch.org/get-started/locally/).
  * Install packages
  ```
  pip install -r requirements.txt
  ```
### 3. Dataset
  * Download [Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection), uncompress it and put `images` folder and `annotations` folder in the `datasets` folder.
  * Prepocess the dataset:
  ```
  python preprocessing_data.py
  ```

## Usage
### 1. Training
  * The following command loads a pre-trained YOLOv8n model and trains it for 50 epochs:
  ```
  python train.py --epochs 50 --weights_path yolov8n.pt
  ```
  * To resume interrupted training, use:
  ```
  python train.py --weights_path <path/to/last.pt> --resume True
  ```
When finish the training, you will get the best weights of model through training `best.pt`

### 2. Detection
  * Detect masks in images or real-time camera using a Streamlit UI:
  ```
  streamlit run UI.py
  ```
  * **Upload Model Weights**
    * To start, please upload the weights file of the model (file `.pt`) by select the checkbox `Upload new weights`.
    * You can use your own weights or [download pre-trained weights](https://drive.google.com/file/d/1EjRwsiWIiLy60cKZLk_zM7c1SfrbueES/view?usp=sharing).
  
  * **Detection method**: Choose your preferred detection method:
    * **Image upload**: Toggle "Using real-time camera" off to upload a picture and detect masks in it.
    * **Real-time camera**: Toggle "Using real-time camera" on to activate mask detection from your webcam.

 
