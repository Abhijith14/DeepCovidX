import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# define the path to the dataset directory
DATASET_PATH = 'CS552J_DMDL_Assessment_1_Dataset'

# define the size of the images
IMG_SIZE = (224, 224)

# define the number of channels
CHANNELS = 3

# define the label categories
CATEGORIES = ['Covid-19', 'Normal']

# initialize the data and labels arrays
data = []
labels = []

# loop over the image paths
for category in CATEGORIES:
    path = os.path.join(DATASET_PATH, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)
        image = cv2.resize(image, IMG_SIZE)
        image = np.array(image, dtype=np.float32)
        image /= 255.0
        data.append(image)
        labels.append(category)


print("Total number of images in the dataset: ", len(data))
print("Total number of labels in the dataset: ", len(labels))
