import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Concatenate, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0,DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report,confusion_matrix
import ipywidgets as widgets
import io
from PIL import Image
from IPython.display import display,clear_output
from warnings import filterwarnings
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("brain_tumor_model.h5")

# Define class labels
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Define image size
image_size = 150

# Load and preprocess the image for prediction
image_path = "C:\\DATA SET\\Testing\\notumor\\Te-no_0066.jpg"
new_img = Image.open(image_path)
new_img = new_img.resize((image_size, image_size))  # Resize the PIL Image
img = np.array(new_img)  # Convert PIL Image to NumPy array

# Make the prediction
print("Following is our prediction:")
prediction = model.predict([np.expand_dims(img, axis=0), np.expand_dims(img, axis=0)])

# Get the class name
class_index = np.argmax(prediction)
class_name = class_labels[class_index]

# Plot the image with the predicted class name
plt.figure(figsize=(4, 4))
plt.imshow(new_img)
plt.axis('off')
plt.title(class_name)
plt.show()


