import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0,DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import classification_report,confusion_matrix
import ipywidgets as widgets
import io
from PIL import Image
from IPython.display import display,clear_output
from warnings import filterwarnings
for dirname, _, filenames in os.walk('C:\\DATA SET'):

    for filename in filenames:
        print(os.path.join(dirname, filename))

colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
colors_red = ["#331313", "#582626", '#9E1717', '#D35151', '#E9B4B4']
colors_green = ['#01411C','#4B6F44','#4F7942','#74C365','#D0F0C0']

sns.palplot(colors_dark)
sns.palplot(colors_green)
sns.palplot(colors_red)

labels = ['glioma','meningioma','notumor','pituitary']
X_train = []
y_train = []
image_size = 150
for i in labels:
    folderPath = os.path.join('C:\\DATA SET\\Training',i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size, image_size))
        X_train.append(img)
        y_train.append(i)
       
for i in labels:
    folderPath = os.path.join('C:\\DATA SET\\Testing',i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        y_train.append(i)

testdir = 'C:\\DATA SET\\Testing'


X_train = np.array(X_train)
y_train = np.array(y_train)

k=0
fig, ax = plt.subplots(1,4,figsize=(20,20))
fig.text(s='Sample Image From Each Label',size=18,fontweight='bold',
             fontname='monospace',color=colors_dark[1],y=0.62,x=0.4,alpha=0.8)
for i in labels:
    j=0
    while True :
        if y_train[j]==i:
            ax[k].imshow(X_train[j])
            ax[k].set_title(y_train[j])
            ax[k].axis('off')
            k+=1
            break
        j+=1

X_train, y_train = shuffle(X_train,y_train, random_state=101)

X_train.shape

X_train,X_test,y_train,y_test = train_test_split(X_train,y_train, test_size=0.1,random_state=101)

y_train_new = []
for i in y_train:
    y_train_new.append(labels.index(i))
y_train = y_train_new
y_train = tf.keras.utils.to_categorical(y_train)


efficientnet = EfficientNetB0(weights=None,include_top=False,input_shape=(image_size,image_size,3))
densenet = DenseNet121(weights=None, include_top=False, input_shape=(image_size, image_size, 3))
# Add classification head for EfficientNetB0
x_eff = efficientnet.output
x_eff = tf.keras.layers.GlobalAveragePooling2D()(x_eff)
x_eff = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x_eff)
x_eff = Dropout(0.5)(x_eff)

# Add classification head for DenseNet
x_dense = densenet.output
x_dense = tf.keras.layers.GlobalAveragePooling2D()(x_dense)
x_dense = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x_dense)
x_dense = Dropout(0.5)(x_dense)

# Concatenate the outputs of EfficientNetB0 and DenseNet
x = Concatenate()([x_eff, x_dense])
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(4, activation='softmax')(x)  


# Create the final model
model = tf.keras.models.Model(inputs=[efficientnet.input, densenet.input], outputs=output)

model.summary()


tensorboard = TensorBoard(log_dir = 'logs')
checkpoint = ModelCheckpoint("model.h5",monitor="val_accuracy",save_best_only=True,mode="auto",verbose=1)
# Define a custom learning rate scheduler
def lr_schedule(epoch):
    lr = 0.001  
    if epoch >= 10:
        lr *= 0.5  
    return lr

# Define the LearningRateScheduler callback
learning_rate_scheduler = LearningRateScheduler(lr_schedule)

# Define the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',   
    patience=10,          
    verbose=1,            
    restore_best_weights=True  
)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Train the model with EarlyStopping and the Learning Rate Scheduler
history = model.fit(
    [X_train, X_train],  
    y_train,             
    validation_split=0.1,
    epochs=2,
    verbose=1,
    batch_size=32,
    callbacks=[tensorboard, checkpoint, learning_rate_scheduler, early_stopping]  
)


model.save("brain_tumor_model.h5")

X_train = np.save('X_train.npy',X_train)
X_test = np.save('X_test.npy',X_test)
y_test = np.save('y_test.npy',y_test)

filterwarnings('ignore')

# Extract training history
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
epochs = range(1, len(train_acc) + 1)

# Create subplots for accuracy and loss
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Plot training and validation accuracy
axes[0].plot(epochs, train_acc, marker='o', label='Training Accuracy')
axes[0].plot(epochs, val_acc, marker='o', label='Validation Accuracy')
axes[0].set_title('Training and Validation Accuracy')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Plot training and validation loss
axes[1].plot(epochs, train_loss, marker='o', label='Training Loss')
axes[1].plot(epochs, val_loss, marker='o', label='Validation Loss')
axes[1].set_title('Training and Validation Loss')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.show()

test_loss, test_accuracy = model.evaluate([X_test, X_test], y_test, verbose=0)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

predictions = model.predict([X_test, X_test], batch_size=32)
predicted_classes = np.argmax(predictions, axis=1)


# Generate a classification report
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print("Classification report:\n")
print(report)
