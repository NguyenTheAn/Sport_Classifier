# -*- coding: utf-8 -*-
"""Test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13UZ1DVjIFBQziMP-3IUauO2MW1tq6dxy
"""

!wget -O Data.zip https://www.dropbox.com/s/v7o09o6b2mkv9xf/Data.zip?dl=0
!unzip Data.zip

import matplotlib
matplotlib.use("Agg")

import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import AveragePooling2D, Dense, Dropout, Flatten, Input
from keras.models import Model, Sequential
from keras.applications import ResNet50
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from imutils import paths
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2

from tqdm import tqdm
data_dir = os.path.join(os.getcwd(), "data")
LABELS = set(["weight_lifting", "tennis", "football"])
 
imagePaths = list(paths.list_images(data_dir))
data = []
Labels = []
errors = 0

for imagePath in tqdm(imagePaths):
  label = imagePath.split(os.path.sep)[-2]

#   if label not in LABELS:
#     continue
  if os.path.isfile(imagePath) == True:
    try:
      image = cv2.imread(imagePath)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image = cv2.resize(image, (224, 224))
      data.append(image)
      Labels.append(label)
    except:
      errors = errors+1

Labels = pd.get_dummies(Labels)
data = np.array(data)
X_train, X_test, Y_train, Y_test = train_test_split(data, Labels, test_size = 0.25, random_state = 12)

trainAug = ImageDataGenerator(
    rotation_range = 30, 
    zoom_range = 0.15,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.15,
    horizontal_flip = True,
    fill_mode = "nearest")

valAug = ImageDataGenerator()

mean = np.array([123.68, 116.779, 103.939], dtype = "float32") # ImageNet mean

trainAug.mean = mean
valAug.mean = mean

# ResNet50 without output layer
baseModel = ResNet50(include_top = False, weights = "imagenet", input_tensor = Input(shape = (224, 224, 3)))

# tao headmodel thay the cho output layer cua ResNet50
headModel = baseModel.output
headModel = AveragePooling2D(pool_size = (7, 7))(headModel)
headModel = Flatten()(headModel)
headModel = Dense(512, activation = 'relu')(headModel)
headModel = Dropout(0.4)(headModel)
headModel = Dense(Labels.shape[1], activation = "softmax")(headModel)

# dat headModel len dinh cua baseModel
model = Model(inputs = baseModel.input, outputs = headModel)

#freeze the baseModel
for layer in baseModel.layers:
    layer.trainable = False

opt = SGD(lr = 1e-3, momentum = 0.9, decay = 1e-3/100)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ['accuracy'])

filepath="model1.h5"
checkpoint = ModelCheckpoint(filepath, 
                             monitor = 'val_acc', 
                             verbose = 1, 
                             save_best_only = True, 
                             mode = 'max')

callbacks_list = [checkpoint]
H = model.fit_generator(
	trainAug.flow(X_train, Y_train, batch_size=128),
	steps_per_epoch=len(X_train) // 128,
	validation_data=valAug.flow(X_test, Y_test),
	validation_steps=len(X_test) // 128,
	epochs=20,
  callbacks = callbacks_list)

N = 20
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot1.png")

from google.colab import files
files.download('model1.h5')
files.download("plot1.png")

!wget -O model1.h5 https://www.dropbox.com/s/kk9t2z2zcmmxtvv/model1.h5?dl=0
model= load_model("model1.h5")

for layer in model.layers[65:]:
#   print(layer.trainable)
    layer.trainable=True

opt = SGD(lr = 1e-3, momentum = 0.9, decay = 1e-3/100)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ['accuracy'])

filepath="model2.h5"
checkpoint = ModelCheckpoint(filepath, 
                             monitor = 'val_acc', 
                             verbose = 1, 
                             save_best_only = True, 
                             mode = 'max')

callbacks_list = [checkpoint]
H = model.fit_generator(
	trainAug.flow(X_train, Y_train, batch_size=128),
	steps_per_epoch=len(X_train) // 128,
	validation_data=valAug.flow(X_test, Y_test),
	validation_steps=len(X_test) // 128,
	epochs=30,
  callbacks = callbacks_list)

N = 30
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot2.png")

from google.colab import files
files.download('model2.h5')
files.download("plot2.png")