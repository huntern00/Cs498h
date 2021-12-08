# -*- coding: utf-8 -*-
"""Copy of Final Project

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HdEjM2PeSp07SWPm5Y2w2MD5T06gcDcV
"""

#imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
import random

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

#labels for reference
#0- Shirt
#1- Pants
#2- Hoodie
#3- Dress
#4- Jacket
#5- Sandal
#6- Shirt
#7- Sneaker
#8- Bag
#9- Boots

#data processing
train_images = train_images / 255.0
test_images = test_images / 255.0

#shows first 10 images
plt.figure(figsize=(15, 15))
for i in range (10):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i], cmap=plt.cm.gray)
  plt.xlabel(train_labels[i])
plt.show()

#define model
model = keras.Sequential([
  keras.layers.InputLayer(input_shape=(28, 28)),
  keras.layers.Reshape(target_shape=(28, 28, 1)),
  keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu),
  keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.Dropout(0.25),
  keras.layers.Flatten(),
  keras.layers.Dense(10)
])

#compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#training
model.fit(train_images, train_labels, epochs=5) #20

#testing
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test Accuracy:', test_acc)

def get_label_color(val1, val2):
  if val1 == val2:
    return 'black'
  else:
    return 'red'

predictions = model.predict(test_images)
prediction_digits = np.argmax(predictions, axis=1)

plt.figure(figsize=(18, 18))
for i in range(100):
  ax = plt.subplot(10, 10, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  image_index = random.randint(0, len(prediction_digits))
  plt.imshow(test_images[image_index], cmap=plt.cm.gray)
  ax.xaxis.label.set_color(get_label_color(prediction_digits[image_index],\
                                           test_labels[image_index]))
  plt.xlabel('Predicted: %d' % prediction_digits[image_index])
plt.show()