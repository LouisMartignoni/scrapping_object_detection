import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def imgprcs(file, label, IMG_SIZE=250):
    img = tf.io.read_file(file)
    img = tf.image.decode_jpeg(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img, label


def show(image, y_pred=None, y_true=None):
    plt.figure()
    plt.imshow(image)
    plt.title('prediction:'+y_pred+' ; true:'+y_true)
    plt.axis('off')
    plt.show()


TEST_DIR = 'data/validation'

base_model = tf.keras.models.load_model("models/mobilnetv2_freezed.h5")

labels = []
paths = []

train_subdirs = [subdir for subdir in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, subdir))]
for subdir in train_subdirs:
    subdir_fullpath = os.path.join(TEST_DIR, subdir)
    for filename in os.listdir(subdir_fullpath):
        paths.append(os.path.join(subdir_fullpath,filename))
        labels.append(subdir)

ds = tf.data.Dataset.from_tensor_slices((paths, labels))
ds = ds.map(imgprcs)
ds = ds.shuffle(buffer_size=300)
ds = ds.prefetch(1)

predictions = []
true_labels = []
label_encoder = {0:"carrot", 1:"poivron", 2:"tomato"}
for im, l in ds:
    #true_labels.extend(l.numpy())
    p = base_model.predict(tf.expand_dims(im, axis=0))
    pred = np.argmax(p, axis=1).item(0)
    y_pred = label_encoder[pred]
    y_true = l.numpy().decode("utf-8")
    if y_pred != y_true:
        show(im, y_pred, y_true)
    #predictions.extend(pred)

