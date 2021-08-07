import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from training_functions import img_train_test_split, get_run_logdir
import scipy
import os

#img_train_test_split('data/source_data', 0.7)

################################# Pipeline ######################################
TRAIN_DIR = 'data/train'
batch_size = 32


train_generator = ImageDataGenerator(rescale=1. / 255,
                                     rotation_range=40
                                     ,horizontal_flip=True,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     fill_mode="nearest",
                                     width_shift_range=0.2,
                                     height_shift_range=0.2
                                     )


train_datagen = train_generator.flow_from_directory(
    TRAIN_DIR,
    target_size=(250, 250),
    batch_size=batch_size,
    class_mode='categorical')

TEST_DIR = 'data/validation'
test_generator = ImageDataGenerator(rescale=1. / 255)

test_datagen = test_generator.flow_from_directory(
    TEST_DIR,
    target_size=(250, 250),
    batch_size=batch_size,
    class_mode='categorical')

n_classes = train_datagen.num_classes
n_train_im = train_datagen.samples
EPOCHS = 200

############################### Models #########################################
# input = tf.keras.layers.Input(shape=(250, 250, 3), name='input')
# x = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', kernel_initializer='he_normal')(input)
# x = tf.keras.layers.MaxPool2D((2, 2))(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_normal')(input)
# x = tf.keras.layers.MaxPool2D((2, 2))(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_normal')(input)
# x = tf.keras.layers.MaxPool2D((2, 2))(x)
# x = tf.keras.layers.Flatten()(x)
# x = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x)
# out = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
#
# base_model = tf.keras.Model(inputs=[input], outputs=[out])

#################### Transfer learning ########################
#pretrained_model = InceptionResNetV2(include_top=False, weights='imagenet', pooling='avg')
pretrained_model = tf.keras.applications.MobileNetV2(input_shape=(250, 250, 3), include_top=False, weights='imagenet', pooling='avg')
pretrained_model.summary()
for layer in pretrained_model.layers[:-30]:
    layer.trainable = False

#last_layer = pretrained_model.get_layer('global_average_pooling2d')
last_output = pretrained_model.output
x = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal')(last_output)
x = tf.keras.layers.Dropout(0.2)(x)
out = tf.keras.layers.Dense(n_classes, activation='softmax', kernel_initializer='he_normal')(x)
base_model = tf.keras.Model(inputs=pretrained_model.inputs, outputs=out)

base_model.compile(optimizer=tf.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics='accuracy')

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
# Creation log_dir
root_logdir = os.path.join(os.curdir, 'tensorboard_logs')
run_logdir = get_run_logdir(root_logdir)
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

history = base_model.fit(train_datagen,
                         epochs=200,
                         validation_data=test_datagen,
                         callbacks=[early_stop, tensorboard_cb],
                         verbose=2)

#base_model.save("models/mobilnetv2_unfreezed_30.h5")
# 0.8845
# 0.9688 inception resnet 100
# 0.98 mobilnetv2_freezed
# 0.99 mobilnetv2_unfreezed_30