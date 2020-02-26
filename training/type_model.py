# -*- coding: utf-8 -*-
import time, os, json
import tensorflow as tf
from shutil import copyfile
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D, Add

DIR_DATASET = "/home/michuxd/MachineLearning/ZaraTech/FashionDataset"
PREPARE_DATASET = False

IMG_HEIGHT = 56
IMG_WIDTH = 56
TARGET_SIZE = (IMG_WIDTH, IMG_HEIGHT)
BATCH_SIZE = 512

WEIGHTS_PATH = "model.h5"

IMG_PER_EPOCHS = 4096
IMG_PER_TEST = 1024
NUM_EPOCHS = 20

NUM_CLASSES = 13

def check_or_create(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)

def prepare_validation_dataset(dir_dataset):
    new_dir = f"{dir_dataset}/validation/annos"
    for subdir, dirs, files in os.walk(new_dir):
        for f in files:
            filepath = subdir + os.sep + f
            if filepath.endswith(".json"):
                data = json.load(open(filepath))
                for key in data.keys():
                    if(key.startswith('item')):
                        newpath = f"{dir_dataset}/validation/type/{data[key]['category_name']}"
                        name = f[:-5]
                        check_or_create(newpath)
                        copyfile(f"{dir_dataset}/validation/image/{name}.jpg", f"{newpath}/{name}.jpg")

# Prepare dataset to be loaded as ImageGenerator
if PREPARE_DATASET:
    prepare_validation_dataset(DIR_DATASET)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    zoom_range=.2,
    horizontal_flip=True,
    vertical_flip=True)
train_generator = train_datagen.flow_from_directory(
    f"{DIR_DATASET}/train/type",
    target_size=TARGET_SIZE,
    color_mode = "grayscale",
    batch_size=BATCH_SIZE
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True)
test_generator = train_datagen.flow_from_directory(
    f"{DIR_DATASET}/validation/type",
    target_size=TARGET_SIZE,
    color_mode = "grayscale",
    batch_size=BATCH_SIZE
)

class ResidualBlock(Model):
    def __init__(self, channel_in = 64, channel_out = 256):
        super().__init__()
        
        channel = channel_out // 4
        
        self.conv1 = Conv2D(channel, kernel_size = (1, 1), padding = "same")
        self.bn1 = BatchNormalization()
        self.av1 = Activation(tf.nn.relu)
        self.conv2 = Conv2D(channel, kernel_size = (3, 3), padding = "same")
        self.bn2 = BatchNormalization()
        self.av2 = Activation(tf.nn.relu)
        self.conv3 = Conv2D(channel_out, kernel_size = (1, 1), padding = "same")
        self.bn3 = BatchNormalization()
        self.shortcut = self._shortcut(channel_in, channel_out)
        self.add = Add()
        self.av3 = Activation(tf.nn.relu)
        
    def call(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.av1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.av2(h)
        h = self.conv3(h)
        h = self.bn3(h)
        shortcut = self.shortcut(x)
        h = self.add([h, shortcut])
        y = self.av3(h)
        return y
    
    def _shortcut(self, channel_in, channel_out):
        if channel_in == channel_out:
            return lambda x : x
        else:
            return self._projection(channel_out)
        
    def _projection(self, channel_out):
        return Conv2D(channel_out, kernel_size = (1, 1), padding = "same")
           
class ResNet50(Model):
    def __init__(self, input_shape, output_dim):
        super().__init__()                
        
        self._layers = [
            # conv1
            Conv2D(64, input_shape = input_shape, kernel_size = (7, 7), strides=(2, 2), padding = "same"),
            BatchNormalization(),
            Activation(tf.nn.relu),
            # conv2_x
            MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = "same"),
            ResidualBlock(64, 256),
            [
                ResidualBlock(256, 256) for _ in range(2)                
            ],
            # conv3_x
            Conv2D(512, kernel_size = (1, 1), strides=(2, 2)),
            [
                ResidualBlock(512, 512) for _ in range(4)                
            ],
            # conv4_x
            Conv2D(1024, kernel_size = (1, 1), strides=(2, 2)),
            [
                ResidualBlock(1024, 1024) for _ in range(4)                
            ],
            # conv5_x
            Conv2D(2048, kernel_size = (1, 1), strides=(2, 2)),
            [
                ResidualBlock(2048, 2048) for _ in range(4)                
            ],
            # last part
            GlobalAveragePooling2D(),
            Dense(1000, activation = tf.nn.relu),
            Dense(output_dim, activation = tf.nn.softmax)
        ]
        
    def call(self, x):
        for layer in self._layers:
            if isinstance(layer, list):
                for l in layer:
                    x = l(x)    
            else:
                x = layer(x)
        return x    
    
model = ResNet50((IMG_WIDTH, IMG_HEIGHT, 1), NUM_CLASSES)
model.build(input_shape = (None, IMG_WIDTH, IMG_HEIGHT, 1))
model.summary()

if os.path.isfile(WEIGHTS_PATH):
    print("Loaded weights from file.")
    model.load_weights(WEIGHTS_PATH)

loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name = 'test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(image, label):
    with tf.GradientTape() as tape:
        predictions = model(image)
        loss = loss_object(label, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(label, predictions)
        
@tf.function
def test_step(image, label):
    predictions = model(image)
    loss = loss_object(label, predictions)
    
    test_loss(loss)
    test_accuracy(label, predictions)

start_time = time.time()

train_accuracies = []
test_accuracies = []
print("Starting Training.")
for epoch in range(NUM_EPOCHS):
    progress = 0
    for image, label in train_generator:
        progress += BATCH_SIZE
        train_step(image, label)
        
        if progress >= IMG_PER_EPOCHS:
            print(f"Progress: 100.00%")
            break

        print(f"Progress: {progress*100/IMG_PER_EPOCHS:.2f}%")
        
    test_progress = 0
    print("Testing.")
    for image, label in test_generator:
        progress += BATCH_SIZE
        test_step(image, label)

        if progress >= IMG_PER_EPOCHS:
            break

    train_accuracies.append(train_accuracy.result())
    test_accuracies.append(test_accuracy.result())
    model.save_weights(WEIGHTS_PATH)
    
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}, spent_time: {} min'
    spent_time = time.time() - start_time
    print(template.format(epoch + 1, train_loss.result(), train_accuracy.result() * 100, test_loss.result(), test_accuracy.result() * 100, spent_time / 60))

plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')

plt.legend()
plt.show()