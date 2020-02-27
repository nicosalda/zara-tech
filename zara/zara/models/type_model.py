import os
import tensorflow as tf
from zara.models.ResNet50 import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Model CONSTANTS
IMG_HEIGHT = 56
IMG_WIDTH = 56
TARGET_SIZE = (IMG_WIDTH, IMG_HEIGHT)
# Number of current dress types available
NUM_CLASSES = 13
# Path to the h5 file holding the model weights
WEIGHTS_PATH = "model.h5"
# Dictionary to convert from id to type
id2x = {
    0: "long_sleeve_dress",
    1: "long_sleeve_outwear",
    2: "long_sleeve_top",
    3: "short_sleeve_dress",
    4: "short_sleeve_outwear",
    5: "short_sleeve_top",
    6: "shorts",
    7: "skirt",
    8: "sling",
    9: "sling_dress",
    10: "trousers",
    11: "vest",
    12: "vest_dress"
}

# Creates the model
def create_model():
    # Creating the Residual Network
    model = ResNet50((IMG_WIDTH, IMG_HEIGHT, 1), NUM_CLASSES)
    # Building keras model
    model.build(input_shape = (None, IMG_WIDTH, IMG_HEIGHT, 1))
    # We load the trained weights
    model.load_weights(os.path.dirname(os.path.abspath( __file__ )) + f'/{WEIGHTS_PATH}')

    return model

model = create_model()

# Returns the type of the dress
def dress_type(file):
    # We load the image and transform it as needed
    img = load_img(file, color_mode="grayscale", target_size=TARGET_SIZE)
    # Get the weights of this image
    types = model.call(tf.expand_dims(img_to_array(img), 0))
    # Calculate the class that match the most
    max_type = tf.math.argmax(types, 1).numpy()[0]
    # Return the type
    return id2x[max_type]