""" This file contains code for a fully convolutional
(i.e. contains zero fully connected layers) neural network
for detecting lanes. This version assumes the inputs
to be road images in the shape of 80 x 160 x 3 (RGB) with
the labels as 80 x 160 x 1 (just the G channel with a
re-drawn lane). Note that in order to view a returned image,
the predictions is later stacked with zero'ed R and B layers
and added back to the initial road image.
"""

import numpy as np
import sklearn.model_selection
import tensorflow as tf
from tensorflow import keras
import pickle
import sklearn
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# import cv2

# Import necessary items from Keras


def create_model(input_shape, pool_size = (2,2)):
        """
        This will be the architecture of the model used to detect the lane lines.

        input_shape: Shape of the input image
        pool_size: Size of the pooling layer

        TODO: Search for the best architecture for the lane detection (Maybe Image Segmentation)
        """

        model = keras.Sequential()
        # Normalizes incoming inputs. First layer needs the input shape to work
        model.add(keras.layers.BatchNormalization(input_shape=input_shape))

        # Below layers were re-named for easier reading of model summary; this not necessary
        # Conv Layer 1
        model.add(keras.layers.Conv2D(8, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv1'))

        # Conv Layer 2
        model.add(keras.layers.Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv2'))

        # Pooling 1
        model.add(keras.layers.MaxPooling2D(pool_size=pool_size))

        # Conv Layer 3
        model.add(keras.layers.Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv3'))
        model.add(keras.layers.Dropout(0.2))

        # Conv Layer 4
        model.add(keras.layers.Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv4'))
        model.add(keras.layers.Dropout(0.2))

        # Conv Layer 5
        model.add(keras.layers.Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv5'))
        model.add(keras.layers.Dropout(0.2))

        # Pooling 2
        model.add(keras.layers.MaxPooling2D(pool_size=pool_size))

        # Conv Layer 6
        model.add(keras.layers.Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv6'))
        model.add(keras.layers.Dropout(0.2))

        # Conv Layer 7
        model.add(keras.layers.Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv7'))
        model.add(keras.layers.Dropout(0.2))

        # Pooling 3
        model.add(keras.layers.MaxPooling2D(pool_size=pool_size))

        # Upsample 1
        model.add(keras.layers.UpSampling2D(size=pool_size))

        # Deconv 1
        model.add(keras.layers.Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv1'))
        model.add(keras.layers.Dropout(0.2))

        # Deconv 2
        model.add(keras.layers.Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv2'))
        model.add(keras.layers.Dropout(0.2))

        # Upsample 2
        model.add(keras.layers.UpSampling2D(size=pool_size))

        # Deconv 3
        model.add(keras.layers.Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv3'))
        model.add(keras.layers.Dropout(0.2))

        # Deconv 4
        model.add(keras.layers.Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv4'))
        model.add(keras.layers.Dropout(0.2))

        # Deconv 5
        model.add(keras.layers.Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv5'))
        model.add(keras.layers.Dropout(0.2))

        # Upsample 3
        model.add(keras.layers.UpSampling2D(size=pool_size))

        # Deconv 6
        model.add(keras.layers.Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv6'))

        # Final layer - only including one channel so 1 filter
        model.add(keras.layers.Conv2DTranspose(1, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Final'))

        return model


def main():
    # Load training images
    train_images = pickle.load(open("full_CNN_train.p", "rb" ))

    # Load image labels
    labels = pickle.load(open("full_CNN_labels.p", "rb" ))

    # Make into arrays as the neural network wants these
    train_images = np.array(train_images)
    labels = np.array(labels)

    # Normalize labels - training images get normalized to start in the network
    labels = labels / 255

    # Shuffle images along with their labels, then split into training/validation sets
    train_images, labels = sklearn.utils.shuffle(train_images, labels)

    # Test size may be 10% or 20%
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(train_images, labels, test_size=0.1)

    # Batch size, epochs and pool size below are all paramaters to fiddle with for optimization
    batch_size = 128
    epochs = 10
    pool_size = (2, 2)
    input_shape = X_train.shape[1:]

    # Create the neural network
    model = create_model(input_shape, pool_size)

    # Using a generator to help the model use less data
    # Channel shifts help with shadows slightly
    # Compiling and training the model
    model.compile(optimizer='Adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))

    # Freeze layers since training is done
    # model.trainable = False
    # model.compile(optimizer='Adam', loss='mean_squared_error')

    # Save model architecture and weights
    model.save('full_CNN_model.h5')

    # Show summary of model
    model.summary()

# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

if __name__ == '__main__':
    # main()

    model = keras.models.load_model('full_CNN_model.h5')
    lanes = Lanes()

    train_images = pickle.load(open("full_CNN_train.p", "rb" ))

    image = train_images[0]

    small_img = np.array(image)
    small_img = small_img[None,:,:,:]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255

    lanes.recent_fit.append(prediction)

    # Add lane prediction to list for averaging
    # lanes.recent_fit.append(prediction)
    # Only using last five for average
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
    lane_image = lane_drawn

    print(lane_image.shape)
    print(image.shape)

    img1 = Image.fromarray(lane_image, 'RGB')
    img2 = Image.fromarray(image, 'RGB')

    # Combine the result with the original image
    result = Image.blend(img2, img1, alpha=1)
    plt.imshow(result)
    plt.show()
