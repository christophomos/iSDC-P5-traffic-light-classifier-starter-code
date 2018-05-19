import cv2 # computer vision library
import helpers # helper functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images

# Image data directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"

# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)

## Display an image in IMAGE_LIST (try finding a yellow traffic light!)
## Print out 1. The shape of the image and 2. The image's label

# The first image in IMAGE_LIST is displayed below (without information about shape or label)
selected_image = IMAGE_LIST[0][0]
plt.title(IMAGE_LIST[0][1])
plt.imshow(selected_image)
plt.show()

'''
import pandas as pd

plt.clf()
image_df = pd.DataFrame(IMAGE_LIST)
print(image_df.iloc[:,1].unique())
yellow_image_df = image_df.loc[image_df.iloc[:,1] == 'yellow']
yellow_image_df.describe()
plt.title(yellow_image_df.iloc[0,1])
plt.imshow(yellow_image_df.iloc[0,0])
plt.show()
'''


# This function should take in an RGB image and return a new, standardized version
def standardize_input(image):
    ## Resize image and pre-process so that all "standard" images are the same size
    standard_im = cv2.resize(image, (32, 32))
    return standard_im


# One hot encode an image label
# Given a label - "red", "green", or "yellow" - return a one-hot encoded label
# Examples:
# one_hot_encode("red") should return: [1, 0, 0]
# one_hot_encode("yellow") should return: [0, 1, 0]
# one_hot_encode("green") should return: [0, 0, 1]
def one_hot_encode(label):
    # Create a one-hot encoded label that works for all classes of traffic lights
    if label == 'red':
        return [1, 0, 0]
    elif label == 'green':
        return [0, 0, 1]
    elif label == 'yellow':
        return [0, 1, 0]
    else:
        raise ValueError('{} is not a valid image label'.format(label))

# Importing the tests
import test_functions
tests = test_functions.Tests()

# Test for one_hot_encode function
tests.test_one_hot(one_hot_encode)


def standardize(image_list):
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)

        # Append the image, and it's one hot encoded label to the full, processed list of image data
        standard_list.append((standardized_im, one_hot_label))

    return standard_list


# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)

import random as rd

## Display a random standardized image and its label
random_image = rd.choice(IMAGE_LIST)
plt.title(random_image[1])
plt.imshow(random_image[0])
plt.show()

random_std_image = rd.choice(STANDARDIZED_LIST)
plt.title(random_std_image[1])
plt.imshow(random_std_image[0])

def get_one_color(image_list, one_hot):
    return list(filter(lambda entry: entry[1] == one_hot, image_list))

red_entries, yellow_entries, green_entries = tuple(get_one_color(STANDARDIZED_LIST, one_hot) for one_hot in [[1,0,0], [0,1,0],[0,0,1]])
red = np.array(red_entries)[:,0]
yellow = np.array(yellow_entries)[:,0]
green = np.array(green_entries)[:,0]
print(red.shape)
print(yellow.shape)
print(green.shape)

sample_size = 10

sample_red = np.array(red[:sample_size])
sample_yellow = np.array(yellow[:sample_size])
sample_green = np.array(green[:sample_size])

def display_all(image_list):
    for index, entry in enumerate(image_list):
        plt.title(index)
        plt.imshow(entry)
        plt.grid(True)
        plt.show()

# Remove [:2] to see all
display_all(sample_red[:2])

# Create a custom kernel

# 3x3 array for edge detection
high_pass = np.array([[0, -1, 0],
                   [ -1, 4, -1],
                   [ 0, -1, 0]])

def display_edges(image, title):
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)
    filtered_image = cv2.filter2D(grayscale, -1, high_pass)
    plt.title(title)
    plt.imshow(filtered_image, cmap='gray')
    plt.grid(True)
    plt.show()

for index, image in enumerate(sample_red):
    display_edges(image, index)
