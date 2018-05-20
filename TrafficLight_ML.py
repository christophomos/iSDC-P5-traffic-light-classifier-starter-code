
import helpers # helper functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images

# Image data directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"

# Using the load_dataset function in helpers.py
# Load training data and test data
# For each entry, [0] is image and [1] is label: 'red','yellow','green'
training_images = helpers.load_dataset(IMAGE_DIR_TRAINING)
testing_images = helpers.load_dataset(IMAGE_DIR_TEST)

# Display first image in training_images
'''
selected_image = training_images[0][0]
plt.title(training_images[0][1])
plt.imshow(selected_image)
plt.show()
'''

# Importing the tests
import test_functions
tests = test_functions.Tests()

# Test for one_hot_encode function
tests.test_one_hot(helpers.one_hot_encode)


# Standardize all training and test images
standardized_training_images = helpers.standardize(training_images)
standardized_testing_images = helpers.standardize(testing_images)

import random as rd

## Display a random standardized images and its label
'''
random_std_image = rd.choice(standardized_training_images)
plt.title(random_std_image[1])
plt.imshow(random_std_image[0])
'''

red_entries, yellow_entries, green_entries = tuple(helpers.get_one_color(standardized_training_images, one_hot) for one_hot in [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
red = np.array(red_entries)[:,0]
yellow = np.array(yellow_entries)[:,0]
green = np.array(green_entries)[:,0]
print("Red shape", red.shape)
print("Yellow shape", yellow.shape)
print("Green shape", green.shape)

sample_size = 2

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
#display_all(sample_red)

# Got the idea to use logistic regression from the following website
# https://www.codementor.io/mgalarny/making-your-first-machine-learning-classifier-in-scikit-learn-python-db7d7iqdh
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

def flatten(image):
    np_array_image = np.array(image)
    return np_array_image.flatten()

images, labels = zip(*standardized_training_images)
test_images, test_labels = zip(*standardized_testing_images)

images = np.array(images)
test_images = np.array(test_images)
labels = list(map(helpers.one_hot_to_ordinal, labels))
test_labels = list(map(helpers.one_hot_to_ordinal, test_labels))

images = list(map(flatten, images))
test_images = list(map(flatten, test_images))

print(len(images))
print(len(images[0]))

print("labels[:10]",labels[:10])
#print(len(labels))

classifier.fit(images, labels)

predictions = classifier.predict(test_images)
print('predictions[:50]',predictions[:50])
print('sum(predictions)',sum(predictions))
accuracy = (predictions == test_labels).mean()
print('test_labels[:50]',test_labels[:50])
print('sum(test_labels)',sum(test_labels))

for p, t in zip(predictions, test_labels):
    print("{}, {}".format(p, t))

print('accuracy',accuracy)

