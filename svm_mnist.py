import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn import svm
import pickle

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Reshape the data
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Create and train the model
clf = svm.SVC(kernel='rbf')
clf.fit(x_train, y_train)

# Test the model
accuracy = clf.score(x_test, y_test)

# Print the accuracy
print("Accuracy:", accuracy)

# Save the trained model
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

"""
# Select an image from the dataset
image_index = 0  # Change this index to plot a different image
selected_image = x_train[image_index].reshape(28, 28)

# Plot the image
plt.imshow(selected_image, cmap='gray')
plt.title(f"Label: {y_train[image_index]}")
plt.axis('off')
plt.show()
"""

