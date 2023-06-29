import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

accuracy_list = []
x_values = []

for i in range(1,101):
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Flatten the input images
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # Convert pixel values to floating point
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalize the pixel values between 0 and 1
    x_train /= 255.0
    x_test /= 255.0

    # Create k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=i)

    # Train the classifier
    knn.fit(x_train, y_train)

    # Predict the labels for the test set
    y_pred = knn.predict(x_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    accuracy_list.append(accuracy)
    x_values.append(i)

# Plotting the accuracy depending on how many neighbors there are. (Optimal seems to be 3 for this dataset) 
# I added 100 vlaues, which took a lot of time for curiosity, but of course this can be decreased

# Create the list of values for the x-axis  

# Create the plot
plt.plot(x_values, accuracy_list)

# Add labels to the x and y axes
plt.xlabel('Number of nearest neighbors')
plt.ylabel('Accuracy')

# Display the plot
plt.show()