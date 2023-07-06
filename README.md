# MNIST_classification_project
A project to explore different classification methods on the MNIST dataset
# Project Title: MNIST_classification_project

## Introduction
Welcome to the MNIST Classification Exploration project! This project aims to test various classification methods on the famous MNIST dataset. The MNIST dataset consists of a large collection of handwritten digit images, where the goal is to correctly classify each image into its corresponding digit (0-9).

This README specifically focuses on the `knn_mnist.py` file, which is the first file in the project. It is important to note that this project is a work in progress, and additional classification methods will be added in the future.

## File: knn_mnist.py
The `knn_mnist.py` file contains code that implements the k-Nearest Neighbors (k-NN) classification algorithm on the MNIST dataset. The purpose of this file is to explore the performance of k-NN with varying numbers of nearest neighbors.

### Dependencies
The following dependencies are required to run the code in `knn_mnist.py`:
- NumPy (imported as `np`)
- scikit-learn library (imported as `sklearn`)
- matplotlib library (imported as `plt`)
- TensorFlow library (imported as `tensorflow`)

### Functionality
The code in `knn_mnist.py` performs the following steps:

1. Imports the necessary libraries and modules.
2. Initializes empty lists to store accuracy values and the corresponding number of nearest neighbors.
3. Iterates over a range of values from 1 to 100 (inclusive) to define the number of neighbors for k-NN.
4. Loads the MNIST dataset using `mnist.load_data()` function from TensorFlow.
5. Reshapes the input images into a 1D array using NumPy's `reshape()` function.
6. Converts the pixel values to floating point and normalizes them between 0 and 1.
7. Creates a k-NN classifier object with the specified number of neighbors.
8. Trains the k-NN classifier on the training data using the `fit()` method.
9. Predicts the labels for the test set using the trained classifier.
10. Calculates the accuracy of the classification using scikit-learn's `accuracy_score()` function.
11. Appends the accuracy and number of neighbors to their respective lists.
12. Plots a line graph to visualize the relationship between the number of nearest neighbors and accuracy.
13. Displays the plot.

### Usage
To run the code in `knn_mnist.py`, ensure that you have installed the required dependencies mentioned above. Execute the script, and it will output the accuracy for each iteration of the k-NN algorithm with varying numbers of neighbors. Additionally, a line graph will be displayed, showing the relationship between the number of nearest neighbors and accuracy.

Feel free to modify the code to experiment with different parameters or incorporate additional functionality.

## Future Work
This project is an ongoing exploration of different classification methods on the MNIST dataset. In the future, more classification algorithms will be added, such as Support Vector Machines (SVM), Random Forests, and Artificial Neural Networks (ANN).


