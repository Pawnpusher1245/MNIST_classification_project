import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 5)
model.save('mnist.model')
loss, accuracy = model.evaluate(x_test, y_test)
print(loss, accuracy)


# Code for testing the model
"""
model = tf.keras.models.load_model('mnist.model')

for image in x_test:
    # Expand dimensions to match the input shape expected by the model
    image = np.expand_dims(image, axis=0)
    
    # Predict the digit in the image
    prediction = model.predict(image)
    
    # Get the predicted digit label
    predicted_label = np.argmax(prediction)
    
    # Print the predicted label
    print(f"This digit is probably a {predicted_label}")
    
    # Display the image
    plt.imshow(image.squeeze(), cmap=plt.cm.binary)
    plt.show()
"""