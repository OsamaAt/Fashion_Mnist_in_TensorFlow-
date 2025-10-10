# Importing required libraries
import tensorflow as tf
from tensorflow import keras  
import matplotlib.pyplot as plt
import numpy as np  

# Loading the MNIST dataset (handwritten digits)
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

# Normalizing the image data to scale the pixel values between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0

# One-hot encoding the labels to convert them into categorical format
Y_train = keras.utils.to_categorical(Y_train, 10) 
Y_test = keras.utils.to_categorical(Y_test, 10)

# Defining the architecture of the neural network using Keras Sequential API
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compiling the model with Adam optimizer, categorical crossentropy loss, and accuracy as the evaluation metric
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Displaying a summary of the model architecture
model.summary()

# Training the model on the training data with a validation split of 20%
history = model.fit(X_train, Y_train, batch_size=64, epochs=25, validation_split=0.2)

# Evaluating the model on the test data to calculate accuracy and loss
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f'Accuracy Test : {test_acc} , Loss Test : {test_loss}')

# Making predictions on the test data
Y_pred = model.predict(X_test)
print(f'Final Prediction : {Y_pred[10]}')  

# Displaying a sample test image and its prediction
index = 5  # Choose the index of the test image
plt.imshow(X_test[index], cmap='gray')  
plt.title(f'Prediction : {np.argmax(Y_pred[index])} , Actual : {np.argmax(Y_test[index])}')  
plt.show()
