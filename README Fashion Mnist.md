🧥 Fashion MNIST Classification using TensorFlow
📘 Overview

This project builds and trains a simple Neural Network using TensorFlow to classify images from the Fashion MNIST dataset — a dataset containing 70,000 grayscale images of clothing items across 10 categories (such as shirts, shoes, bags, etc.).

The goal is to correctly identify the clothing item category based on its pixel values.

🧩 Features

Loads and preprocesses the Fashion MNIST dataset.

Normalizes image pixel values between 0 and 1 for better training performance.

Defines a simple Fully Connected Neural Network (Dense layers) using the Sequential API.

Implements a custom callback that stops training when loss drops below 0.4.

Demonstrates how Softmax works on tensor inputs before model training.

Evaluates model accuracy on the test dataset.

🧠 Model Architecture
Layer Type	Details
Input	(28, 28, 1) grayscale image
Flatten	Converts 2D image into 1D vector
Dense	128 neurons, ReLU activation
Dense (Output)	10 neurons, Softmax activation
⚙️ Technologies Used

Python 3
NumPy
Matplotlib
TensorFlow / Keras

📊 Results

Model trained for up to 5 epochs.

Training stops early when the loss goes below 0.4.

Achieved around 87–89% accuracy on the test dataset (may vary slightly per run).

.

📈 Example Output
Loss is under 0.4 — stopping training early.
Test Accuracy: 0.88

🧑‍💻 Author
Osama Al Attar
