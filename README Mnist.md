# MNIST Digit Classification with TensorFlow and Keras
This project implements a **Convolutional Neural Network (CNN)** using TensorFlow and Keras to classify handwritten digits from the **MNIST dataset**.

## Overview
The goal of this project is to train a deep learning model capable of accurately predicting handwritten digits (0–9) based on grayscale images. The project demonstrates the complete pipeline, including data preprocessing, model training, evaluation, and visualization.

## Features
- **Dataset**: MNIST dataset (`keras.datasets.mnist`) containing 60,000 training samples and 10,000 test samples of 28x28 grayscale images.
- **Architecture**: A CNN with:
  - Two convolutional layers (32 and 64 filters respectively).
  - MaxPooling, Dropout, Flattening layers.
  - Fully connected Dense layers with ReLU and softmax activations.
- **Training**: Model trained for 25 epochs with a batch size of 64.
- **Evaluation**: Metrics include accuracy, loss (on training and test sets).
- **Visualization**: Prediction and actual values comparison with a sample image.
## Installation
To run this project, ensure you have the following dependencies installed:
- Python (3.x)
- TensorFlow
- NumPy
- Matplotlib

Install these packages using pip:
```bash
pip install tensorflow numpy matplotlib
```

---
## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/MNIST-Digit-Classifier.git
   cd MNIST-Digit-Classifier
   ```
2. Run the script:
   ```bash
   python mnist_classifier.py
   ```
3. The script will:
   - Preprocess the dataset.
   - Train the CNN.
   - Evaluate the model on the test dataset.
   - Display the prediction vs actual value for a sample image.
---
## Results
- **Training Accuracy**: ~87%
- **Testing Accuracy**: ~82%
- **Error Metrics**:
  - Mean Absolute Error (MAE): 1.70
  - Root Mean Squared Error (RMSE): 1.86
- Feature visualization through prediction comparison and model performance.
# Author ✍️ :: Osama At
