# CIFAR-10 Object Recognition Using ResNet50

This repository contains a deep learning project for object recognition using the CIFAR-10 dataset. The project utilizes both a custom neural network built with Keras and a pre-trained ResNet50 model, showcasing manual model training and transfer learning to classify images into ten object categories.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)

## Introduction

CIFAR-10 Object Recognition is a fundamental task in computer vision, involving classification of small images (32x32 pixels) into ten object categories such as airplanes, cars, birds, cats, and more. This project demonstrates two approaches:

Custom Neural Network Training: A fully connected dense model implemented using Keras.
Transfer Learning with ResNet50: A pre-trained ResNet50 model fine-tuned on the CIFAR-10 dataset to improve performance and reduce training time.

## Dataset

The dataset used in this project is the CIFAR-10 dataset from kaggle, which contains 60,000 images divided into 10 classes. The dataset is split into training and testing sets.

Dataset Details:

- Training: 50,000 images
- Testing: 10,000 images
- Image size: 32x32 pixels, RGB format
- Accessing the Dataset: The dataset is downloaded using the Kaggle API. Ensure your kaggle.json file is placed in the correct directory for access.

- [CIFAR-10 dataset](https://www.kaggle.com/c/cifar-10/)

## Installation

To run this project, you need to have Python installed on your machine. You can install the required dependencies using `pip`.

```
pip install numpy pandas matplotlib tensorflow keras scikit-learn kaggle

```

Requirements
Python 3.x
NumPy
Pandas
Matplotlib
TensorFlow
Keras
Scikit-learn
Kaggle

## Usage

1. Clone the repository to your local machine:

```
   git clone https://github.com/srijosh/CIFAR-10-Object-Recognition-Using-ResNet50.git
```

2. Navigate to the project directory:
   cd CIFAR-10-Object-Recognition-Using-ResNet50

3. Download the kaggle.json file from kaggle website by creating new api token and place it in the project directory

4. Open and run the Jupyter Notebook:
   jupyter notebook ObjectRecognitionusingResNet50.ipynb

## Model

1. Custom Neural Network
   A simple neural network is built using Keras to classify CIFAR-10 images. The architecture includes:

- Input Layer: Flatten layer to convert 3D images to 1D input.
- Dense Layers: Fully connected layers with ReLU activation.
- Output Layer: Dense layer with Softmax activation for classification.

2. Transfer Learning with ResNet50
   The ResNet50 model is fine-tuned on the CIFAR-10 dataset. Key configurations include:

- Pre-trained Weights: From ImageNet, excluding the top layer.
- Fine-tuning: Fully connected dense layers added for CIFAR-10 classification.
- Compilation:
- Optimizer: RMSProp with a learning rate of 2e-5.
- Loss: Sparse Categorical Crossentropy.
- Metrics: Accuracy.
