# Building a Brain in 10 Minutes: Image Classification with Keras

This repository contains my completed implementation of the **"Building a Brain in 10 Minutes"** interactive tutorial provided by **NVIDIA's Deep Learning Institute (NVDLI)**. 

The project demonstrates how to build and train a basic single-layer artificial neural network (a Perceptron framework) using TensorFlow and Keras to classify articles of clothing.

## 🚀 Project Overview
The goal of this project is to explore the foundational mathematical concepts behind deep learning (weights, biases, and linear functions) and apply them to a computer vision challenge: image classification. 

The trained network analyzes grayscale images of clothing and predicts their category with roughly **80% accuracy** after only 5 training cycles (epochs).

## 📊 Dataset: Fashion MNIST
The model trains on the popular **Fashion MNIST** dataset, which consists of 70,000 $28 \times 28$ pixel grayscale images split into training (study) and validation (quiz) sets. 

Images are divided into 10 distinct categories:
| Label | Description |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## 🧠 Model Architecture & Technical Breakdown
The artificial neural network is built using Keras's `Sequential` API and consists of the following pipeline:

1. **`Flatten` Layer:** Converts the 2D image arrays ($28 \times 28$ pixels) into a 1D array of 784 individual pixel inputs.
2. **`Dense` Layer (The Brain):** Employs 10 fully connected neurons (one for each class). Each neuron maps a weight ($w$) to every incoming pixel and adds a unique bias ($b$), calculating outputs via multivariate linear regression:
   $$y = w_0x_0 + w_1x_1 + w_2x_2 + \dots + b$$
3. **Compilation:** Optimized using the `adam` optimizer and graded via `SparseCategoricalCrossentropy` loss to measure confidence errors during training.

## 📈 Training Results
The network was trained on an NVIDIA GPU backend instance for 5 epochs:
* **Epoch 1/5:** Accuracy: ~74.5% | Loss: ~16.87
* **Epoch 5/5:** Accuracy: **~80.0%** | Loss: ~10.59

## 🔮 Predictions & Visualization
The final section of the notebook passes raw testing images into `model.predict()` to evaluate real-world capability, generating confidence bar charts using `matplotlib` to visualize how strongly the network favors a given category.

---
*Acknowledge: The base framework and notebook materials are properties of the NVIDIA Deep Learning Institute (NVDLI).*
