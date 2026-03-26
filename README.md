iDNA_SMP: A Semi-Supervised Model for DNA Methylation Detection

Overview

iDNA_SMP is a deep learning model designed for detecting 6mA modifications in DNA sequences. This implementation combines convolutional neural networks (CNNs) with transformer architecture to leverage both sequence information and positional weight matrix (PWM) data for improved prediction accuracy.

Requirements：
Python 3.8
PyTorch 2.0.0

Model Architecture：
The iDNA_SMP model consists of:
Three convolutional blocks (Conv1d with ReLU and Dropout)
Positional weight matrix processing layer
Transformer encoder component
Fully connected layers with batch normalization and activation functions
Key architectural features:

Input channels: 4 (representing DNA nucleotides A, C, G, T)
Convolutional kernel size: 5
Dropout rate: 0.1
Adaptive average pooling for PWM processing
Transformer with 1 head, 41 positions, 4 layers, 100 hidden dimensions, and 400 feed-forward dimension
Training Configuration
Batch size: 64
Optimizer: Adam with weight decay of 5×10⁻⁴
Learning rate: 10⁻³
Validation: Five-fold cross-validation on training dataset
Hardware: Single NVIDIA GeForce RTX 4090 GPU

Model Initialization

import torch
import torch.nn as nn
from Transformer import *  
model = iDNA_SMP(channels=32, r=4)
Forward Pass
//x: DNA sequence input [batch_size, 4, sequence_length]
//xpos: Positional weight matrix [batch_size, 4, 41]
output = model.trainModel(x, xpos)

Example Training Loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()  
        
Model Components
The model implements two key methods:
trainModel(x, xpos): Returns the final classification output with softmax activation.

Notes
The model expects DNA sequence input in one-hot encoding format (4 channels).
Positional weight matrices should be preprocessed to 41 positions.
