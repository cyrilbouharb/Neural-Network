# README for COMPSCI 589 Homework 4

## Overview
This homework focuses on developing a neural network from scratch, focusing specifically on the implementation of the backpropagation algorithm. The main goal of this assignment is to deepen my understanding of neural network architectures, the impact of regularization, and the backpropagation learning process. In addition to the main tasks, this homework includes two extra credit assignments. The code is developed in Google Colab for an interactive analysis experience and is also available in Python script format.

## Environment Setup
Before running the scripts, please ensure you have a Python 3 environment set up with the necessary libraries installed. 

## Note for TAs
The original code was developed in Google Colab (notebook), it has been converted into Python scripts for submission as requested. To review the code's logic and the execution flow please see the notebook (.ipynb) version because this is how I was working on it, and it's easier to understand. I answer the questions in the notebook and write the code too.

## Note on Implementation
The scripts for this assignment utilize a vectorized form of the backpropagation algorithm. Vectorization enhances the efficiency of neural network training by allowing batch operations on entire datasets rather than individual data points, significantly accelerating the computational process. This approach is aligned with modern practices in neural network implementations and provides a robust framework for understanding and applying neural network concepts effectively.

## Datasets Analyzed
Wine Dataset: Aims to classify wine types based on chemical contents. It includes 178 instances with 13 numerical attributes and 3 classes. </br>
1984 United States Congressional Voting Dataset: Aims to predict the party affiliation of U.S. House of Representatives members based on their voting behavior. It includes 435 instances with 16 categorical attributes and 2 classes.</br>
Breast Cancer Dataset (Extra Credits): Aims to classify whether tissue removed via a biopsy indicates whether a person may or may not have breast cancer. There are 699 instances in this dataset. Each instance is described by 9 numerical attributes, and
there are 2 classes.</br>


### Part 1: Reproducing Example Results
**File Name: 'hmw4_examples.py' or the notebook version for interactive visualization 'hmw4_examples.ipynb'.**
This script reproduces the results of the examples given. It makes sure our implementation of the Neural Network is correct. <br/>
To run this script, use the following command in your terminal: **python hmw4_examples.py**

### Part 2: Neural Network on Wine Dataset (Vectorized)
**File Name: 'hmw4_wine.py' or the notebook version for interactive visualization 'hmw4_wine.ipynb'.**
This script trains a neural network to classify types of wine based on their chemical properties using the Wine Dataset. It explores various network architectures and regularization parameters, and evaluates the model using stratified cross-validation with detailed performance metrics.<br />
To run this script, use the following command in your terminal: **python hmw4_wine.py**

### Part 3: Neural Network on Congressional Voting Dataset (Vectorized)
**File Name: 'hmw4_voting.py' or the notebook version for interactive visualization 'hmw4_voting.ipynb'.**
This script applies a neural network to predict party affiliation (Democrat or Republican) using the 1984 United States Congressional Voting Dataset. It includes an evaluation of different network configurations and regularization settings.<br />
To run this script, use the following command in your terminal: **python hmw4_voting.py**

### Extra Credit 2: Neural Network on Breast Cancer Dataset (Vectorized)
**File Name: 'hmw4_cancer.py' or the notebook version for interactive visualization 'hmw4_cancer.ipynb'.**
This script uses a neural network to classify biopsy samples from the Breast Cancer Dataset. It analyzes the impact of different network architectures and regularization parameters on model performance. <br />
To run this script, use the following command in your terminal: **python hmw4_cancer.py**

