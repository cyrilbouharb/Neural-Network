#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# In[2]:


class NN:
    def __init__(self, sizes, initial_weights=None, eta=0.01, reg_lambda=0.01, use_softmax=False):
        self.sizes = sizes
        self.eta = eta
        self.reg_lambda = reg_lambda
        self.use_softmax = use_softmax
        self.weights = self.init_weights() if initial_weights is None else initial_weights

    def init_weights(self):
        return [np.random.randn(y, x + 1) * 0.1 for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def add_bias(self, X):
        return np.insert(X, 0, 1, axis=1)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        act = [self.add_bias(X)]
        ll = []
        for index, weight in enumerate(self.weights[:-1]):
            ll.append(act[-1] @ weight.T)
            act.append(self.add_bias(1 / (1 + np.exp(-ll[-1]))))
        ll.append(act[-1] @ self.weights[-1].T)
        act.append(self.softmax(ll[-1]) if self.use_softmax else 1 / (1 + np.exp(-ll[-1])))
        return act, ll

    def cost(self, Y, act):
        xq = Y.shape[0]
        return -np.sum(Y * np.log(act[-1] + 1e-8)) / xq if self.use_softmax else -np.sum(Y * np.log(act[-1]) + (1 - Y) * np.log(1 - act[-1])) / xq

    def backward(self, targets, activations, weights):
        num_samples = targets.shape[0]
        errors = [activations[-1] - targets]
        gradients = []
        for i in range(len(weights) - 1, 0, -1):
            delta = (errors[0] @ weights[i][:, 1:]) * activations[i][:, 1:] * (1 - activations[i][:, 1:])
            errors.insert(0, delta)
        for i in range(len(weights)):
            grad = errors[i].T @ activations[i] / num_samples
            gradients.append(grad)
        return errors, gradients

    def update_weights(self, gradients):
        for i in range(len(self.weights)):
            regularization_grad = (self.reg_lambda * self.weights[i]) / self.weights[i].shape[0]
            regularization_grad[:, 0] = 0
            self.weights[i] -= self.eta * (gradients[i] + regularization_grad)

    def predict(self, inputs):
        final_activations, _ = self.forward(inputs)
        return np.argmax(final_activations[-1], axis=1) + 1

    def train(self, inputs, labels, iterations):
        for _ in range(iterations):
            final_activations, _ = self.forward(inputs)
            _, weight_gradients = self.backward(labels, final_activations, self.weights)
            self.update_weights(weight_gradients)
        return self.weights


# In[3]:


def examples_testing(nn, X_example, Y_example):
    print(f"Regularization parameter lambda={nn.reg_lambda:.3f}")
    print("\nInitializing the network with the following structure (number of neurons per layer):")
    neuron_counts = [nn.weights[0].shape[1] - 1] + [layer.shape[0] for layer in nn.weights]
    print(neuron_counts)
    print("\nInitial weights:")
    for idx, layer in enumerate(nn.weights):
        print(f"\nInitial Theta{idx+1} (the weights of each neuron, including the bias weight, are stored in the rows):")
        for row in layer:
            print('\t' + '  '.join(f"{weight:.5f}" for weight in row))
    print("\nTraining set")
    total_cost = 0
    forward_results = []
    for i in range(len(X_example)):
        print(f"\n\tTraining instance {i+1}")
        x_str = '  '.join(f"{num:.5f}" for num in X_example[i])
        print(f"\t\tx: [{x_str}]")
        y_str = '  '.join(f"{num:.5f}" for num in Y_example[i])
        print(f"\t\ty: [{y_str}]")
    print('--------------------------------------------')
    print("Computing the error/cost, J, of the network")
    for i in range(len(X_example)):
        print(f"Processing training instance {i+1}")
        activations, zs = nn.forward(np.array([X_example[i]]))
        cost = nn.cost(np.array([Y_example[i]]), activations[-1])
        total_cost += cost
        forward_results.append((activations, zs, cost))
        print(f"\t\tForward propagating the input [{x_str}]")
        for j, a in enumerate(activations):
            a_str = '  '.join(f"{x:.5f}" for x in a[0])
            print(f"\t\ta{j+1}: [{a_str}]")
            if j < len(zs):
                z_str = '  '.join(f"{x:.5f}" for x in zs[j][0])
                print(f"\t\tz{j+2}: [{z_str}]")
        pred_output = activations[-1][0]
        pred_output_str = '  '.join(f"{p:.5f}" for p in pred_output)
        y_str = '  '.join(f"{num:.5f}" for num in Y_example[i])
        print(f"\t\tf(x): [{pred_output_str}]")
        print(f"\t\tPredicted output for instance {i+1}: [{pred_output_str}]")
        print(f"\t\tExpected output for instance {i+1}: [{y_str}]")
        print(f"\t\tCost, J, associated with instance {i+1}: {cost:.3f}")
    average_unregularized_cost = total_cost / len(X_example)
    reg_cost = (nn.reg_lambda / (2 * len(X_example))) * sum(np.sum(np.square(w[:, 1:])) for w in nn.weights)
    final_cost = average_unregularized_cost + reg_cost
    print(f"\nFinal (regularized) cost, J, based on the complete training set: {final_cost:.5f}\n")
    print("--------------------------------------------")
    print("Running backpropagation")
    all_gradients = [[] for _ in nn.weights]
    for i, (activations, zs, cost) in enumerate(forward_results):
        print(f"\n\tComputing gradients based on training instance {i+1}")
        deltas, gradients = nn.backward(np.array([Y_example[i]]), activations, nn.weights)
        for j in reversed(range(len(deltas))):
            delta_str = '  '.join(f"{x:.5f}" for x in deltas[j].flatten())
            print(f"\t\tdelta{j+2}: [{delta_str}]")
        for idx in reversed(range(len(gradients))):
            all_gradients[idx].append(gradients[idx])
            grad_str = '\n\t\t'.join('  '.join(f"{g:.5f}" for g in row) for row in gradients[idx])
            print(f"\n\t\tGradients of Theta{idx+1} based on training instance {i+1}:")
            print(f"\t\t{grad_str}")
    print("\nThe entire training set has been processed. Computing the average (regularized) gradients:")
    for idx in range(len(all_gradients)):
        avg_grad = np.mean(all_gradients[idx], axis=0)
        avg_grad[:, 1:] += (nn.reg_lambda / len(X_example)) * nn.weights[idx][:, 1:]
        avg_grad_str = '\n\t'.join('  '.join(f"{g:.5f}" for g in row) for row in avg_grad)
        print(f"\n\tFinal regularized gradients of Theta{idx + 1}:")
        print(f"\t{avg_grad_str}")


# In[6]:


layer_sizes = [2, 2, 1]
weights_example = [np.array([[0.4, 0.1], [0.3, 0.2]]), np.array([[0.7, 0.5, 0.6]])]
nn = NN(layer_sizes,weights_example, eta=0.01, reg_lambda=0.00, use_softmax=False)
examples_testing(nn, np.array([[0.13], [0.42]]), np.array([[0.9], [0.23]]))


# **Example 2**

# In[5]:


layer_sizes = [2, 4, 3, 2]
weights_example =  [np.array([[0.42, 0.15, 0.4], [0.72, 0.1, 0.54], [0.01, 0.19, 0.42], [0.3, 0.35, 0.68]]),
    np.array([[0.21, 0.67, 0.14, 0.96, 0.87], [0.87, 0.42, 0.2, 0.32, 0.89], [0.03, 0.56, 0.8, 0.69, 0.09]]),
    np.array([[0.04, 0.87, 0.42, 0.53], [0.17, 0.1, 0.95, 0.69]])]
nn = NN(layer_sizes, weights_example, eta=0.01, reg_lambda=0.25, use_softmax=False)
examples_testing(nn, np.array([[0.32, 0.68], [0.83, 0.02]]), np.array([[0.75, 0.98], [0.75, 0.28]]))

