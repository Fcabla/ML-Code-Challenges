"""
problem_id: 40
Category: deep learning
Difficulty: hard
URL: https://www.deep-ml.com/problem/Dense%20Layer
Page: 2

==== Title ====
Implementing a Custom Dense Layer in Python

==== Description ====
Implementing a Custom Dense Layer in Python
You are provided with a base Layer class that defines the structure of a neural network layer. Your task is to implement a subclass called Dense, which represents a fully connected neural network layer. The Dense class should extend the Layer class and implement the following methods:


Initialization (__init__):

Define the layer with a specified number of neurons (n_units) and an optional input shape (input_shape).
Set up placeholders for the layer's weights (W), biases (w0), and optimizers.



Weight Initialization (initialize):

Initialize the weights W  using a uniform distribution with a limit of 1 / sqrt(input_shape[0]), and bias w0 should be set to zero
Initialize optimizers for W and w0.



Parameter Count (parameters):

Return the total number of trainable parameters in the layer, which includes the parameters in W and w0.



Forward Pass (forward_pass):

Compute the output of the layer by performing a dot product between the input X and the weight matrix W, and then adding the bias w0.



Backward Pass (backward_pass):

Calculate and return the gradient with respect to the input.
If the layer is trainable, update the weights and biases using the optimizer's update rule.



Output Shape (output_shape):

Return the shape of the output produced by the forward pass, which should be (self.n_units,).



Objective: Extend the Layer class by implementing the Dense class to ensure it functions correctly within a neural network framework.

==== Example ====
Example Usage:

# Initialize a Dense layer with 3 neurons and input shape (2,)
dense_layer = Dense(n_units=3, input_shape=(2,))

# Define a mock optimizer with a simple update rule
class MockOptimizer:
    def update(self, weights, grad):
        return weights - 0.01 * grad

optimizer = MockOptimizer()

# Initialize the Dense layer with the mock optimizer
dense_layer.initialize(optimizer)

# Perform a forward pass with sample input data
X = np.array([[1, 2]])
output = dense_layer.forward_pass(X)
print("Forward pass output:", output)

# Perform a backward pass with sample gradient
accum_grad = np.array([[0.1, 0.2, 0.3]])
back_output = dense_layer.backward_pass(accum_grad)
print("Backward pass output:", back_output)

Expected Output:

Forward pass output: [[-0.00655782  0.01429615  0.00905812]]
Backward pass output: [[ 0.00129588  0.00953634]]

==== Learn More ====
Understanding the Dense Layer
The Dense layer, also known as a fully connected layer, is a fundamental building block in neural networks. It connects each input neuron to each output neuron, hence the term "fully connected."
1. Weight Initialization
In the `initialize` method, weights are typically initialized using a uniform distribution within a certain range. For a Dense layer, a common practice is to set this range as:
$$ 	ext{limit} = \frac{1}{\sqrt{	ext{input_shape}}} $$
This initialization helps in maintaining a balance in the distribution of weights, preventing issues like vanishing or exploding gradients during training.
2. Forward Pass
During the forward pass, the input data \(X\) is multiplied by the weight matrix \(W\) and added to the bias \(w0\) to produce the output:
$$ 	ext{output} = X \cdot W + w0 $$

3. Backward Pass
The backward pass computes the gradients of the loss function with respect to the input data, weight, and bias. If the layer is trainable, it updates the weights and biases using the optimizer's update rule:
$$ W = W - \eta \cdot \text{grad}_W $$
$$ w0 = w0 - \eta \cdot \text{grad}_{w0} $$
where \(\eta\) is the learning rate and \(\text{grad}_W\) and \(\text{grad}_{w0}\) are the gradients of the weights and biases, respectively.
4. Output Shape
The shape of the output from a Dense layer is determined by the number of neurons in the layer. If a layer has `n_units` neurons, the output shape will be `(n_units,)`.
Resources:

CS231n: Fully Connected Layer
"""

# ==== Code ====

import copy
import math

import numpy as np

# DO NOT CHANGE SEED
np.random.seed(42)


# DO NOT CHANGE LAYER CLASS
class Layer(object):
    def set_input_shape(self, shape):
        self.input_shape = shape

    def layer_name(self):
        return self.__class__.__name__

    def parameters(self):
        return 0

    def forward_pass(self, X, training):
        raise NotImplementedError()

    def backward_pass(self, accum_grad):
        raise NotImplementedError()

    def output_shape(self):
        raise NotImplementedError()


# Your task is to implement the Dense class based on the above structure
class Dense(Layer):
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.w0 = None

    def initialize(self, optimizer):
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
        self.w0 = np.zeros((1, self.n_units))
        self.W_opt = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)

    def parameters(self):
        # return the number of elements of W and W0
        return np.prod(self.W.shape) + np.prod(self.w0.shape)

    def forward_pass(self, x):
        self.layer_input = x
        return x @ self.W + self.w0

    def backward_pass(self, accum_grad):
        w = self.W.copy()
        if self.trainable:
            # partial derivatives
            # layer_input.T to match sizes
            grad_w = self.layer_input.T @ accum_grad
            grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)
            self.W = self.W_opt.update(self.W, grad_w)
            self.w0 = self.W_opt.update(self.w0, grad_w0)
        accum_grad = accum_grad @ w.T
        return accum_grad

    def output_shape(self):
        return (self.n_units,)
        # return self.n_units


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    # Initialize a Dense layer with 3 neurons and input shape (2,)
    dense_layer = Dense(n_units=3, input_shape=(2,))

    # Define a mock optimizer with a simple update rule
    class MockOptimizer:
        def update(self, weights, grad):
            return weights - 0.01 * grad

    optimizer = MockOptimizer()

    # Initialize the Dense layer with the mock optimizer
    dense_layer.initialize(optimizer)

    # Perform a forward pass with sample input data
    X = np.array([[1, 2]])
    print("Forward pass input:", X)
    output = dense_layer.forward_pass(X)
    print("Forward pass output:", output)

    # Perform a backward pass with sample gradient
    accum_grad = np.array([[0.1, 0.2, 0.3]])
    print("Backward pass input accumulated gradient:", accum_grad)
    back_output = np.round(dense_layer.backward_pass(accum_grad), 8).tolist()
    print("Backward pass output:", back_output)

    exp = [[0.20816524, -0.22928937]]
    # ---

    print(f"Expected: {exp}")
    print("Accepted" if back_output == exp else "Error")
    print("---")
