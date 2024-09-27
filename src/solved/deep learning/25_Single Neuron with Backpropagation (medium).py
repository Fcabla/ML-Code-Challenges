"""
problem_id: 25
Category: deep learning
Difficulty: medium
URL: https://www.deep-ml.com/problem/Single%20Neuron%20with%20Backpropagation
Page: 1

==== Title ====
Single Neuron with Backpropagation (medium)

==== Description ====
Write a Python function that simulates a single neuron with sigmoid activation, and implements backpropagation to update the neuron's weights and bias. The function should take a list of feature vectors, associated true binary labels, initial weights, initial bias, a learning rate, and the number of epochs. The function should update the weights and bias using gradient descent based on the MSE loss, and return the updated weights, bias, and a list of MSE values for each epoch, each rounded to four decimal places.

==== Example ====
Example:
        input: features = [[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]], labels = [1, 0, 0], initial_weights = [0.1, -0.2], initial_bias = 0.0, learning_rate = 0.1, epochs = 2
        output: updated_weights = [0.0808, -0.1916], updated_bias = -0.0214, mse_values = [0.2386, 0.2348]
        reasoning: The neuron receives feature vectors and computes predictions using the sigmoid activation. Based on the predictions and true labels, the gradients of MSE loss with respect to weights and bias are computed and used to update the model parameters across epochs.

==== Learn More ====
Neural Network Learning with Backpropagation

This question involves implementing backpropagation for a single neuron in a neural network. The neuron processes inputs and updates parameters to minimize the Mean Squared Error (MSE) between predicted outputs and true labels.

Mathematical Background

Forward Pass:

Compute the neuron output by calculating the dot product of the weights and input features and adding the bias:
                \[
                z = w_1x_1 + w_2x_2 + ... + w_nx_n + b
                \]
                \[
                \sigma(z) = \frac{1}{1 + e^{-z}}
                \]



Loss Calculation (MSE):

The Mean Squared Error is used to quantify the error between the neuron's predictions and the actual labels:
                \[
                MSE = \frac{1}{n} \sum_{i=1}^{n} (\sigma(z_i) - y_i)^2
                \]



Backward Pass (Gradient Calculation):

Compute the gradient of the MSE with respect to each weight and the bias. This involves the partial derivatives of the loss function with respect to the output of the neuron, multiplied by the derivative of the sigmoid function:
                \[
                \frac{\partial MSE}{\partial w_j} = \frac{2}{n} \sum_{i=1}^{n} (\sigma(z_i) - y_i) \sigma'(z_i) x_{ij}
                \]
                \[
                \frac{\partial MSE}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} (\sigma(z_i) - y_i) \sigma'(z_i)
                \]



Parameter Update:

Update each weight and the bias by subtracting a portion of the gradient determined by the learning rate:
                \[
                w_j = w_j - \alpha \frac{\partial MSE}{\partial w_j}
                \]
                \[
                b = b - \alpha \frac{\partial MSE}{\partial b}
                \]




Practical Implementation

This process refines the neuron's ability to predict accurately by iteratively adjusting the weights and bias based on the error gradients, optimizing the neural network's performance over multiple iterations.
"""

# ==== Code ====
import numpy as np


def train_neuron(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: float,
    learning_rate: float,
    epochs: int,
) -> (np.ndarray, float, list[float]):
    # Your code here
    def sigmoid(num: float) -> float:
        return 1 / (1 + np.exp(-num))

    features = np.array(features)
    labels = np.array(labels)
    updated_weights = initial_weights
    updated_bias = initial_bias
    mse_values = []
    for _ in range(epochs):
        # fordward pass
        z = features @ updated_weights + updated_bias
        predictions = np.array([sigmoid(zz) for zz in z])

        # loss calc
        mse = np.mean((predictions - labels) ** 2)
        mse_values.append(round(mse, 4))

        # Backward Pass (Gradient Calculation) partial derivatives :(
        errors = predictions - labels
        weight_gradients = (2 / len(labels)) * np.dot(
            features.T, errors * predictions * (1 - predictions)
        )
        bias_gradient = (2 / len(labels)) * np.sum(
            errors * predictions * (1 - predictions)
        )

        # update bias and weights
        updated_weights -= learning_rate * weight_gradients
        updated_bias -= learning_rate * bias_gradient

    updated_weights = np.round(updated_weights, 4)
    updated_bias = round(updated_bias, 4)
    return (
        np.round(updated_weights, 4).tolist(),
        float(np.round(updated_bias, 4)),
        mse_values,
    )


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {
        "features": np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]]),
        "labels": np.array([1, 0, 0]),
        "initial_weights": np.array([0.1, -0.2]),
        "initial_bias": 0.0,
        "learning_rate": 0.1,
        "epochs": 2,
    }
    print(inp)
    out = train_neuron(**inp)
    print(f"Output: {out}")
    exp = ([0.1036, -0.1425], -0.0167, [0.3033, 0.2942])
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {
        "features": np.array([[1, 2], [2, 3], [3, 1]]),
        "labels": np.array([1, 0, 1]),
        "initial_weights": np.array([0.5, -0.2]),
        "initial_bias": 0,
        "learning_rate": 0.1,
        "epochs": 3,
    }
    print(inp)
    out = train_neuron(**inp)
    print(f"Output: {out}")
    exp = ([0.4892, -0.2301], 0.0029, [0.21, 0.2087, 0.2076])
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
