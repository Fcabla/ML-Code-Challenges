"""
problem_id: 47
Category: machine learning
Difficulty: medium
URL: https://www.deep-ml.com/problem/Gradient%20Descent%20Variants
Page: 3

==== Title ====
Implement Gradient Descent Variants with MSE Loss

==== Description ====
In this problem, you need to implement a single function that can perform three variants of gradient descent—Stochastic Gradient Descent (SGD), Batch Gradient Descent, and Mini-Batch Gradient Descent—using Mean Squared Error (MSE) as the loss function. The function will take an additional parameter to specify which variant to use.

==== Example ====
Example:
import numpy as np

# Sample data
X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
y = np.array([2, 3, 4, 5])

# Parameters
learning_rate = 0.01
n_iterations = 1000
batch_size = 2

# Initialize weights
weights = np.zeros(X.shape[1])

# Test Batch Gradient Descent
final_weights = gradient_descent(X, y, weights, learning_rate, n_iterations, method='batch')
output: [float,float]
# Test Stochastic Gradient Descent
final_weights = gradient_descent(X, y, weights, learning_rate, n_iterations, method='stochastic')
output: [float, float]
# Test Mini-Batch Gradient Descent
final_weights = gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size, method='mini_batch')
output: [float, float]

==== Learn More ====
Understanding Gradient Descent Variants with MSE Loss
Gradient Descent is an optimization algorithm used to minimize the cost function in machine learning models, particularly in linear regression and neural networks. The Mean Squared Error (MSE) loss function is commonly used in regression tasks. There are three main types of gradient descent based on how much data is used to compute the gradient at each iteration:
1. Batch Gradient Descent
Batch Gradient Descent computes the gradient of the MSE loss function with respect to the parameters for the entire training dataset. It updates the parameters after processing the entire dataset:

\[
\theta = \theta - \alpha \cdot \frac{2}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)}
\]

Where \( \alpha \) is the learning rate, \( m \) is the number of samples, and \( \nabla_{\theta} J(\theta) \) is the gradient of the MSE loss function.
2. Stochastic Gradient Descent (SGD)
Stochastic Gradient Descent updates the parameters for each training example individually, making it faster but more noisy:

\[
\theta = \theta - \alpha \cdot 2 \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)}
\]

Where \( x^{(i)}, y^{(i)} \) are individual training examples.
3. Mini-Batch Gradient Descent
Mini-Batch Gradient Descent is a compromise between Batch and Stochastic Gradient Descent. It updates the parameters after processing a small batch of training examples, without shuffling the data:

\[
\theta = \theta - \alpha \cdot \frac{2}{b} \sum_{i=1}^{b} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)}
\]

Where \( b \) is the batch size, a subset of the training dataset.
Each method has its advantages: Batch Gradient Descent is more stable but slower, Stochastic Gradient Descent is faster but noisy, and Mini-Batch Gradient Descent strikes a balance between the two.
"""

# ==== Code ====
import numpy as np


def gradient_descent(
    X, y, weights, learning_rate, n_iterations, batch_size=1, method="batch"
):
    # Your code here
    m, n = X.shape
    # y = y.reshape(-1, 1)

    match method:
        case "batch":
            # Update with all data
            for iteration in range(n_iterations):
                predictions = X @ weights  # dot product
                errors = predictions - y
                updates = 2 * (X.T @ errors) / m
                weights = weights - learning_rate * updates
        case "stochastic":
            # update for each datapoint
            for iteration in range(n_iterations):
                for i in range(m):
                    prediction = X[i] @ weights
                    errors = prediction - y[i]
                    updates = 2 * X[i].T * errors
                    weights = weights - learning_rate * updates
        case "mini_batch":
            # update with subset of data
            for iteration in range(n_iterations):
                for i in range(0, m, batch_size):
                    x_batch = X[i : i + batch_size]
                    y_batch = y[i : i + batch_size]
                    prediction = x_batch @ weights
                    errors = prediction - y_batch
                    updates = 2 * (x_batch.T @ errors) / batch_size
                    weights = weights - learning_rate * updates
        case _:
            return -1
    return np.round(weights, 8).tolist()


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {
        "X": np.array([[1, 1], [2, 1], [3, 1], [4, 1]]),
        "y": np.array([2, 3, 4, 5]),
        "weights": np.zeros(np.array([[1, 1], [2, 1], [3, 1], [4, 1]]).shape[1]),
        "learning_rate": 0.01,
        "n_iterations": 100,
        "method": "batch",
    }
    print(inp)
    out = gradient_descent(**inp)
    print(f"Output: {out}")
    exp = [1.14905239, 0.56176776]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {
        "X": np.array([[1, 1], [2, 1], [3, 1], [4, 1]]),
        "y": np.array([2, 3, 4, 5]),
        "weights": np.zeros(np.array([[1, 1], [2, 1], [3, 1], [4, 1]]).shape[1]),
        "learning_rate": 0.01,
        "n_iterations": 100,
        "method": "stochastic",
    }
    print(inp)
    out = gradient_descent(**inp)
    print(f"Output: {out}")
    exp = [1.0507814, 0.83659454]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 3
    print("Test Case 3:")
    inp = {
        "X": np.array([[1, 1], [2, 1], [3, 1], [4, 1]]),
        "y": np.array([2, 3, 4, 5]),
        "weights": np.zeros(np.array([[1, 1], [2, 1], [3, 1], [4, 1]]).shape[1]),
        "learning_rate": 0.01,
        "n_iterations": 100,
        "batch_size": 2,
        "method": "mini_batch",
    }
    print(inp)
    out = gradient_descent(**inp)
    print(f"Output: {out}")
    exp = [1.10334065, 0.68329431]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
