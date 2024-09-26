"""
problem_id: 15
Category: machine learning
Difficulty: easy
URL: https://www.deep-ml.com/problem/Linear%20Regression%20Using%20Gradient%20Descent
Page: 1

==== Title ====
Linear Regression Using Gradient Descent (easy)

==== Description ====
Write a Python function that performs linear regression using gradient descent. The function should take NumPy arrays X (features with a column of ones for the intercept) and y (target) as input, along with learning rate alpha and the number of iterations, and return the coefficients of the linear regression model as a NumPy array. Round your answer to four decimal places. -0.0 is a valid result for rounding a very small number.

==== Example ====
Example:
        input: X = np.array([[1, 1], [1, 2], [1, 3]]), y = np.array([1, 2, 3]), alpha = 0.01, iterations = 1000
        output: np.array([0.1107, 0.9513])
        reasoning: The linear model is y = 0.0 + 1.0*x, which fits the input data after gradient descent optimization.

==== Learn More ====
Linear Regression Using Gradient Descent

Linear regression can also be performed using a technique called gradient descent, where the coefficients (or weights) of the model are iteratively adjusted to minimize a cost function (usually mean squared error). This method is particularly useful when the number of features is too large for analytical solutions like the normal equation or when the feature matrix is not invertible.

The gradient descent algorithm updates the weights by moving in the direction of the negative gradient of the cost function with respect to the weights. The updates occur iteratively until the algorithm converges to a minimum of the cost function.

The update rule for each weight is given by:
\[
\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( h_{\theta}(x^{(i)}) - y^{(i)} \right)x_j^{(i)}
\]
Where:

 \(\alpha\) is the learning rate,
 \(m\) is the number of training examples,
 \(h_{\theta}(x^{(i)})\) is the hypothesis function at iteration \(i\),
 \(x^{(i)}\) is the feature vector of the \(i^{th}\) training example,
 \(y^{(i)}\) is the actual target value for the \(i^{th}\) training example,
 \(x_j^{(i)}\) is the value of feature \(j\) for the \(i^{th}\) training example.

Things to note: The choice of learning rate and the number of iterations are crucial for the convergence and performance of gradient descent. Too small a learning rate may lead to slow convergence, while too large a learning rate may cause overshooting and divergence.
 Practical Implementation

Implementing gradient descent involves initializing the weights, computing the gradient of the cost function, and iteratively updating the weights according to the update rule.
"""

# ==== Code ====
import numpy as np


def linear_regression_gradient_descent(
    X: np.ndarray, y: np.ndarray, alpha: float, iterations: int
) -> np.ndarray:
    # Your code here, make sure to round
    m, n = X.shape
    theta = np.zeros((n, 1))
    y = y.reshape(-1, 1)
    for iteration in range(iterations):
        predictions = X @ theta
        # the "hypothetical function"
        errors = predictions - y
        updates = (X.T @ errors) / m
        theta -= alpha * updates

    theta = np.round(theta.flatten(), 4)
    return theta.tolist()


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {
        "X": np.array([[1, 1], [1, 2], [1, 3]]),
        "y": np.array([1, 2, 3]),
        "alpha": 0.01,
        "iterations": 1000,
    }
    print(inp)
    out = linear_regression_gradient_descent(**inp)
    print(f"Output: {out}")
    exp = [0.1107, 0.9513]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
