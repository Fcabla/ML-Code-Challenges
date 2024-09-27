"""
problem_id: 39
Category: deep learning
Difficulty: easy
URL: https://www.deep-ml.com/problem/Log%20Softmax
Page: 2

==== Title ====
Implementation of Log Softmax Function

==== Description ====
In machine learning and statistics, the softmax function is a generalization of the logistic function that converts a vector of scores into probabilities. The log-softmax function is the logarithm of the softmax function, and it is often used for numerical stability when computing the softmax of large numbers.

Given a 1D numpy array of scores, implement a Python function to compute the log-softmax of the array.

==== Example ====
Example:
A = np.array([1, 2, 3])
print(log_softmax(A))

Output:
array([-2.4076, -1.4076, -0.4076])

==== Learn More ====
Understanding Log Softmax Function
The log softmax function is a numerically stable way of calculating the logarithm of the softmax function. The softmax function converts a vector of arbitrary values (logits) into a vector of probabilities, where each value lies between 0 and 1, and the values sum to 1. The softmax function is given by:

\[
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}
\]

However, directly applying the logarithm to the softmax function can lead to numerical instability, especially when dealing with large numbers. To prevent this, we use the log-softmax function, which incorporates a shift by subtracting the maximum value from the input vector:

\[
\text{log softmax}(x_i) = x_i - \max(x) - \log\left(\sum_{j=1}^n e^{x_j - \max(x)}\right)
\]

This formulation helps to avoid overflow issues that can occur when exponentiating large numbers. The log-softmax function is particularly useful in machine learning for calculating probabilities in a stable manner, especially when used with cross-entropy loss functions.
"""

import math

# ==== Code ====
import numpy as np


def log_softmax(scores: list) -> np.ndarray:
    # Your code here
    # Get the maximum score
    ma = max(scores)
    # Calculate the log term
    lo = sum([math.exp(score - ma) for score in scores])
    lo = math.log(lo)
    # Initialize result obj
    result = np.zeros(len(scores))
    # Calculate for each elem
    for i in range(len(scores)):
        result[i] = scores[i] - ma - lo
    return np.round(result, 4).tolist()


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"scores": [1, 2, 3]}
    print(inp)
    out = log_softmax(**inp)
    print(f"Output: {out}")
    exp = [-2.4076, -1.4076, -0.4076]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"scores": [1, 1, 1]}
    print(inp)
    out = log_softmax(**inp)
    print(f"Output: {out}")
    exp = [-1.0986, -1.0986, -1.0986]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 3
    print("Test Case 3:")
    inp = {"scores": [1, 1, 0.0000001]}
    print(inp)
    out = log_softmax(**inp)
    print(f"Output: {out}")
    exp = [-0.862, -0.862, -1.862]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
