"""
problem_id: 45
Category: machine learning
Difficulty: easy
URL: https://www.deep-ml.com/problem/Linear%20Kernel
Page: 2

==== Title ====
Linear Kernel Function

==== Description ====
Write a Python function `kernel_function` that computes the linear kernel between two input vectors `x1` and `x2`. The linear kernel is defined as the dot product (inner product) of two vectors.

==== Example ====
Example:
import numpy as np

x1 = np.array([1, 2, 3])
x2 = np.array([4, 5, 6])

result = kernel_function(x1, x2)
print(result)
# Expected Output: 32

==== Learn More ====
Understanding the Linear Kernel
A kernel function in machine learning is used to measure the similarity between two data points in a higher-dimensional space without having to compute the coordinates of the points in that space explicitly. The linear kernel is one of the simplest and most commonly used kernel functions. It computes the dot product (or inner product) of two vectors.
Mathematical Definition
The linear kernel between two vectors \( \mathbf{x}_1 \) and \( \mathbf{x}_2 \) is mathematically defined as:

\[
K(\mathbf{x}_1, \mathbf{x}_2) = \mathbf{x}_1 \cdot \mathbf{x}_2 = \sum_{i=1}^{n} x_{1,i} \cdot x_{2,i}
\]

Where \( n \) is the number of features, and \( x_{1,i} \) and \( x_{2,i} \) are the components of the vectors \( \mathbf{x}_1 \) and \( \mathbf{x}_2 \) respectively.
The linear kernel is widely used in support vector machines (SVMs) and other machine learning algorithms for linear classification and regression tasks. It is computationally efficient and works well when the data is linearly separable.
Characteristics

Simplicity: The linear kernel is straightforward to implement and compute.
Efficiency: It is computationally less expensive compared to other complex kernels like polynomial or RBF kernels.
Interpretability: The linear kernel is interpretable because it corresponds directly to the dot product, a well-understood operation in vector algebra.

In this problem, you will implement a function capable of computing the linear kernel between two vectors.
"""

# ==== Code ====
import numpy as np


def kernel_function(x1, x2):
    # Your code here
    # return np.inner(x1, x2)
    return sum(x1 * x2)


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"x1": np.array([1, 2, 3]), "x2": np.array([4, 5, 6])}
    print(inp)
    out = kernel_function(**inp)
    print(f"Output: {out}")
    exp = 32
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"x1": np.array([0, 1, 2]), "x2": np.array([3, 4, 5])}
    print(inp)
    out = kernel_function(**inp)
    print(f"Output: {out}")
    exp = 14
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
