"""
problem_id: 37
Category: linear algebra
Difficulty: medium
URL: https://www.deep-ml.com/problem/Calculate%20Correlation%20Matrix
Page: 2

==== Title ====
Calculate Correlation Matrix

==== Description ====
Write a Python function to calculate the correlation matrix for a given dataset. The function should take in a 2D numpy array X and an optional 2D numpy array Y. If Y is not provided, the function should calculate the correlation matrix of X with itself. It should return the correlation matrix as a 2D numpy array.

==== Example ====
Example:
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])
    output = calculate_correlation_matrix(X)
    print(output)
    # Output:
    # [[1. 1.]
    #  [1. 1.]]

    Reasoning:
    The function calculates the correlation matrix for the dataset X. In this example, the correlation between the two features is 1, indicating a perfect linear relationship.

==== Learn More ====
Understanding Correlation Matrix
A correlation matrix is a table showing the correlation coefficients between variables. Each cell in the table shows the correlation between two variables. The value is between -1 and 1, indicating the strength and direction of the linear relationship between the variables.
The correlation coefficient between two variables \(X\) and \(Y\) is given by:

\[ \text{corr}(X, Y) = \frac{\text{cov}(X, Y)}{\sigma_X \sigma_Y} \]

Where:

\(\text{cov}(X, Y)\) is the covariance between \(X\) and \(Y\).
\(\sigma_X\) and \(\sigma_Y\) are the standard deviations of \(X\) and \(Y\), respectively.

In this problem, you will write a function to calculate the correlation matrix for a given dataset. The function will take in a 2D numpy array \(X\) and an optional 2D numpy array \(Y\). If \(Y\) is not provided, the function will calculate the correlation matrix of \(X\) with itself.
"""

# ==== Code ====
import numpy as np


def calculate_correlation_matrix(X, Y=None):
    # Your code here
    if Y is None:
        Y = X.copy()
    n_samples = np.shape(X)[0]
    # cov = np.cov(X, Y, rowvar=False) #uses n−1 (unbiased estimation)
    cov = (1 / n_samples) * (X - X.mean(0)).T.dot(Y - Y.mean(0))
    stdx = np.std(X, axis=0).reshape(-1, 1)
    stdy = np.std(Y, axis=0).reshape(-1, 1)
    corr = np.divide(cov, stdx.dot(stdy.T))
    return np.round(np.array(corr, dtype=float), 8).tolist()


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"X": np.array([[1, 2], [3, 4], [5, 6]]), "Y": None}
    print(inp)
    out = calculate_correlation_matrix(**inp)
    print(f"Output: {out}")
    exp = [[1.0, 1.0], [1.0, 1.0]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"X": np.array([[1, 2, 3], [7, 15, 6], [7, 8, 9]]), "Y": None}
    print(inp)
    out = calculate_correlation_matrix(**inp)
    print(f"Output: {out}")
    exp = [
        [1.0, 0.84298868, 0.8660254],
        [0.84298868, 1.0, 0.46108397],
        [0.8660254, 0.46108397, 1.0],
    ]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 3
    print("Test Case 3:")
    inp = {"X": np.array([[1, 0], [0, 1]]), "Y": np.array([[1, 2], [3, 4]])}
    print(inp)
    out = calculate_correlation_matrix(**inp)
    print(f"Output: {out}")
    exp = [[-1.0, -1.0], [1.0, 1.0]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
