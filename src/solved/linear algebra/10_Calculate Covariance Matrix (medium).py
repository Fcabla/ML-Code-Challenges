"""
problem_id: 10
Category: linear algebra
Difficulty: medium
URL: https://www.deep-ml.com/problem/Calculate%20Covariance%20Matrix
Page: 1

==== Title ====
Calculate Covariance Matrix (medium)

==== Description ====
Write a Python function that calculates the covariance matrix from a list of vectors. Assume that the input list represents a dataset where each vector is a feature, and vectors are of equal length.

==== Example ====
Example:
        input: vectors = [[1, 2, 3], [4, 5, 6]]
        output: [[1.0, 1.0], [1.0, 1.0]]
        reasoning: The dataset has two features with three observations each. The covariance between each pair of features (including covariance with itself) is calculated and returned as a 2x2 matrix.

==== Learn More ====
Calculate Covariance Matrix

The covariance matrix is a fundamental concept in statistics, illustrating how much two random variables change together. It's essential for understanding the relationships between variables in a dataset.

For a dataset with \(n\) features, the covariance matrix is an \(n \times n\) square matrix where each element (i, j) represents the covariance between the \(i^{th}\) and \(j^{th}\) features. Covariance is defined by the formula:
\[
\text{cov}(X, Y) = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{n-1}
\]

Where:

- \(X\) and \(Y\) are two random variables (features),
- \(x_i\) and \(y_i\) are individual observations of \(X\) and \(Y\),
- \(\bar{x}\) (x-bar) and \(\bar{y}\) (y-bar) are the means of \(X\) and \(Y\),
- \(n\) is the number of observations.

In the covariance matrix:

- The diagonal elements (where \(i = j\)) indicate the variance of each feature.
- The off-diagonal elements show the covariance between different features. This matrix is symmetric, as the covariance between \(X\) and \(Y\) is equal to the covariance between \(Y\) and \(X\), denoted as \(\text{cov}(X, Y) = \text{cov}(Y, X)\).
"""


# ==== Code ====
def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    n_feat = len(vectors)
    n_obs = len(vectors[0])
    covariance_matrix = [[0 for _ in range(n_feat)] for _ in range(n_feat)]

    for i in range(n_feat):
        for j in range(n_feat):
            # calculate means
            x_bar = sum(vectors[i]) / len(vectors[i])
            y_bar = sum(vectors[j]) / len(vectors[j])
            suma = 0
            for k in range(n_obs):
                suma += (vectors[i][k] - x_bar) * (vectors[j][k] - y_bar)
            covariance_matrix[i][j] = suma / (n_obs - 1)
    return covariance_matrix


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"vectors": [[1, 2, 3], [4, 5, 6]]}
    print(inp)
    out = calculate_covariance_matrix(**inp)
    print(f"Output: {out}")
    exp = [[1.0, 1.0], [1.0, 1.0]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"vectors": [[1, 5, 6], [2, 3, 4], [7, 8, 9]]}
    print(inp)
    out = calculate_covariance_matrix(**inp)
    print(f"Output: {out}")
    exp = [[7.0, 2.5, 2.5], [2.5, 1.0, 1.0], [2.5, 1.0, 1.0]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
