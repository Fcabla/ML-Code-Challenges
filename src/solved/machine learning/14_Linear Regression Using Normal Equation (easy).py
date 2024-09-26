"""
problem_id: 14
Category: machine learning
Difficulty: easy
URL: https://www.deep-ml.com/problem/Linear%20Regression%20Using%20Normal%20Equation
Page: 1

==== Title ====
Linear Regression Using Normal Equation (easy)

==== Description ====
Write a Python function that performs linear regression using the normal equation. The function should take a matrix X (features) and a vector y (target) as input, and return the coefficients of the linear regression model. Round your answer to four decimal places, -0.0 is a valid result for rounding a very small number.

==== Example ====
Example:
        input: X = [[1, 1], [1, 2], [1, 3]], y = [1, 2, 3]
        output: [0.0, 1.0]
        reasoning: The linear model is y = 0.0 + 1.0*x, perfectly fitting the input data.

==== Learn More ====
Linear Regression Using the Normal Equation

Linear regression aims to model the relationship between a scalar dependent variable \(y\) and one or more explanatory variables (or independent variables) \(X\). The normal equation provides an analytical solution to finding the coefficients \(\theta\) that minimize the cost function for linear regression.

Given a matrix \(X\) (with each row representing a training example and each column a feature) and a vector \(y\) (representing the target values), the normal equation is:
\[
\theta = (X^TX)^{-1}X^Ty
\]

Where:

 \(X^T\) is the transpose of \(X\),
 \((X^TX)^{-1}\) is the inverse of the matrix \(X^TX\),
 \(y\) is the vector of target values.


**Things to note**: This method does not require any feature scaling, and there's no need to choose a learning rate. However, computing the inverse of \(X^TX\) can be computationally expensive if the number of features is very large.

 Practical Implementation

A practical implementation involves augmenting \(X\) with a column of ones to account for the intercept term and then applying the normal equation directly to compute \(\theta\).
"""

# ==== Code ====
import numpy as np


def linear_regression_normal_equation(
    X: list[list[float]], y: list[float]
) -> list[float]:
    # Your code here, make sure to round
    # (X^T*X)^-1*X^T*y
    X_t = np.transpose(X)
    inverse = np.linalg.inv(X_t @ X)
    theta = np.round(inverse @ X_t @ y, 4)
    return theta.tolist()


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"X": [[1, 1], [1, 2], [1, 3]], "y": [1, 2, 3]}
    print(inp)
    out = linear_regression_normal_equation(**inp)
    print(f"Output: {out}")
    exp = [-0.0, 1.0]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"X": [[1, 3, 4], [1, 2, 5], [1, 3, 2]], "y": [1, 2, 1]}
    print(inp)
    out = linear_regression_normal_equation(**inp)
    print(f"Output: {out}")
    exp = [4.0, -1.0, -0.0]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 3
