"""
problem_id: 43
Category: machine learning
Difficulty: easy
URL: https://www.deep-ml.com/problem/Ridge%20Regression%20Loss
Page: 2

==== Title ====
Implement Ridge Regression Loss Function

==== Description ====
Write a Python function `ridge_loss` that implements the Ridge Regression loss function. The function should take a 2D numpy array `X` representing the feature matrix, a 1D numpy array `w` representing the coefficients, a 1D numpy array `y_true` representing the true labels, and a float `alpha` representing the regularization parameter. The function should return the Ridge loss, which combines the Mean Squared Error (MSE) and a regularization term.

==== Example ====
Example:
import numpy as np

X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
w = np.array([0.2, 2])
y_true = np.array([2, 3, 4, 5])
alpha = 0.1

loss = ridge_loss(X, w, y_true, alpha)
print(loss)
# Expected Output: 2.204

==== Learn More ====
Ridge Regression Loss
Ridge Regression is a linear regression method with a regularization term to prevent overfitting by controlling the size of the coefficients.
Key Concepts:

Regularization: Adds a penalty to the loss function to discourage large coefficients, helping to generalize the model.
Mean Squared Error (MSE): Measures the average squared difference between actual and predicted values.
Penalty Term: The sum of the squared coefficients, scaled by the regularization parameter \( \lambda \), which controls the strength of the regularization.

Ridge Loss Function:
The Ridge Loss function combines MSE and the penalty term:

\[
L(\beta) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2
\]

Implementation Steps:

Calculate MSE: Compute the average squared difference between actual and predicted values.
Add Regularization Term: Compute the sum of squared coefficients multiplied by \( \lambda \).
Combine and Minimize: Sum MSE and the regularization term to form the Ridge loss, then minimize this loss to find the optimal coefficients.
"""

# ==== Code ====
import numpy as np


def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    # Your code here
    # Input through weights
    X_w = X * w
    # Get prediction
    y_pred = np.sum(X_w, axis=1)
    # Calculate mean square error
    mse = np.mean(np.power(y_true - y_pred, 2))
    # Ridge/regularization term
    reg = alpha * np.sum(np.power(w, 2))
    # Calculate final loss
    loss = mse + reg
    return loss


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {
        "X": np.array([[1, 1], [2, 1], [3, 1], [4, 1]]),
        "w": np.array([0.2, 2]),
        "y_true": np.array([2, 3, 4, 5]),
        "alpha": 0.1,
    }
    print(inp)
    out = ridge_loss(**inp)
    print(f"Output: {out}")
    exp = 2.204
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {
        "X": np.array([[1, 1, 4], [2, 1, 2], [3, 1, 0.1], [4, 1, 1.2], [1, 2, 3]]),
        "w": np.array([0.2, 2, 5]),
        "y_true": np.array([2, 3, 4, 5, 2]),
        "alpha": 0.1,
    }
    print(inp)
    out = ridge_loss(**inp)
    print(f"Output: {out}")
    exp = 164.402
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
