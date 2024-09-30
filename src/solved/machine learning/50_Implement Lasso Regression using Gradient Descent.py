"""
problem_id: 50
Category: machine learning
Difficulty: medium
URL: https://www.deep-ml.com/problem/Lasso%20Regression%20using%20Gradient%20Descent
Page: 3

==== Title ====
Implement Lasso Regression using Gradient Descent

==== Description ====
In this problem, you need to implement the Lasso Regression algorithm using Gradient Descent. Lasso Regression (L1 Regularization) adds a penalty equal to the absolute value of the coefficients to the loss function. Your task is to update the weights and bias iteratively using the gradient of the loss function and the L1 penalty.
The objective function of Lasso Regression is:

\[
J(w, b) = \frac{1}{2n} \sum_{i=1}^{n} \left( y_i - \left( \sum_{j=1}^{p} X_{ij} w_j + b \right) \right)^2 + \alpha \sum_{j=1}^{p} | w_j |
\]

Where:

\(y_i\) is the actual value for the \(i\)-th sample
\(\hat{y}_i = \sum_{j=1}^{p} X_{ij} w_j + b\) is the predicted value for the \(i\)-th sample
\(w_j\) is the weight associated with the \(j\)-th feature
\(\alpha\) is the regularization parameter
\(b\) is the bias

Your task is to use the L1 penalty to shrink some of the feature coefficients to zero during gradient descent, thereby helping with feature selection.

==== Example ====
Example:
import numpy as np

X = np.array([[0, 0], [1, 1], [2, 2]])
y = np.array([0, 1, 2])

alpha = 0.1
weights, bias = l1_regularization_gradient_descent(X, y, alpha=alpha, learning_rate=0.01, max_iter=1000)

# Expected Output:
(weights,bias)
(array([float, float]), float)

==== Learn More ====
Understanding Lasso Regression and L1 Regularization
Lasso Regression is a type of linear regression that applies L1 regularization to the model. It adds a penalty equal to the sum of the absolute values of the coefficients, encouraging some of them to be exactly zero. This makes Lasso Regression particularly useful for feature selection, as it can shrink the coefficients of less important features to zero, effectively removing them from the model.
Steps to Implement Lasso Regression using Gradient Descent

Initialize Weights and Bias: Start with the weights and bias set to zero.
Make Predictions: Use the formula:
    \[
    \hat{y}_i = \sum_{j=1}^{p} X_{ij} w_j + b
    \]
    where \( \hat{y}_i \) is the predicted value for the \(i\)-th sample.
Compute Residuals: Find the difference between the actual values \( y_i \) and the predicted values \( \hat{y}_i \). These residuals are the errors in the model.
Update the Weights and Bias: Update the weights and bias using the gradient of the loss function with respect to the weights and bias:


For weights \( w_j \):
        \[
        \frac{\partial J}{\partial w_j} = \frac{1}{n} \sum_{i=1}^{n} X_{ij}(y_i - \hat{y}_i) + \alpha \cdot \text{sign}(w_j)
        \]

For bias \( b \) (without the regularization term):
        \[
        \frac{\partial J}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)
        \]

Update the weights and bias:
        \[
        w_j = w_j - \eta \cdot \frac{\partial J}{\partial w_j}
        \]
        \[
        b = b - \eta \cdot \frac{\partial J}{\partial b}
        \]



Check for Convergence: The algorithm stops when the L1 norm of the gradient with respect to the weights becomes smaller than a predefined threshold \( \text{tol} \):
    \[
    ||\nabla w ||_1 = \sum_{j=1}^{p} \left| \frac{\partial J}{\partial w_j} \right|
    \]

Return the Weights and Bias: Once the algorithm converges, return the optimized weights and bias.
"""

# ==== Code ====
import numpy as np


def l1_regularization_gradient_descent(
    X: np.array,
    y: np.array,
    alpha: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    n_samples, n_features = X.shape
    # Initialize
    weights = np.zeros(n_features)
    bias = 0
    # Your code here
    for iteration in range(max_iter):
        # Make predictions
        predictions = X @ weights + bias
        # Compute error
        errors = predictions - y
        # Gradients for weight
        gradient_weights = (1 / n_samples) * X.T @ errors + alpha * np.sign(weights)
        # Gradients for bias
        gradient_bias = (1 / n_samples) * np.sum(errors)

        # Update
        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias

        # Check for Convergence
        if np.linalg.norm(gradient_weights) < tol:
            # return weights, bias
            break
    return np.round(weights, 8).tolist(), float(bias)


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1

    print("Test Case 1:")
    inp = {
        "X": np.array([[0, 0], [1, 1], [2, 2]]),
        "y": np.array([0, 1, 2]),
        "alpha": 0.1,
        "learning_rate": 0.01,
        "max_iter": 1000,
    }
    print(inp)
    out = l1_regularization_gradient_descent(**inp)
    print(f"Output: {out}")
    exp = ([0.42371644, 0.42371644], 0.15385068459377865)
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {
        "X": np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]),
        "y": np.array([1, 2, 3, 4, 5]),
        "alpha": 0.1,
        "learning_rate": 0.01,
        "max_iter": 1000,
    }
    print(inp)
    out = l1_regularization_gradient_descent(**inp)
    print(f"Output: {out}")
    exp = ([0.27280148, 0.68108784], 0.4082863608718005)
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 3
