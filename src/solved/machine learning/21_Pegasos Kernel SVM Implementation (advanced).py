"""
problem_id: 21
Category: machine learning
Difficulty: advanced
URL: https://www.deep-ml.com/problem/Pegasos%20Kernel%20SVM%20Implementation
Page: 1

==== Title ====
Pegasos Kernel SVM Implementation (advanced)

==== Description ====
Write a Python function that implements the Pegasos algorithm to train a kernel SVM classifier from scratch. The function should take a dataset (as a 2D NumPy array where each row represents a data sample and each column represents a feature), a label vector (1D NumPy array where each entry corresponds to the label of the sample), and training parameters such as the choice of kernel (linear or RBF), regularization parameter (lambda), and the number of iterations. The function should perform binary classification and return the model's alpha coefficients and bias.

==== Example ====
Example:
        input: data = np.array([[1, 2], [2, 3], [3, 1], [4, 1]]), labels = np.array([1, 1, -1, -1]), kernel = 'rbf', lambda_val = 0.01, iterations = 100
        output: alpha = [0.03, 0.02, 0.05, 0.01], b = -0.05
        reasoning: Using the RBF kernel, the Pegasos algorithm iteratively updates the weights based on a sub-gradient descent method, taking into account the non-linear separability of the data induced by the kernel transformation.

==== Learn More ====
Pegasos Algorithm and Kernel SVM

The Pegasos (Primal Estimated sub-GrAdient SOlver for SVM) algorithm is a simple and efficient stochastic gradient descent method designed for solving the SVM optimization problem in its primal form.

 Key Concepts:

Kernel Trick: Allows SVM to classify data that is not linearly separable by implicitly mapping input features into high-dimensional feature spaces.
Regularization Parameter (\(\lambda\)): Controls the trade-off between achieving a low training error and a low model complexity.
 Sub-Gradient Descent: Used in the Pegasos algorithm to optimize the objective function, which includes both the hinge loss and a regularization term.

 Steps in the Pegasos Algorithm:

1. **Initialize Parameters**: Start with zero weights and choose an appropriate value for the regularization parameter \( \lambda \).
2. **Iterative Updates**: For each iteration and for each randomly selected example, update the model parameters using the learning rule derived from the sub-gradient of the loss function.
3. **Kernel Adaptation**: Use the chosen kernel to compute the dot products required in the update step, allowing for non-linear decision boundaries.

 Practical Implementation:

The implementation involves selecting a kernel function, calculating the kernel matrix, and performing iterative updates on the alpha coefficients according to the Pegasos rule:
\[
\alpha_{t+1} = (1 - \eta_t \lambda) \alpha_t + \eta_t (y_i K(x_i, x))
\]
where \( \eta_t \) is the learning rate at iteration \( t \), and \( K \) denotes the kernel function.

This method is particularly well-suited for large-scale learning problems due to its efficient use of data and incremental learning nature.
"""

# ==== Code ====
import numpy as np


def pegasos_kernel_svm(
    data: np.ndarray,
    labels: np.ndarray,
    kernel="linear",
    lambda_val=0.01,
    iterations=100,
    sigma=1.0,
) -> (list, float):
    # Your code here
    def linear_kernel(x, y):
        return np.dot(x, y)

    def rbf_kernel(x, y, sigma=1.0):
        return np.exp(-(np.linalg.norm(x - y) ** 2) / (2 * (sigma**2)))

    # initialize weights
    n_samples = len(data)
    alphas = np.zeros(n_samples)
    b = 0
    # kernel_function = None
    # if kernel == "linear":
    #    kernel_function = linear_kernel
    # elif kernel == "rbf":
    #    kernel_function = lambda x, y: rbf_kernel(x, y, sigma)
    # else:
    #    return -1

    for iteration in range(1, iterations + 1):
        eta_t = 1 / (lambda_val * iteration)
        # For each sample (assume they are randomly sorted)
        for i in range(n_samples):
            decision = 0
            for j in range(n_samples):
                # kernel_product = kernel_function(data[j], data[i])
                if kernel == "linear":
                    kernel_product = linear_kernel(data[j], data[i])
                elif kernel == "rbf":
                    kernel_product = rbf_kernel(data[j], data[i], sigma)
                else:
                    return -1
                decision += alphas[j] * labels[j] * kernel_product
            decision += b
            if labels[i] * decision < 1:
                # (misclassification or within margin)
                alphas[i] += eta_t * (labels[i] - lambda_val * alphas[i])
                b += eta_t * labels[i]
    return np.round(alphas, 4).tolist(), np.round(b, 4)


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {
        "data": np.array([[1, 2], [2, 3], [3, 1], [4, 1]]),
        "labels": np.array([1, 1, -1, -1]),
        "kernel": "linear",
        "lambda_val": 0.01,
        "iterations": 100,
    }
    print(inp)
    out = pegasos_kernel_svm(**inp)
    print(f"Output: {out}")
    exp = ([100.0, 0.0, -100.0, -100.0], -937.4755)
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {
        "data": np.array([[1, 2], [2, 3], [3, 1], [4, 1]]),
        "labels": np.array([1, 1, -1, -1]),
        "kernel": "rbf",
        "lambda_val": 0.01,
        "iterations": 100,
        "sigma": 0.5,
    }
    print(inp)
    out = pegasos_kernel_svm(**inp)
    print(f"Output: {out}")
    exp = ([100.0, 99.0, -100.0, -100.0], -115.0)
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
