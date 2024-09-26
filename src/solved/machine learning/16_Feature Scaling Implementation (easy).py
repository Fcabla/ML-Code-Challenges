"""
problem_id: 16
Category: machine learning
Difficulty: easy
URL: https://www.deep-ml.com/problem/Feature%20Scaling%20Implementation
Page: 1

==== Title ====
Feature Scaling Implementation (easy)

==== Description ====
Write a Python function that performs feature scaling on a dataset using both standardization and min-max normalization. The function should take a 2D NumPy array as input, where each row represents a data sample and each column represents a feature. It should return two 2D NumPy arrays: one scaled by standardization and one by min-max normalization. Make sure all results are rounded to the nearest 4th decimal.

==== Example ====
Example:
        input: data = np.array([[1, 2], [3, 4], [5, 6]])
        output: ([[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]], [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        reasoning: Standardization rescales the feature to have a mean of 0 and a standard deviation of 1.
        Min-max normalization rescales the feature to a range of [0, 1], where the minimum feature value
        maps to 0 and the maximum to 1.

==== Learn More ====
Feature Scaling Techniques

Feature scaling is crucial in many machine learning algorithms that are sensitive to the magnitude of features. This includes algorithms that use distance measures like k-nearest neighbors and gradient descent-based algorithms like linear regression.

 Standardization:
Standardization (or Z-score normalization) is the process where the features are rescaled so that they have the properties of a standard normal distribution with a mean of zero and a standard deviation of one:
\[
z = \frac{(x - \mu)}{\sigma}
\]
Where \(x\) is the original feature, \(\mu\) is the mean of that feature, and \(\sigma\) is the standard deviation.

 Min-Max Normalization:
Min-max normalization rescales the feature to a fixed range, typically 0 to 1, or it can be shifted to any range \([a, b]\) by transforming the data according to the formula:
\[
x' = \frac{(x - \text{min}(x))}{(\text{max}(x) - \text{min}(x))} \times (\text{max} - \text{min}) + \text{min}
\]
Where \(x\) is the original value, \(\text{min}(x)\) is the minimum value for that feature, \(\text{max}(x)\) is the maximum value, and \(\text{min}\) and \(\text{max}\) are the new minimum and maximum values for the scaled data.

Implementing these scaling techniques will ensure that the features contribute equally to the development of the model and improve the convergence speed of learning algorithms.
"""

# ==== Code ====
import numpy as np


def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    # Your code here
    standardized_data = (data - data.mean(axis=0)) / (data.std(axis=0))

    min_x = np.min(data, axis=0)
    max_x = np.max(data, axis=0)
    min, max = 0, 1
    normalized_data = ((data - min_x) / (max_x - min_x)) / ((max - min) + min)

    # Shorter and probably better solution
    # normalized_data = data / np.linalg.norm(data)

    return (
        np.round(standardized_data, 4).tolist(),
        np.round(normalized_data, 4).tolist(),
    )


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"data": np.array([[1, 2], [3, 4], [5, 6]])}
    print(inp)
    out = feature_scaling(**inp)
    print(f"Output: {out}")
    exp = (
        [[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]],
        [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
    )
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
