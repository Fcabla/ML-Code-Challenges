"""
problem_id: 31
Category: machine learning
Difficulty: medium
URL: https://www.deep-ml.com/problem/Divide%20Dataset%20Based%20on%20Feature%20Threshold
Page: 2

==== Title ====
Divide Dataset Based on Feature Threshold

==== Description ====
Write a Python function to divide a dataset based on whether the value of a specified feature is greater than or equal to a given threshold. The function should return two subsets of the dataset: one with samples that meet the condition and another with samples that do not.

==== Example ====
Example:
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8],
                  [9, 10]])
    feature_i = 0
    threshold = 5
    output: [array([[ 5,  6],
                    [ 7,  8],
                    [ 9, 10]]),
             array([[1, 2],
                    [3, 4]])]
    Reasoning:
    The dataset X is divided based on whether the value in the 0th feature (first column) is greater than or equal to 5. Samples with the first column value >= 5 are in the first subset, and the rest are in the second subset.

==== Learn More ====
Understanding Dataset Division Based on Feature Threshold
Dividing a dataset based on a feature threshold is a common operation in machine learning, especially in decision tree algorithms. This technique helps in creating splits that can be used for further processing or model training.
In this problem, you will write a function to split a dataset based on whether the value of a specified feature is greater than or equal to a given threshold. You'll need to create two subsets: one for samples that meet the condition and another for samples that do not.
This method is crucial for algorithms that rely on data partitioning, such as decision trees and random forests. By splitting the data, the model can create rules to make predictions based on the threshold values of certain features.
"""

# ==== Code ====
import numpy as np


def divide_on_feature(X, feature_i, threshold):
    X = X.T
    yes = X[:, X[feature_i] >= threshold].T
    no = X[:, X[feature_i] < threshold].T
    return [yes.tolist(), no.tolist()]
    return [yes, no]


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {
        "X": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        "feature_i": 0,
        "threshold": 5,
    }
    print(inp)
    out = divide_on_feature(**inp)
    print(f"Output: {out}")
    exp = [np.array([[5, 6], [7, 8], [9, 10]]), np.array([[1, 2], [3, 4]])]
    exp = [el.tolist() for el in exp]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {
        "X": np.array([[1, 1], [2, 2], [3, 3], [4, 4]]),
        "feature_i": 1,
        "threshold": 3,
    }
    print(inp)
    out = divide_on_feature(**inp)
    print(f"Output: {out}")
    exp = [np.array([[3, 3], [4, 4]]), np.array([[1, 1], [2, 2]])]
    exp = [el.tolist() for el in exp]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
    # Test case 3
