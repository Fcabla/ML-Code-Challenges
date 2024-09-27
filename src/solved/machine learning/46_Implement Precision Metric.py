"""
problem_id: 46
Category: machine learning
Difficulty: easy
URL: https://www.deep-ml.com/problem/Precision%20Metric
Page: 3

==== Title ====
Implement Precision Metric

==== Description ====
Write a Python function `precision` that calculates the precision metric given two numpy arrays: `y_true` and `y_pred`. The `y_true` array contains the true binary labels, and the `y_pred` array contains the predicted binary labels. Precision is defined as the ratio of true positives to the sum of true positives and false positives.

==== Example ====
Example:
import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])

result = precision(y_true, y_pred)
print(result)
# Expected Output: 1.0

==== Learn More ====
Understanding Precision in Classification
Precision is a key metric used in the evaluation of classification models, particularly in binary classification. It provides insight into the accuracy of the positive predictions made by the model.
Mathematical Definition
Precision is defined as the ratio of true positives (TP) to the sum of true positives and false positives (FP):

\[
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
\]

Where:

True Positives (TP): The number of positive samples that are correctly identified as positive.
False Positives (FP): The number of negative samples that are incorrectly identified as positive.

Characteristics of Precision

Range: Precision ranges from 0 to 1, where 1 indicates perfect precision (no false positives) and 0 indicates no true positives.
Interpretation: High precision means that the model has a low false positive rate, meaning it rarely labels negative samples as positive.
Use Case: Precision is particularly useful when the cost of false positives is high, such as in medical diagnosis or fraud detection.

In this problem, you will implement a function to calculate precision given the true labels and predicted labels of a binary classification task.
"""

# ==== Code ====
import numpy as np


def precision(y_true, y_pred):
    # Your code here
    tp = 0
    tf = 0
    for y_t, y_p in zip(y_true, y_pred):
        if y_p == 1:
            if y_p == y_t:
                tp += 1
            else:
                tf += 1

    # Other solution
    # true_positives = np.sum((y_true == 1) & (y_pred == 1))
    # false_positives = np.sum((y_true == 0) & (y_pred == 1))
    # return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    return tp / (tp + tf)


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {
        "y_true": np.array([1, 0, 1, 1, 0, 1]),
        "y_pred": np.array([1, 0, 1, 0, 0, 1]),
    }
    print(inp)
    out = precision(**inp)
    print(f"Output: {out}")
    exp = 1.0
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 1:")
    inp = {
        "y_true": np.array([1, 0, 1, 1, 0, 0]),
        "y_pred": np.array([1, 0, 0, 0, 0, 1]),
    }
    print(inp)
    out = precision(**inp)
    print(f"Output: {out}")
    exp = 0.5
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
