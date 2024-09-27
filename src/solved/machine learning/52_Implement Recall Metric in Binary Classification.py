"""
problem_id: 52
Category: machine learning
Difficulty: easy
URL: https://www.deep-ml.com/problem/Recall
Page: 3

==== Title ====
Implement Recall Metric in Binary Classification

==== Description ====
Task: Implement Recall in Binary Classification
Your task is to implement the recall metric in a binary classification setting. Recall is a performance measure that evaluates how effectively a machine learning model identifies positive instances from all the actual positive cases in a dataset.
You need to write a function recall(y_true, y_pred) that calculates the recall metric. The function should accept two inputs
Your function should return the recall value rounded to three decimal places. If the denominator (TP + FN) is zero, the recall should be 0.0 to avoid division by zero.

==== Example ====
Example:
import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])

print(recall(y_true, y_pred))

# Expected Output:
# 0.75

==== Learn More ====
Understanding Recall in Classification
Recall is a metric that measures how often a machine learning model correctly identifies positive instances, also called true positives, from all the actual positive samples in the dataset.
Mathematical Definition
Recall, also known as sensitivity, is the fraction of relevant instances that were retrieved and it's calculated using the following equation:

\[
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
\]

Where:

True Positives (TP): The number of positive samples that are correctly identified as positive.
False Negatives (FN): The number of positive samples that are incorrectly identified as negative.

In this problem, you will implement a function to calculate recall given the true labels and predicted labels of a binary classification task. The results should be rounded to three decimal places.
"""

# ==== Code ====
import numpy as np


def recall(y_true, y_pred):
    tp = 0
    fn = 0
    for y_t, y_p in zip(y_true, y_pred):
        if y_p == y_t and y_p == 1:
            tp += 1
        if y_p != y_t and y_t == 1:
            fn += 1
    return round(tp / (tp + fn), 3)
    # Other solution
    # tp = np.sum((y_true == 1) & (y_pred == 1))
    # fn = np.sum((y_true == 1) & (y_pred == 0))
    # try:
    #    return round(tp / (tp + fn), 3)
    # except ZeroDivisionError:
    #    return 0.0


# ==== Test cases ====
if __name__ == "__main__":
    print("Test Case 1:")
    inp = {
        "y_true": np.array([1, 0, 1, 1, 0, 1]),
        "y_pred": np.array([1, 0, 1, 0, 0, 1]),
    }
    print(inp)
    out = recall(**inp)
    print(f"Output: {out}")
    exp = 0.75
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {
        "y_true": np.array([1, 0, 1, 1, 0, 0]),
        "y_pred": np.array([1, 0, 0, 0, 0, 1]),
    }
    print(inp)
    out = recall(**inp)
    print(f"Output: {out}")
    exp = 0.333
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 3
    print("Test Case 3:")
    inp = {
        "y_true": np.array([1, 0, 1, 1, 0, 0]),
        "y_pred": np.array([1, 0, 1, 1, 0, 0]),
    }
    print(inp)
    out = recall(**inp)
    print(f"Output: {out}")
    exp = 1.0
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 4
    print("Test Case 4:")
    inp = {
        "y_true": np.array([1, 0, 1, 1, 0, 1]),
        "y_pred": np.array([0, 0, 0, 1, 0, 1]),
    }
    print(inp)
    out = recall(**inp)
    print(f"Output: {out}")
    exp = 0.5
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 5
    print("Test Case 4:")
    inp = {
        "y_true": np.array([1, 0, 1, 1, 0, 1]),
        "y_pred": np.array([0, 1, 0, 0, 1, 0]),
    }
    print(inp)
    out = recall(**inp)
    print(f"Output: {out}")
    exp = 0.0
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
