"""
problem_id: 36
Category: machine learning
Difficulty: easy
URL: https://www.deep-ml.com/problem/Calculate%20Accuracy%20Score
Page: 2

==== Title ====
Calculate Accuracy Score

==== Description ====
Write a Python function to calculate the accuracy score of a model's predictions. The function should take in two 1D numpy arrays: y_true, which contains the true labels, and y_pred, which contains the predicted labels. It should return the accuracy score as a float.

==== Example ====
Example:
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 0, 1])
    output = accuracy_score(y_true, y_pred)
    print(output)
    # Output:
    # 0.8333333333333334

    Reasoning:
    The function compares the true labels with the predicted labels and calculates the ratio of correct predictions to the total number of predictions. In this example, there are 5 correct predictions out of 6, resulting in an accuracy score of 0.8333333333333334.

==== Learn More ====
Understanding Accuracy Score
Accuracy is a metric used to evaluate the performance of a classification model. It is defined as the ratio of the number of correct predictions to the total number of predictions made. Mathematically, accuracy is given by:

\[ \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} \]

In this problem, you will write a function to calculate the accuracy score given the true labels and the predicted labels. The function will compare the two arrays and compute the accuracy as the proportion of matching elements.
Accuracy is a straightforward and commonly used metric for classification tasks. It provides a quick way to understand how well a model is performing, but it may not always be the best metric, especially for imbalanced datasets.
"""

# ==== Code ====
import numpy as np


def accuracy_score(y_true, y_pred):
    # Your code here
    return sum(y_true == y_pred) / len(y_true)


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {
        "y_true": np.array([1, 0, 1, 1, 0, 1]),
        "y_pred": np.array([1, 0, 0, 1, 0, 1]),
    }
    print(inp)
    out = accuracy_score(**inp)
    print(f"Output: {out}")
    exp = 0.8333333333333334
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"y_true": np.array([1, 1, 1, 1]), "y_pred": np.array([1, 0, 1, 0])}
    print(inp)
    out = accuracy_score(**inp)
    print(f"Output: {out}")
    exp = 0.5
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
