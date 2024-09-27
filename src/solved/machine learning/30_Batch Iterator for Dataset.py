"""
problem_id: 30
Category: machine learning
Difficulty: easy
URL: https://www.deep-ml.com/problem/Batch%20Iterator%20for%20Dataset
Page: 2

==== Title ====
Batch Iterator for Dataset

==== Description ====
Write a Python function to create a batch iterator for the samples in a numpy array X and an optional numpy array y. The function should yield batches of a specified size. If y is provided, the function should yield batches of (X, y) pairs; otherwise, it should yield batches of X only.

==== Example ====
Example:
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8],
                  [9, 10]])
    y = np.array([1, 2, 3, 4, 5])
    batch_size = 2
    batch_iterator(X, y, batch_size)
    output:
    [[[[1, 2], [3, 4]], [1, 2]],
     [[[5, 6], [7, 8]], [3, 4]],
     [[[9, 10]], [5]]]

     Reasoning:
    The dataset X contains 5 samples, and we are using a batch size of 2. Therefore, the function will divide the dataset into 3 batches. The first two batches will contain 2 samples each, and the last batch will contain the remaining sample. The corresponding values from y are also included in each batch.

==== Learn More ====
Understanding Batch Iteration
Batch iteration is a common technique used in machine learning and data processing to handle large datasets more efficiently. Instead of processing the entire dataset at once, which can be memory-intensive, data is processed in smaller, more manageable batches.
Here's a step-by-step method to create a batch iterator:

Determine the Number of Samples: Calculate the total number of samples in the dataset.
Iterate in Batches: Loop through the dataset in increments of the specified batch size.
Yield Batches: For each iteration, yield a batch of samples from X and, if provided, the corresponding samples from y.

This method ensures efficient processing and can be used for both training and evaluation phases in machine learning workflows.
"""

# ==== Code ====
import numpy as np


def batch_iterator(X, y=None, batch_size=64):
    n_samples = len(X)
    assert batch_size < n_samples
    # samples_per_group = int(np.ceil(n_samples / batch_size))

    batches = []
    for i in range(0, n_samples, batch_size):
        if y is not None:
            batches.append(
                [X[i : i + batch_size].tolist(), y[i : i + batch_size].tolist()]
            )
        else:
            batches.append(X[i : i + batch_size].tolist())
    return batches


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {
        "X": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        "y": np.array([1, 2, 3, 4, 5]),
        "batch_size": 2,
    }
    print(inp)
    out = batch_iterator(**inp)
    print(f"Output: {out}")
    exp = [[[[1, 2], [3, 4]], [1, 2]], [[[5, 6], [7, 8]], [3, 4]], [[[9, 10]], [5]]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"X": np.array([[1, 1], [2, 2], [3, 3], [4, 4]]), "y": None, "batch_size": 3}
    print(inp)
    out = batch_iterator(**inp)
    print(f"Output: {out}")
    exp = [[[1, 1], [2, 2], [3, 3]], [[4, 4]]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
