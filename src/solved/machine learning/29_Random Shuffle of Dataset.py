"""
problem_id: 29
Category: machine learning
Difficulty: easy
URL: https://www.deep-ml.com/problem/Random%20Shuffle%20of%20Dataset
Page: 2

==== Title ====
Random Shuffle of Dataset

==== Description ====
Write a Python function to perform a random shuffle of the samples in two numpy arrays, X and y, while maintaining the corresponding order between them. The function should have an optional seed parameter for reproducibility.

==== Example ====
Example:
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8]])
    y = np.array([1, 2, 3, 4])
    output: (array([[5, 6],
                    [1, 2],
                    [7, 8],
                    [3, 4]]),
             array([3, 1, 4, 2]))

==== Learn More ====
Understanding Dataset Shuffling
Random shuffling of a dataset is a common preprocessing step in machine learning to ensure that the data is randomly distributed before training a model. This helps to avoid any potential biases that may arise from the order in which data is presented to the model.
Here's a step-by-step method to shuffle a dataset:

Generate a Random Index Array: Create an array of indices corresponding to the number of samples in the dataset.
Shuffle the Indices: Use a random number generator to shuffle the array of indices.
Reorder the Dataset: Use the shuffled indices to reorder the samples in both X and y.

This method ensures that the correspondence between X and y is maintained after shuffling.
"""

# ==== Code ====
import numpy as np


def shuffle_data(X, y, seed=None):
    # Your code here
    if seed:
        np.random.seed(seed)
    assert len(X) == len(y)
    idxs = np.arange(len(X))
    np.random.shuffle(idxs)
    X = X[idxs]
    y = y[idxs]

    # Other solution:
    # y = y.reshape(-1,1)
    # X = np.append(X, y, axis=1)
    # np.random.shuffle(X)
    # y = X[:,-1]
    # X = np.delete(X, -1, axis=1)
    return X.tolist(), y.tolist()
    return X, y


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {
        "X": np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
        "y": np.array([1, 2, 3, 4]),
        "seed": 42,
    }
    print(inp)
    out = shuffle_data(**inp)
    print(f"Output: {out}")
    # exp = (np.array([[3, 4], [7, 8], [1, 2], [5, 6]]), np.array([2, 4, 1, 3]))
    exp = ([[3, 4], [7, 8], [1, 2], [5, 6]], [2, 4, 1, 3])
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {
        "X": np.array([[1, 1], [2, 2], [3, 3], [4, 4]]),
        "y": np.array([10, 20, 30, 40]),
        "seed": 24,
    }
    print(inp)
    out = shuffle_data(**inp)
    print(f"Output: {out}")
    # exp = (np.array([[4, 4], [2, 2], [1, 1], [3, 3]]), np.array([40, 20, 10, 30]))
    exp = ([[4, 4], [2, 2], [1, 1], [3, 3]], [40, 20, 10, 30])
    print(f"Expected: {exp}")
    # print("Accepted" if np.array_equal(out, exp) else "Error")
    print("Accepted" if out == exp else "Error")
    print("---")
