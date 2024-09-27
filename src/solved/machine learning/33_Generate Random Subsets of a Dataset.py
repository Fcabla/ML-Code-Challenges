"""
problem_id: 33
Category: machine learning
Difficulty: medium
URL: https://www.deep-ml.com/problem/Generate%20Random%20Subsets%20of%20a%20Dataset
Page: 2

==== Title ====
Generate Random Subsets of a Dataset

==== Description ====
Write a Python function to generate random subsets of a given dataset. The function should take in a 2D numpy array X, a 1D numpy array y, an integer n_subsets, and a boolean replacements. It should return a list of n_subsets random subsets of the dataset, where each subset is a tuple of (X_subset, y_subset). If replacements is True, the subsets should be created with replacements; otherwise, without replacements.

==== Example ====
Example:
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8],
                  [9, 10]])
    y = np.array([1, 2, 3, 4, 5])
    n_subsets = 3
    replacements = False
    get_random_subsets(X, y, n_subsets, replacements)

    Output:
    [array([[7, 8],
            [1, 2]]),
     array([4, 1])]

    [array([[9, 10],
            [5, 6]]),
     array([5, 3])]

    [array([[3, 4],
            [5, 6]]),
     array([2, 3])]

    Reasoning:
    The function generates three random subsets of the dataset without replacements.
    Each subset includes 50% of the samples (since replacements=False). The samples
    are randomly selected without duplication.

==== Learn More ====
Understanding Random Subsets of a Dataset
Generating random subsets of a dataset is a useful technique in machine learning, particularly in ensemble methods like bagging and random forests. By creating random subsets, models can be trained on different parts of the data, which helps in reducing overfitting and improving generalization.
In this problem, you will write a function to generate random subsets of a given dataset. Given a 2D numpy array X, a 1D numpy array y, an integer n_subsets, and a boolean replacements, the function will create a list of n_subsets random subsets. Each subset will be a tuple of (X_subset, y_subset).
If replacements is True, the subsets will be created with replacements, meaning that samples can be repeated in a subset. If replacements is False, the subsets will be created without replacements, meaning that samples cannot be repeated within a subset.
By understanding and implementing this technique, you can enhance the performance of your models through techniques like bootstrapping and ensemble learning.
"""

# ==== Code ====
import numpy as np


def get_random_subsets(X, y, n_subsets, replacements=True, seed=42):
    np.random.seed(seed)
    result = []
    n_samples = len(X) if replacements else len(X) // 2
    for _ in range(n_subsets):
        idxs = np.random.choice(len(X), n_samples, replace=replacements)
        result.append((X[idxs].tolist(), y[idxs].tolist()))
    return result


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {
        "X": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        "y": np.array([1, 2, 3, 4, 5]),
        "n_subsets": 3,
        "replacements": False,
        "seed": 42,
    }
    print(inp)
    out = get_random_subsets(**inp)
    print(f"Output: {out}")
    # exp = [
    #    [[3, 4], [9, 10]],
    #    [2, 5],
    #    [[7, 8], [3, 4]],
    #    [4, 2],
    #    [[3, 4], [1, 2]],
    #    [2, 1],
    # ]
    exp = [
        ([[3, 4], [9, 10]], [2, 5]),
        ([[7, 8], [3, 4]], [4, 2]),
        ([[3, 4], [1, 2]], [2, 1]),
    ]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 1:")
    inp = {
        "X": np.array([[1, 1], [2, 2], [3, 3], [4, 4]]),
        "y": np.array([10, 20, 30, 40]),
        "n_subsets": 1,
        "replacements": True,
        "seed": 42,
    }
    print(inp)
    out = get_random_subsets(**inp)
    print(f"Output: {out}")
    # exp = [[[[3, 3], [4, 4], [1, 1], [3, 3]], [30, 40, 10, 30]]]
    exp = [([[3, 3], [4, 4], [1, 1], [3, 3]], [30, 40, 10, 30])]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
