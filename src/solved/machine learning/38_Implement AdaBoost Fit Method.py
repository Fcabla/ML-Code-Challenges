"""
problem_id: 38
Category: machine learning
Difficulty: hard
URL: https://www.deep-ml.com/problem/Implement%20AdaBoost%20Fit%20Method
Page: 2

==== Title ====
Implement AdaBoost Fit Method

==== Description ====
Write a Python function `adaboost_fit` that implements the fit method for an AdaBoost classifier. The function should take in a 2D numpy array `X` of shape `(n_samples, n_features)` representing the dataset, a 1D numpy array `y` of shape `(n_samples,)` representing the labels, and an integer `n_clf` representing the number of classifiers. The function should initialize sample weights, find the best thresholds for each feature, calculate the error, update weights, and return a list of classifiers with their parameters.

==== Example ====
Example:
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([1, 1, -1, -1])
    n_clf = 3

    clfs = adaboost_fit(X, y, n_clf)
    print(clfs)
    # Output (example format, actual values may vary):
    # [{'polarity': 1, 'threshold': 2, 'feature_index': 0, 'alpha': 0.5},
    #  {'polarity': -1, 'threshold': 3, 'feature_index': 1, 'alpha': 0.3},
    #  {'polarity': 1, 'threshold': 4, 'feature_index': 0, 'alpha': 0.2}]

==== Learn More ====
Understanding AdaBoost
AdaBoost, short for Adaptive Boosting, is an ensemble learning method that combines multiple weak classifiers to create a strong classifier. The basic idea is to fit a sequence of weak learners on weighted versions of the data.
Here's how to implement the fit method for an AdaBoost classifier:

Initialize Weights: Start by initializing the sample weights uniformly:
    \[
    w_i = \frac{1}{N}, \text{ where } N \text{ is the number of samples}
    \]

Iterate Through Classifiers: For each classifier, determine the best threshold for each feature to minimize the error.
Calculate Error and Flip Polarity: If the error is greater than 0.5, flip the polarity:
    \[
    \text{error} = \sum_{i=1}^N w_i [y_i \neq h(x_i)]
    \]
    \[
    \text{if error} > 0.5: \text{error} = 1 - \text{error}, \text{ and flip the polarity}
    \]

Calculate Alpha: Compute the weight (alpha) of the classifier based on its error rate:
    \[
    \alpha = \frac{1}{2} \ln \left( \frac{1 - \text{error}}{\text{error} + 1e-10} \right)
    \]

Update Weights: Adjust the sample weights based on the classifier's performance and normalize them:
    \[
    w_i = w_i \exp(-\alpha y_i h(x_i))
    \]
    \[
    w_i = \frac{w_i}{\sum_{j=1}^N w_j}
    \]

Save Classifier: Store the classifier with its parameters.

This method helps in focusing more on the misclassified samples in subsequent rounds, thereby improving the overall performance.


Learn Contributors

Moe Chabot
"""

import math

# ==== Code ====
import numpy as np


def adaboost_fit(X, y, n_clf):
    n_samples, n_features = np.shape(X)
    w = np.full(n_samples, (1 / n_samples))
    clfs = []

    # Your code here
    for _ in range(n_clf):
        clf = {}
        min_error = float("inf")

        for feat in range(n_features):
            # The algorithm goes through each feature in X, treating each one as a potential weak classifier (stump).
            # It gathers unique values from the feature and then tests each value as a threshold to split the data.
            feature_values = np.expand_dims(X[:, feat], axis=1)  # TO COLUMN
            unique_values = np.unique(feature_values)

            for threshold in unique_values:
                # Initialize predictions
                p = 1
                prediction = np.ones(np.shape(y))
                # If a feature value is less than the threshold, it assigns a prediction of -1 (negative class).
                prediction[X[:, feat] < threshold] = -1
                # Calc errors
                error = sum(w[y != prediction])

                # If the error is greater than 0.5, the algorithm flips the predictions (p = -1) to minimize the error, ensuring that weak classifiers are at least somewhat better than random guessing.
                if error > 0.5:
                    error = 1 - error
                    p = -1

                # The weak classifier's parameters (threshold, feature index, polarity) are updated whenever a better threshold with a lower error is found.
                if error < min_error:
                    clf["polarity"] = p
                    clf["threshold"] = threshold
                    clf["feature_index"] = feat
                    min_error = error

        # The classifier's weight alpha is calculated for the best weak classifier for this iteration
        # This alpha reflects the importance of the weak classifier based on its performance (lower error gives a higher weight).

        clf["alpha"] = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))
        predictions = np.ones(np.shape(y))
        negative_idx = (
            clf["polarity"] * X[:, clf["feature_index"]]
            < clf["polarity"] * clf["threshold"]
        )
        predictions[negative_idx] = -1
        # The sample weights are updated based on the predictions of the current weak classifier:
        # This increases the weights of the misclassified samples and decreases the weights of correctly classified ones, so the next weak classifier will focus more on hard-to-classify samples.
        w *= np.exp(-clf["alpha"] * y * predictions)
        w /= np.sum(w)  # normalize
        clfs.append(clf)
    return clfs


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {
        "X": np.array([[1, 2], [2, 3], [3, 4], [4, 5]]),
        "y": np.array([1, 1, -1, -1]),
        "n_clf": 3,
    }
    print(inp)
    out = adaboost_fit(**inp)
    print(f"Output: {out}")
    exp = [
        {
            "polarity": -1,
            "threshold": 3,
            "feature_index": 0,
            "alpha": 11.512925464970229,
        },
        {
            "polarity": -1,
            "threshold": 3,
            "feature_index": 0,
            "alpha": 11.512924909859024,
        },
        {
            "polarity": -1,
            "threshold": 1,
            "feature_index": 0,
            "alpha": 11.512925464970229,
        },
    ]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {
        "X": np.array(
            [
                [8, 7],
                [3, 4],
                [5, 9],
                [4, 0],
                [1, 0],
                [0, 7],
                [3, 8],
                [4, 2],
                [6, 8],
                [0, 2],
            ]
        ),
        "y": np.array([1, -1, 1, -1, 1, -1, -1, -1, 1, 1]),
        "n_clf": 2,
    }
    print(inp)
    out = adaboost_fit(**inp)
    print(f"Output: {out}")
    exp = [
        {
            "polarity": 1,
            "threshold": 5,
            "feature_index": 0,
            "alpha": 0.6931471803099453,
        },
        {
            "polarity": -1,
            "threshold": 3,
            "feature_index": 0,
            "alpha": 0.5493061439673882,
        },
    ]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
