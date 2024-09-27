"""
problem_id: 24
Category: deep learning
Difficulty: easy
URL: https://www.deep-ml.com/problem/Single%20Neuron
Page: 1

==== Title ====
Single Neuron (easy)

==== Description ====
Write a Python function that simulates a single neuron with a sigmoid activation function for binary classification, handling multidimensional input features. The function should take a list of feature vectors (each vector representing multiple features for an example), associated true binary labels, and the neuron's weights (one for each feature) and bias as input. It should return the predicted probabilities after sigmoid activation and the mean squared error between the predicted probabilities and the true labels, both rounded to four decimal places.

==== Example ====
Example:
        input: features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]], labels = [0, 1, 0], weights = [0.7, -0.4], bias = -0.1
        output: ([0.4626, 0.4134, 0.6682], 0.3349)
        reasoning: For each input vector, the weighted sum is calculated by multiplying each feature by its corresponding weight, adding these up along with the bias, then applying the sigmoid function to produce a probability. The MSE is calculated as the average squared difference between each predicted probability and the corresponding true label.

==== Learn More ====
Single Neuron Model with Multidimensional Input and Sigmoid Activation

This task involves a neuron model designed for binary classification with multidimensional input features, using the sigmoid activation function to output probabilities. It also involves calculating the mean squared error (MSE) to evaluate prediction accuracy.

Mathematical Background

Neuron Output Calculation:
        \[
        z = \sum (weight_i \times feature_i) + bias
        \]
        \[
        \sigma(z) = \frac{1}{1 + e^{-z}}
        \]

MSE Calculation:
        \[
        MSE = \frac{1}{n} \sum (predicted - true)^2
        \]
        Where:

\(z\) is the sum of weighted inputs plus bias,
\(\sigma(z)\) is the sigmoid activation output,
\(predicted\) are the probabilities after sigmoid activation,
\(true\) are the true binary labels.



Practical Implementation

Each feature vector is processed to calculate a combined weighted sum, which is then passed through the sigmoid function to determine the probability of the input belonging to the positive class.
MSE provides a measure of error, offering insights into the model's performance and aiding in its optimization.
"""

# ==== Code ====
import math


def single_neuron_model(
    features: list[list[float]], labels: list[int], weights: list[float], bias: float
) -> (list[float], float):
    # Your code here
    def sigmoid(num: float) -> float:
        return 1 / (1 + math.exp(-num))

    probabilities = []

    # For each input
    for feature in features:
        z = 0
        # Multiply each feature element with the weight and add the bias
        for feat, weight in zip(feature, weights):
            z += weight * feat
        z += bias
        # Calculate the sigmoid
        probabilities.append(round(sigmoid(z), 4))

    # Calculate mse
    mse = 0
    for pred, label in zip(probabilities, labels):
        mse += (pred - label) ** 2
    mse = round(mse / len(probabilities), 4)

    return probabilities, mse


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {
        "features": [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]],
        "labels": [0, 1, 0],
        "weights": [0.7, -0.4],
        "bias": -0.1,
    }
    print(inp)
    out = single_neuron_model(**inp)
    print(f"Output: {out}")
    exp = ([0.4626, 0.4134, 0.6682], 0.3349)
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {
        "features": [[1, 2], [2, 3], [3, 1]],
        "labels": [1, 0, 1],
        "weights": [0.5, -0.2],
        "bias": 0,
    }
    print(inp)
    out = single_neuron_model(**inp)
    print(f"Output: {out}")
    exp = ([0.525, 0.5987, 0.7858], 0.21)
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
