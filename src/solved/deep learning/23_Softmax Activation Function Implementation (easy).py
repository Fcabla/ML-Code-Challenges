"""
problem_id: 23
Category: deep learning
Difficulty: easy
URL: https://www.deep-ml.com/problem/Softmax%20Activation%20Function%20Implementation
Page: 1

==== Title ====
Softmax Activation Function Implementation (easy)

==== Description ====
Write a Python function that computes the softmax activation for a given list of scores. The function should return the softmax values as a list, each rounded to four decimal places.

==== Example ====
Example:
        input: scores = [1, 2, 3]
        output: [0.0900, 0.2447, 0.6652]
        reasoning: The softmax function converts a list of values into a probability distribution. The probabilities are proportional to the exponential of each element divided by the sum of the exponentials of all elements in the list.

==== Learn More ====
Understanding the Softmax Activation Function

The softmax function is a generalization of the sigmoid function and is used in the output layer of a neural network model that handles multi-class classification tasks.

Mathematical Definition

The softmax function is mathematically represented as:

\[
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
\]

Characteristics

Output Range: Each output value is between 0 and 1, and the sum of all outputs is 1.
Purpose: It transforms scores into probabilities, which are easier to interpret and are useful for classification.


This function is essential for models where the output needs to represent a probability distribution across multiple classes.
"""

# ==== Code ====
import math


def softmax(scores: list[float]) -> list[float]:
    # Your code here
    suma = sum([math.exp(score) for score in scores])
    probabilities = [round(math.exp(score) / suma, 4) for score in scores]
    return probabilities


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"scores": [1, 2, 3]}
    print(inp)
    out = softmax(**inp)
    print(f"Output: {out}")
    exp = [0.09, 0.2447, 0.6652]

    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"scores": [-1, 0, 5]}
    print(inp)
    out = softmax(**inp)
    print(f"Output: {out}")
    exp = [0.0025, 0.0067, 0.9909]

    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
