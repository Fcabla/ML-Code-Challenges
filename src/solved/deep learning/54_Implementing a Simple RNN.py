"""
problem_id: 54
Category: deep learning
Difficulty: medium
URL: https://www.deep-ml.com/problem/RNN%20Layer
Page: 3

==== Title ====
Implementing a Simple RNN

==== Description ====
Write a Python function that implements a simple Recurrent Neural Network (RNN) cell. The function should process a sequence of input vectors and produce the final hidden state. Use the tanh activation function for the hidden state updates. The function should take as inputs the sequence of input vectors, the initial hidden state, the weight matrices for input-to-hidden and hidden-to-hidden connections, and the bias vector. The function should return the final hidden state after processing the entire sequence, rounded to four decimal places.

==== Example ====
Example:
    input_sequence = [[1.0], [2.0], [3.0]]
    initial_hidden_state = [0.0]
    Wx = [[0.5]]  # Input to hidden weights
    Wh = [[0.8]]  # Hidden to hidden weights
    b = [0.0]     # Bias
    output: final_hidden_state = [0.9993]
    reasoning: The RNN processes each input in the sequence, updating the hidden state at each step using the tanh activation function.

==== Learn More ====
Understanding Recurrent Neural Networks (RNNs)
Recurrent Neural Networks are a class of neural networks designed to handle sequential data by maintaining a hidden state that captures information from previous inputs.
Mathematical Formulation
For each time step \( t \), the RNN updates its hidden state \( h_t \) using the current input \( x_t \) and the previous hidden state \( h_{t-1} \):

\[
h_t = \tanh(W_x x_t + W_h h_{t-1} + b)
\]

Where:

\( W_x \) is the weight matrix for the input-to-hidden connections.
\( W_h \) is the weight matrix for the hidden-to-hidden connections.
\( b \) is the bias vector.
\( \tanh \) is the hyperbolic tangent activation function applied element-wise.

Implementation Steps

Initialization: Start with the initial hidden state \( h_0 \).
Sequence Processing: For each input \( x_t \) in the sequence:

Compute \( h_t = \tanh(W_x x_t + W_h h_{t-1} + b) \).


Final Output: After processing all inputs, the final hidden state \( h_T \) (where \( T \) is the length of the sequence) contains information from the entire sequence.

Example Calculation
Given:

Inputs: \( x_1 = 1.0 \), \( x_2 = 2.0 \), \( x_3 = 3.0 \)
Initial hidden state: \( h_0 = 0.0 \)
Weights:

\( W_x = 0.5 \)
\( W_h = 0.8 \)

Bias: \( b = 0.0 \)

Compute:

First time step (\( t = 1 \)):

\[
h_1 = \tanh(0.5 \times 1.0 + 0.8 \times 0.0 + 0.0) = \tanh(0.5) = 0.4621
\]

Second time step (\( t = 2 \)):

\[
h_2 = \tanh(0.5 \times 2.0 + 0.8 \times 0.4621 + 0.0) = \tanh(1.0 + 0.3697) = \tanh(1.3697) = 0.8781
\]

Third time step (\( t = 3 \)):

\[
h_3 = \tanh(0.5 \times 3.0 + 0.8 \times 0.8781 + 0.0) = \tanh(1.5 + 0.7025) = \tanh(2.2025) = 0.9750
\]


The final hidden state \( h_3 \) is approximately 0.9750.
Applications
RNNs are widely used in natural language processing, time-series prediction, and any task involving sequential data.
"""

# ==== Code ====
import numpy as np


def rnn_forward(
    input_sequence: list[list[float]],
    initial_hidden_state: list[float],
    Wx: list[list[float]],
    Wh: list[list[float]],
    b: list[float],
) -> list[float]:
    # Your code here
    h = np.array(initial_hidden_state)
    Wx = np.array(Wx)
    Wh = np.array(Wh)
    b = np.array(b)

    for x in input_sequence:
        x = np.array(x)
        h = np.tanh((Wx @ x) + (Wh @ h) + b)
    return np.round(h, 4).tolist()


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {
        "input_sequence": [[1.0], [2.0], [3.0]],
        "initial_hidden_state": [0.0],
        "Wx": [[0.5]],
        "Wh": [[0.8]],
        "b": [0.0],
    }
    print(inp)
    out = rnn_forward(**inp)
    print(f"Output: {out}")
    exp = [0.9759]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {
        "input_sequence": [[0.5], [0.1], [-0.2]],
        "initial_hidden_state": [0.0],
        "Wx": [[1.0]],
        "Wh": [[0.5]],
        "b": [0.1],
    }
    print(inp)
    out = rnn_forward(**inp)
    print(f"Output: {out}")
    exp = [0.118]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 3
    print("Test Case 3:")
    inp = {
        "input_sequence": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "initial_hidden_state": [0.0, 0.0],
        "Wx": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "Wh": [[0.7, 0.8], [0.9, 1.0]],
        "b": [0.1, 0.2],
    }
    print(inp)
    out = rnn_forward(**inp)
    print(f"Output: {out}")
    exp = [0.7474, 0.9302]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
