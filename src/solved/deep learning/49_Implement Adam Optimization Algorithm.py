"""
problem_id: 49
Category: deep learning
Difficulty: medium
URL: https://www.deep-ml.com/problem/Adam%20Optimization
Page: 3

==== Title ====
Implement Adam Optimization Algorithm

==== Description ====
Implement the Adam (Adaptive Moment Estimation) optimization algorithm in Python. Adam is an optimization algorithm that adapts the learning rate for each parameter. Your task is to write a function `adam_optimizer` that updates the parameters of a given function using the Adam algorithm.
The function should take the following parameters:

`f`: The objective function to be optimized
`grad`: A function that computes the gradient of `f`
`x0`: Initial parameter values
`learning_rate`: The step size (default: 0.001)
`beta1`: Exponential decay rate for the first moment estimates (default: 0.9)
`beta2`: Exponential decay rate for the second moment estimates (default: 0.999)
`epsilon`: A small constant for numerical stability (default: 1e-8)
`num_iterations`: Number of iterations to run the optimizer (default: 1000)

The function should return the optimized parameters.

==== Example ====
Example:
import numpy as np

def objective_function(x):
    return x[0]**2 + x[1]**2

def gradient(x):
    return np.array([2*x[0], 2*x[1]])

x0 = np.array([1.0, 1.0])
x_opt = adam_optimizer(objective_function, gradient, x0)

print("Optimized parameters:", x_opt)

# Expected Output:
# Optimized parameters: [0.99000325 0.99000325]

==== Learn More ====
Understanding the Adam Optimization Algorithm
Adam (Adaptive Moment Estimation) is an optimization algorithm commonly used in training deep neural networks. It combines ideas from two other optimization algorithms: RMSprop and Momentum.
Key Concepts

Adaptive Learning Rates: Adam computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients.
Momentum: It keeps track of an exponentially decaying average of past gradients, similar to momentum.
RMSprop: It also keeps track of an exponentially decaying average of past squared gradients.
Bias Correction: Adam includes bias correction terms to account for the initialization of the first and second moment estimates.

The Adam Algorithm
Given parameters \(\theta\), objective function \(f(\theta)\), and its gradient \(\nabla_\theta f(\theta)\):

Initialize time step \(t = 0\), parameters \(\theta_0\), first moment vector \(m_0 = 0\), second moment vector \(v_0 = 0\), and hyperparameters \(\alpha\) (learning rate), \(\beta_1\), \(\beta_2\), and \(\epsilon\).
While not converged, do:

Increment time step: \(t = t + 1\)
Compute gradient: \(g_t = \nabla_\theta f_t(\theta_{t-1})\)
Update biased first moment estimate: \(m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t\)
Update biased second raw moment estimate: \(v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2\)
Compute bias-corrected first moment estimate: \(\hat{m}_t = m_t / (1 - \beta_1^t)\)
Compute bias-corrected second raw moment estimate: \(\hat{v}_t = v_t / (1 - \beta_2^t)\)
Update parameters: \(\theta_t = \theta_{t-1} - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)\)



Adam combines the advantages of AdaGrad, which works well with sparse gradients, and RMSProp, which works well in online and non-stationary settings. Adam is generally regarded as being fairly robust to the choice of hyperparameters, though the learning rate may sometimes need to be changed from the suggested default.
"""

# ==== Code ====
import numpy as np


def adam_optimizer(
    f,
    grad,
    x0,
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    num_iterations=10,
):
    # Your code here
    x = x0
    # np.zeros_like
    m = np.zeros(x0.shape)
    v = np.zeros(x0.shape)
    for iteration in range(1, num_iterations + 1):
        # Compute gradient
        g = grad(x)

        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * g

        # Update biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * g**2

        # Compute bias-corrected first moment estimate
        mh = m / (1 - beta1**iteration)

        # Compute bias-corrected second raw moment
        vh = v / (1 - beta2**iteration)

        # Update parameters
        x = x - learning_rate * mh / (np.sqrt(vh) + epsilon)

    return np.round(x, 8).tolist()


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")

    def objective_function(x):
        return x[0] ** 2 + x[1] ** 2

    def gradient(x):
        return np.array([2 * x[0], 2 * x[1]])

    inp = {"f": objective_function, "grad": gradient, "x0": np.array([1.0, 1.0])}
    print(inp)
    out = adam_optimizer(**inp)
    print(f"Output: {out}")
    exp = [0.99000325, 0.99000325]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")

    def objective_function(x):
        return x[0] ** 2 + x[1] ** 2

    def gradient(x):
        return np.array([2 * x[0], 2 * x[1]])

    inp = {"f": objective_function, "grad": gradient, "x0": np.array([0.2, 12.3])}
    print(inp)
    out = adam_optimizer(**inp)
    print(f"Output: {out}")
    exp = [0.19001678, 12.29000026]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 3
