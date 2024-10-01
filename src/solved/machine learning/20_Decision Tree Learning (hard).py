"""
problem_id: 20
Category: machine learning
Difficulty: hard
URL: https://www.deep-ml.com/problem/Decision%20Tree%20Learning
Page: 1

==== Title ====
Decision Tree Learning (hard)

==== Description ====
Write a Python function that implements the decision tree learning algorithm for classification. The function should use recursive binary splitting based on entropy and information gain to build a decision tree. It should take a list of examples (each example is a dict of attribute-value pairs) and a list of attribute names as input, and return a nested dictionary representing the decision tree.

==== Example ====
Example:
        input: examples = [
                    {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'No'},
                    {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Strong', 'PlayTennis': 'No'},
                    {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
                    {'Outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'Yes'}
                ],
                attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']
        output: {
            'Outlook': {
                'Sunny': {'Humidity': {'High': 'No', 'Normal': 'Yes'}},
                'Overcast': 'Yes',
                'Rain': {'Wind': {'Weak': 'Yes', 'Strong': 'No'}}
            }
        }
        reasoning: Using the given examples, the decision tree algorithm determines that 'Outlook' is the best attribute to split the data initially. When 'Outlook' is 'Overcast', the outcome is always 'Yes', so it becomes a leaf node. In cases of 'Sunny' and 'Rain', it further splits based on 'Humidity' and 'Wind', respectively. The resulting tree structure is able to classify the training examples with the attributes 'Outlook', 'Temperature', 'Humidity', and 'Wind'.

==== Learn More ====
Decision Tree Learning Algorithm

The decision tree learning algorithm is a method used for classification that predicts the value of a target variable based on several input variables. Each internal node of the tree corresponds to an input variable, and each leaf node corresponds to a class label.

The recursive binary splitting starts by selecting the attribute that best separates the examples according to the entropy and information gain, which are calculated as follows:

Entropy: \(H(X) = -\sum p(x) \log_2 p(x)\)

Information Gain: \(IG(D, A) = H(D) - \sum \frac{|D_v|}{|D|} H(D_v)\)

Where:
- \(H(X)\) is the entropy of the set,
- \(IG(D, A)\) is the information gain of dataset \(D\) after splitting on attribute \(A\),
- \(D_v\) is the subset of \(D\) for which attribute \(A\) has value \(v\).

The attribute with the highest information gain is used at each step, and the dataset is split based on this attribute's values. This process continues recursively until all data is perfectly classified or no remaining attributes can be used to make a split.
"""

import math

# ==== Code ====
from collections import Counter


def calculate_entropy(examples, target_attr):
    total_instances = len(examples)
    if total_instances == 0:
        return 0
    class_counts = Counter([example[target_attr] for example in examples])
    entropy = 0

    for count in class_counts.values():
        p_c = count / total_instances
        entropy -= p_c * math.log2(p_c)

    return entropy


def information_gain(examples, attribute, target_attr, total_entropy):
    """Calculate the information gain of the attribute in the examples."""
    total_instances = len(examples)
    weighted_entropy = 0

    # Get the unique values of the attribute
    values = set(example[attribute] for example in examples)

    for v in values:
        examples_v = [
            example for example in examples if example[attribute] == v
        ]  # Subset where attribute = v
        entropy_v = calculate_entropy(examples_v, target_attr)  # Entropy of the subset
        weighted_entropy += (len(examples_v) / total_instances) * entropy_v

    return total_entropy - weighted_entropy


def learn_decision_tree(
    examples: list[dict], attributes: list[str], target_attr: str
) -> dict:
    # Your code here
    # https://www.analyticsvidhya.com/decision-tree-algorithm/#:~:text=A%20decision%20tree%20algorithm%20is,each%20node%20of%20the%20tree.

    # Check if all examples belong to the same class
    if len(set(example[target_attr] for example in examples)) == 1:
        return examples[0][target_attr]

    # Check if there are no remaining attributes
    if not attributes:
        most_common_class = Counter(
            [example[target_attr] for example in examples]
        ).most_common(1)[0][0]
        return most_common_class

    total_entropy = calculate_entropy(examples, target_attr)
    best_attribute = None
    max_info_gain = -float("inf")

    # Select the attribute with the highest information gain
    for attribute in attributes:
        ig = information_gain(examples, attribute, target_attr, total_entropy)

        if ig > max_info_gain:
            max_info_gain = ig
            best_attribute = attribute

    # If no information gain, return majority class label
    if max_info_gain == 0:
        most_common_class = Counter(
            [example[target_attr] for example in examples]
        ).most_common(1)[0][0]
        return most_common_class

    # Create a new tree node in dictionary format
    decision_tree = {best_attribute: {}}
    remaining_attributes = [attr for attr in attributes if attr != best_attribute]

    # Split the dataset and build child nodes recursively
    for value in set(example[best_attribute] for example in examples):
        examples_v = [
            example for example in examples if example[best_attribute] == value
        ]  # Subset where best_attribute = value
        child_node = learn_decision_tree(examples_v, remaining_attributes, target_attr)
        decision_tree[best_attribute][value] = child_node  # Add child node
    return decision_tree


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {
        "examples": [
            {"Outlook": "Sunny", "Wind": "Weak", "PlayTennis": "No"},
            {"Outlook": "Overcast", "Wind": "Strong", "PlayTennis": "Yes"},
            {"Outlook": "Rain", "Wind": "Weak", "PlayTennis": "Yes"},
            {"Outlook": "Sunny", "Wind": "Strong", "PlayTennis": "No"},
            {"Outlook": "Sunny", "Wind": "Weak", "PlayTennis": "Yes"},
            {"Outlook": "Overcast", "Wind": "Weak", "PlayTennis": "Yes"},
            {"Outlook": "Rain", "Wind": "Strong", "PlayTennis": "No"},
            {"Outlook": "Rain", "Wind": "Weak", "PlayTennis": "Yes"},
        ],
        "attributes": ["Outlook", "Wind"],
        "target_attr": "PlayTennis",
    }
    print(inp)
    out = learn_decision_tree(**inp)
    print(f"Output: {out}")
    exp = {
        "Outlook": {
            "Sunny": {"Wind": {"Weak": "No", "Strong": "No"}},
            "Rain": {"Wind": {"Weak": "Yes", "Strong": "No"}},
            "Overcast": "Yes",
        }
    }
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2

    # Test case 3
