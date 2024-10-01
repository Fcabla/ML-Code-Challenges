"""
problem_id: 51
Category: nlp
Difficulty: medium
URL: https://www.deep-ml.com/problem/Levenshtein%20Distance
Page: 3

==== Title ====
Optimal String Alignment Distance

==== Description ====
In this problem, you need to implement a function that calculates the Optimal String Alignment (OSA) distance between two given strings. The OSA distance represents the minimum number of edits required to transform one string into another. The allowed edit operations are:

Insert a character
Delete a character
Substitute a character
Transpose two adjacent characters

Each of these operations costs 1 unit.
Your task is to find the minimum number of edits needed to convert the first string (s1) into the second string (s2).
For example, the OSA distance between the strings "caper" and "acer" is 2: one deletion (removing "p"), and one transposition (swapping "a" and "c").

==== Example ====
Example:
source = "butterfly"
target = "dragonfly"

distance = OSA(source, target)
print(distance)

# Expected Output: 6

==== Learn More ====
Optimal String Alignment Distance
 Given two strings s1, and s2, find the Optimal String Alignment distance between them
 The OSA distance gives the minimum number of edits that can be done on string s1, to achieve s2. Here are the edit operations you can do:

Insert a character
Delete a character
Substitute a character
Transpose two adjacent characters.

Each operation will cost 1 unit.

For example, the OSA distance between the string "caper", and "acer" will be 2. One deletion - p, and one transposition - a and c.
"""

# ==== Code ====


# def OSA(source: str, target: str) -> int:  # ruff error
def osa(source: str, target: str) -> int:
    # Your code here
    # https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance
    source_len, target_len = len(source), len(target)
    # Set resulting object all to 0s
    d = [[0] * (target_len + 1) for _ in range(source_len + 1)]

    # Initialize the matrix (first row from pos 1 a range of target_len)
    for j in range(1, target_len + 1):
        d[0][j] = j
    # Initialize the matrix (first col from pos 1 a range of source_len)
    for i in range(1, source_len + 1):
        d[i][0] = i

    for i in range(1, source_len + 1):
        for j in range(1, target_len + 1):
            if source[i - 1] == target[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(
                d[i - 1][j] + 1,  # deletion
                d[i][j - 1] + 1,  # insertion
                d[i - 1][j - 1] + cost,  # substitution
            )
            if (
                i > 1
                and j > 1
                and source[i - 1] == target[j - 2]
                and source[i - 2] == target[j - 1]
            ):
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)  # transposicion

    return d[source_len][target_len]


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"source": "butterfly", "target": "dragonfly"}
    print(inp)
    out = osa(**inp)
    print(f"Output: {out}")
    exp = 6
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"source": "caper", "target": "acer"}
    print(inp)
    out = osa(**inp)
    print(f"Output: {out}")
    exp = 2
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 3
    print("Test Case 3:")
    inp = {"source": "telescope", "target": "microscope"}
    print(inp)
    out = osa(**inp)
    print(f"Output: {out}")
    exp = 5
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 4
    print("Test Case 4:")
    inp = {"source": "london", "target": "paris"}
    print(inp)
    out = osa(**inp)
    print(f"Output: {out}")
    exp = 6
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
