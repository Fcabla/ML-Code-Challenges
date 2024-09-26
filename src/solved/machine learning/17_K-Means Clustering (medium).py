"""
problem_id: 17
Category: machine learning
Difficulty: medium
URL: https://www.deep-ml.com/problem/K-Means%20Clustering
Page: 1

==== Title ====
K-Means Clustering (medium)

==== Description ====
Your task is to write a Python function that implements the k-Means clustering algorithm. This function should take specific inputs and produce a list of final centroids. k-Means clustering is a method used to partition n points into k clusters. The goal is to group similar points together and represent each group by its center (called the centroid). Function Inputs:

points: A list of points, where each point is a tuple of coordinates (e.g., (x, y) for 2D points)
k: An integer representing the number of clusters to form
initial_centroids: A list of initial centroid points, each a tuple of coordinates
max_iterations: An integer representing the maximum number of iterations to perform

Function Output:
A list of the final centroids of the clusters, where each centroid is rounded to the nearest fourth decimal.

==== Example ====
Example:
        input: points = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)], k = 2, initial_centroids = [(1, 1), (10, 1)], max_iterations = 10
        output: [(1, 2), (10, 2)]
        reasoning: Given the initial centroids and a maximum of 10 iterations,
        the points are clustered around these points, and the centroids are
        updated to the mean of the assigned points, resulting in the final
        centroids which approximate the means of the two clusters.
        The exact number of iterations needed may vary,
        but the process will stop after 10 iterations at most.

==== Learn More ====
K-Means Clustering Algorithm Implementation
Algorithm Steps:
1. Initialization
Use the provided initial_centroids as your starting point. This step is already done for you in the input.
2. Assignment Step
For each point in your dataset:

Calculate its distance to each centroid (Hint: use Euclidean distance.)
Assign the point to the cluster of the nearest centroid

Hint: Consider creating a helper function to calculate Euclidean distance between two points.
3. Update Step
For each cluster:

Calculate the mean of all points assigned to the cluster
Update the centroid to this new mean position

Hint: Be careful with potential empty clusters. Decide how you'll handle them (e.g., keep the previous centroid).
4. Iteration
Repeat steps 2 and 3 until either:

The centroids no longer change significantly (this case does not need to be included in your solution), or
You reach the max_iterations limit

Hint: You might want to keep track of the previous centroids to check for significant changes.
5. Result
Return the list of final centroids, ensuring each coordinate is rounded to the nearest fourth decimal.
References:
For a visual understanding of how k-Means clustering works, check out this helpful YouTube video.
"""


# ==== Code ====
def k_means_clustering(
    points: list[tuple[float, float]],
    k: int,
    initial_centroids: list[tuple[float, float]],
    max_iterations: int,
) -> list[tuple[float, float]]:
    # Your code here
    # 1. initialization (initial_centroids and groups)
    final_centroids = initial_centroids
    for _ in range(max_iterations):
        # 2. assign group to each point based on closest centroid
        groups = [[] for _ in range(k)]
        for point in points:
            min_distance = 9999
            current_centroid = None
            for i in range(len(final_centroids)):
                centroid = final_centroids[i]
                distance = (
                    (point[0] - centroid[0]) ** 2 + (point[1] - centroid[1]) ** 2
                ) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    current_centroid = i
            groups[current_centroid].append(point)
        # 3. Update centroids
        new_centroids = []
        for i in range(len(groups)):
            group = groups[i]
            if group:
                dimensions = len(group[0])  # 2
                new_centroids.append(
                    [sum(p[i] for p in group) / len(group) for i in range(dimensions)]
                )
            else:
                new_centroids.append(final_centroids[i])
        final_centroids = new_centroids

    return [tuple([round(el, 4) for el in centr]) for centr in final_centroids]


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {
        "points": [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)],
        "k": 2,
        "initial_centroids": [(1, 1), (10, 1)],
        "max_iterations": 10,
    }
    print(inp)
    out = k_means_clustering(**inp)
    print(f"Output: {out}")
    exp = [(1.0, 2.0), (10.0, 2.0)]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {
        "points": [
            (0, 0, 0),
            (2, 2, 2),
            (1, 1, 1),
            (9, 10, 9),
            (10, 11, 10),
            (12, 11, 12),
        ],
        "k": 2,
        "initial_centroids": [(1, 1, 1), (10, 10, 10)],
        "max_iterations": 10,
    }
    print(inp)
    out = k_means_clustering(**inp)
    print(f"Output: {out}")
    exp = [(1.0, 1.0, 1.0), (10.3333, 10.6667, 10.3333)]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 3
    print("Test Case 3:")
    inp = {
        "points": [(1, 1), (2, 2), (3, 3), (4, 4)],
        "k": 1,
        "initial_centroids": [(0, 0)],
        "max_iterations": 10,
    }
    print(inp)
    out = k_means_clustering(**inp)
    print(f"Output: {out}")
    exp = [(2.5, 2.5)]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 4
    print("Test Case 4:")
    inp = {
        "points": [
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (5, 5),
            (6, 5),
            (5, 6),
            (6, 6),
            (0, 5),
            (1, 5),
            (0, 6),
            (1, 6),
            (5, 0),
            (6, 0),
            (5, 1),
            (6, 1),
        ],
        "k": 4,
        "initial_centroids": [(0, 0), (0, 5), (5, 0), (5, 5)],
        "max_iterations": 10,
    }
    print(inp)
    out = k_means_clustering(**inp)
    print(f"Output: {out}")
    exp = [(0.5, 0.5), (0.5, 5.5), (5.5, 0.5), (5.5, 5.5)]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")


"""
Solution from the source (cleaner)
import numpy as np

def euclidean_distance(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=1))

def k_means_clustering(points, k, initial_centroids, max_iterations):
    points = np.array(points)
    centroids = np.array(initial_centroids)

    for iteration in range(max_iterations):
        # Assign points to the nearest centroid
        distances = np.array([euclidean_distance(points, centroid) for centroid in centroids])
        assignments = np.argmin(distances, axis=0)

        new_centroids = np.array([points[assignments == i].mean(axis=0) if len(points[assignments == i]) > 0 else centroids[i] for i in range(k)])

        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
        centroids = np.round(centroids,4)
    return [tuple(centroid) for centroid in centroids]
"""
