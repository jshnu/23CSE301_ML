def minkowski_distance(a, b, p):
    distance = 0

    for x, y in zip(a, b):
        distance += abs(x - y) ** p

    return distance ** (1 / p)

vector1 = [2, 4, 6, 8]
vector2 = [1, 3, 5, 7]

print("Manhattan Distance:", minkowski_distance(vector1, vector2, 1))
print("Euclidean Distance:", minkowski_distance(vector1, vector2, 2))
print("Minkowski Distance (p=3):", minkowski_distance(vector1, vector2, 3))
