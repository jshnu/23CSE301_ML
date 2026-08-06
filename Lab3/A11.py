import pandas as pd
import random
import math

def distance(a, b):
    total = 0
    for x, y in zip(a, b):
        total += (x - y) ** 2
    return math.sqrt(total)

def mean(points):
    result = []

    for values in zip(*points):
        result.append(sum(values) / len(values))

    return result

def kmeans(data, k, iterations):
    centroids = random.sample(data, k)

    for _ in range(iterations):
        clusters = [[] for _ in range(k)]

        for point in data:
            distances = []

            for centroid in centroids:
                distances.append(distance(point, centroid))

            index = distances.index(min(distances))
            clusters[index].append(point)

        new_centroids = []

        for cluster in clusters:
            if len(cluster) == 0:
                new_centroids.append(random.choice(data))
            else:
                new_centroids.append(mean(cluster))

        if new_centroids == centroids:
            break

        centroids = new_centroids

    return centroids, clusters

df = pd.read_excel("data.xlsx", sheet_name="marketing_campaign")

categorical = df.select_dtypes(include=["object", "string"]).columns

for col in categorical:
    df[col] = df[col].astype(str)

df = pd.get_dummies(df, columns=categorical, dtype=int)

numeric = df.select_dtypes(include="number")
numeric = numeric.fillna(numeric.mean())

data = numeric.values.tolist()

centroids, clusters = kmeans(data, 3, 100)

print("Centroids")

for centroid in centroids:
    print(centroid)

print("\nCluster Sizes")

for i in range(len(clusters)):
    print("Cluster", i + 1, ":", len(clusters[i]))
