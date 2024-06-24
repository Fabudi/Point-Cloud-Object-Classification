import csv

import numpy as np
from scipy.spatial import distance
from sklearn.cluster import Birch


def cluster(x, threshold=0.5):
    print("Clusterization...")
    labels = Birch(n_clusters=8, threshold=threshold).fit_predict(x)
    print("Labels: " + str(set(labels)))
    print("Completed")
    return labels


def read_csv_file(path):
    points = []
    print("\nReading " + path)
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            point = [float(column) for column in row][:3]
            points.append(point)
    return points


def write2file(path, points, labels=None):
    print("Writing to " + path)
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for index, point in enumerate(points):
            if labels is None:
                writer.writerow([point[0], point[1], point[2]])
            else:
                writer.writerow([point[0], point[1], point[2], labels[index]])
    print("Completed")


def read_and_cluster(path):
    points = read_csv_file(path)
    return clusterize_points(points)


def clusterize_points(points, threshold=0.5):
    labels = cluster(points, threshold)
    clusters = []
    for label in sorted(list(set(labels))):
        clusters.append([point for point, label_ in zip(points, labels) if label_ == label])
    return points, labels, clusters


def create_voxel_grid(n=100, length=5):
    x = np.linspace(-length, length, n)
    y = np.linspace(-length, length, n)
    z = np.linspace(-length, length, n)
    grid = np.transpose(np.array(np.meshgrid(x, y, z)).reshape(3, -1))
    return grid


def count_points_in_voxels(grid, points):
    points_in_voxel = {}
    for point in points:
        x = point[0]
        y = point[1]
        z = point[2]
        if -grid[0] <= x <= grid[0] and -grid[0] <= y <= grid[0] and -grid[0] <= z <= grid[0]:
            voxel_x = int((x - grid[0]) // grid[1])
            voxel_y = int((y - grid[0]) // grid[1])
            voxel_z = int((z - grid[0]) // grid[1])
            voxel_key = f"{voxel_x}{voxel_y}{voxel_z}"
            points_in_voxel[voxel_key] = points_in_voxel.get(voxel_key, 0)
            points_in_voxel[voxel_key] += 1
    return points_in_voxel


def is_moving_cluster(grid, cluster1, cluster2, threshold=15):
    points_in_voxel_1 = count_points_in_voxels(grid, cluster1)
    points_in_voxel_2 = count_points_in_voxels(grid, cluster2)
    for key in points_in_voxel_1.keys():
        a = points_in_voxel_1[key]
        if key in points_in_voxel_2.keys():
            b = points_in_voxel_2[key]
        else:
            b = 0
        if abs(a - b) > threshold:
            return True
    return False


def find_center_point(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    z = [p[2] for p in points]
    centroid = (sum(x) / len(points), sum(y) / len(points), sum(z) / len(points))
    return centroid


def find_centroids(clusters):
    clusters_centers = []
    for cluster1 in clusters:
        clusters_centers.append(find_center_point(cluster1))
    return clusters_centers


def main():
    # make a grid
    grid = create_voxel_grid()
    write2file("./grid.csv", grid)

    # read and clusterize first frame
    path = "./datasets/real/lidar/Parsed/video_diag_1.csv"
    points1, labels1, clusters1 = read_and_cluster(path)
    write2file("./" + path.split("/")[-1], points1, labels1)

    # find centroids of the first frame
    centroids1 = find_centroids(clusters1)
    write2file("./centroids1.csv", centroids1)

    # read and clusterize second frame
    path = "./datasets/real/lidar/Parsed/video_diag_2.csv"
    points2, labels2, clusters2 = read_and_cluster(path)

    # find centroids of the second frame
    centroids2 = find_centroids(clusters2)
    write2file("./centroids2.csv", centroids1)
    # map centroids
    mapping = map_clusters(centroids1, centroids2)
    print("Mapping: " + str(mapping))
    clusters2, labels2 = apply_mapping(clusters2, labels2, mapping)
    write2file("./" + path.split("/")[-1], points2, labels2)
    # find moving centroids
    moving_clusters = []
    for index, cluster in enumerate(clusters1):
        print("Cluster #" + str(index+1))
        if is_moving_cluster([5, 100], cluster, clusters2[index], 4500):
            moving_clusters.append(cluster)
    print(len(moving_clusters))
    moving_points = []
    for cluster in moving_clusters:
        for point in cluster:
            moving_points.append(point)
    print(len(moving_points))
    write2file("./moving.csv", moving_points)
    p, l, c = clusterize_points(moving_points, 0.1)
    for index, cluster in enumerate(c):
        write2file("./clusters/c" + str(index) + ".csv", cluster)


def map_clusters(centroids1, centroids2):
    mapping = []
    for centroid in centroids2:
        nearest = find_nearest(centroid, centroids1)
        index = centroids1.index(nearest)
        mapping.append(index)
    return mapping


def apply_mapping(clusters, labels, mapping):
    new_labels = []
    new_clusters = []
    for label in labels:
        new_labels.append(mapping[label])
    for index, cluster in enumerate(clusters):
        new_clusters.append(clusters[index])
    return new_clusters, new_labels


def find_nearest(point, points):
    closest_index = distance.cdist([point], points).argmin()
    return points[closest_index]


if __name__ == '__main__':
    main()
