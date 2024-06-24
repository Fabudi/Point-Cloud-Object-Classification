import csv
import glob

import numpy as np
from matplotlib import pyplot as plt


class Point:

    def __init__(self, frame_id, x, y, z, distance):
        self.frame_id = frame_id
        self.x = x
        self.y = y
        self.z = z
        self.distance = distance


def read_csv_file(path):
    points = []
    print(path)
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            points.append(
                Point(row["FrameNumbers"], row["Points_X"], row["Points_Y"], row["Points_Z"], row["Distance"]))
    return points


def write_files(path, points):
    for index in set(o.frame_id for o in points):
        with open(path + "_" + str(index) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for point in points:
                if point.frame_id == index:
                    writer.writerow([point.x, point.y, point.z, point.distance])
        print(str(index) + " Completed")


def exclude_moving(points):
    points_by_frame = {}
    for point in points:
        if point.frame_id not in points_by_frame:
            points_by_frame[point.frame_id] = []
        points_by_frame[point.frame_id].append(point)
    moving_points = []
    for frame_id, frame_points in points_by_frame.items():
        if len(frame_points) < 2:
            continue
        points_arr = np.array([[p.x, p.y, p.z] for p in frame_points])
        median_point = np.median(points_arr, axis=0)
        std_dev = np.std(points_arr, axis=0)
        movement_threshold = 2.0 * std_dev
        for point in frame_points:
            distance_from_median = np.linalg.norm(np.array([point.x, point.y, point.z]) - median_point)
            if distance_from_median > movement_threshold.any():
                moving_points.append(point)
    return moving_points


def main():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    paths = glob.glob('S:\\Keras\\data\\lidar\\drones\\lidar\\video_*.csv')
    for path in paths:
        points = np.array(read_csv_file(path))
        points = exclude_moving(points)
        x_data = np.array([float(point.x) for point in points])
        y_data = np.array([float(point.y) for point in points])
        z_data = np.array([float(point.z) for point in points])
        ax.scatter(x_data,y_data,z_data, marker='.')
        plt.show()
        # write_files(path.split(".csv")[0], points)



if __name__ == '__main__':
    main()
