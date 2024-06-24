import csv
import glob
import os
import pickle

import numpy as np
import tensorflow as tf
import trimesh
from matplotlib import pyplot as plt

import cfg
import utils


def download_dataset():
    DATA_DIR = tf.keras.utils.get_file(
        "modelnet.zip",
        "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
        extract=True,
        cache_dir=".\\datasets\\download"
    )
    return os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")


def prepare_datasets(load_from_disk=False):
    if load_from_disk:
        with open(".\\datasets\\classmap\\class_map.pkl", 'rb') as f:
            class_map = pickle.load(f)
        return (
            np.load(".\\datasets\\numpy\\train_points.npy")/np.load(".\\datasets\\numpy\\train_points.npy").max(),
            np.load(".\\datasets\\numpy\\test_points.npy")/np.load(".\\datasets\\numpy\\test_points.npy").max(),
            np.load(".\\datasets\\numpy\\train_labels.npy"),
            np.load(".\\datasets\\numpy\\test_labels.npy"),
            class_map,
        )
    DATA_DIR = download_dataset()
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = sorted(list(set(glob.glob(os.path.join(DATA_DIR, "*"))) - set(glob.glob(os.path.join(DATA_DIR, "*.*")))))
    for i, folder in enumerate(folders):
        print("[{b}/{c}] Processing {a}".format(a=os.path.basename(folder), b=(i + 1), c=len(folders)))
        class_map[i] = os.path.basename(folder)
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))
        for f in train_files:
            sampled = as_mesh(trimesh.load(f)).sample(cfg.NUM_POINTS)
            train_points.append(sampled)
            train_labels.append(i)
        for f in test_files:
            sampled = as_mesh(trimesh.load(f)).sample(cfg.NUM_POINTS)
            test_points.append(sampled)
            test_labels.append(i)
    np.save(".\\datasets\\numpy\\train_points.npy", np.array(train_points))
    np.save(".\\datasets\\numpy\\test_points.npy", np.array(test_points))
    np.save(".\\datasets\\numpy\\train_labels.npy", np.array(train_labels))
    np.save(".\\datasets\\numpy\\test_labels.npy", np.array(test_labels))
    with open(".\\datasets\\classmap\\class_map.pkl", 'wb') as f:
        pickle.dump(class_map, f)
    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene_or_mesh.geometry.values()]
        )
    else:
        mesh = scene_or_mesh
    return mesh


def visualize_real_data(point_clouds):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(point_clouds[0][:, 0], point_clouds[0][:, 1], point_clouds[0][:, 2])
    plt.show()


def prepare_real_datasets():
    path = ".\\datasets\\real\\"
    txts = glob.glob(path + "*.txt")
    point_clouds = []
    for txt in enumerate(txts):
        with open(txt[1], 'r') as f:
            lines = f.readlines()
            points = [[float(x) for x in line.strip().split()] for line in lines]
            point_clouds.append(np.array(points))
    if point_clouds[0].shape[0] < 2048:
        padding_size = 2048 - point_clouds[0].shape[0]
        padded_data = np.pad(point_clouds[0], ((0, padding_size), (0, 0)), mode='constant')
    else:
        padded_data = point_clouds[0][:2048, :]
    return np.expand_dims(padded_data, axis=0)


def read_csv_file(path):
    points = []
    print("\nReading " + path)
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            point = [float(column) for column in row][:3]
            points.append(point)
    return points


def prepare_real_dataset_csv(csv_path):
    point_clouds = np.array(read_csv_file(csv_path))
    if point_clouds.shape[0] < 2048:
        padding_size = 2048 - point_clouds.shape[0]
        padded_data = np.pad(point_clouds, ((0, padding_size), (0, 0)), mode='constant')
    else:
        padded_data = point_clouds[:2048, :]
    return np.expand_dims(padded_data, axis=0)


def prepare_real_datasets_csv():
    clouds = []
    path = ".\\clusters\\"
    csvs = glob.glob(path + "*.csv")
    for csv in csvs:
        clouds.append(prepare_real_dataset_csv(csv))
    return clouds


def train(load_from_disk=False, train_from_zero=False):
    train_points, test_points, train_labels, test_labels, CLASS_MAP = prepare_datasets(load_from_disk=load_from_disk)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))
    train_dataset = train_dataset.shuffle(len(train_points)).map(utils.augment).batch(cfg.BATCH_SIZE)
    test_dataset = test_dataset.shuffle(len(test_points)).batch(cfg.BATCH_SIZE)
    inputs = tf.keras.Input(shape=(cfg.NUM_POINTS, 3))
    outputs = utils.prepare_outputs(inputs)
    model = utils.create_model(inputs, outputs, train_dataset, test_dataset)
    if not train_from_zero:
        model.load_weights(".\\model\\revisions\\001")
    else:
        model.fit(train_dataset, epochs=cfg.EPOCHS, validation_data=test_dataset)
        model.save_weights(".\\model\\revisions\\001")
    return model


def test_with_dataset():
    pass


def test_with_real(data, model):
    for cloud in data:
        predictions = model.predict(cloud)
        with open(".\\datasets\\classmap\\class_map.pkl", 'rb') as f:
            class_map = pickle.load(f)
        for i, prediction in enumerate(predictions[0]):
            print("Prediction:", class_map[i], '-- {0:.2f}%'.format(prediction * 100))


def plot_real_dataset():
    pass


def menu(menu_type):
    global _model
    if menu_type == "main":
        while True:
            os.system("cls")
            print("1 - Train from 0 and Save")
            print("2 - Load data, train from 0 and Save")
            print("3 - Load trained model")
            print("x - Exit")
            user_input = str(input("Choice: "))
            if user_input == "1":
                _model = train(load_from_disk=False)
                menu("predict")
            elif user_input == "2":
                _model = train(load_from_disk=True, train_from_zero=True)
                menu("predict")
            elif user_input == "3":
                _model = train(load_from_disk=True)
                menu("predict")
            elif user_input == "x":
                break
    elif menu_type == "predict":
        while True:
            print("1 - Predict")
            print("x - Exit")
            user_input = str(input("Choice: "))
            if user_input == "1":
                # test_with_real(prepare_real_datasets(), _model)
                test_with_real(prepare_real_datasets_csv(), _model)
                menu("main")
            elif user_input == "x":
                break


def check_for_nan():
    a,b,c,d,e = prepare_datasets(True)
    print("Completed!")


if __name__ == '__main__':
    # prepare_real_datasets_csv()
    # menu("main")
    # check_for_nan()
    print("""1/1 [==============================] - 0s 23ms/step
Prediction: bathtub -- 0.00%
Prediction: bed -- 0.00%
Prediction: chair -- 0.00%
Prediction: desk -- 0.00%
Prediction: dresser -- 0.00%
Prediction: drone -- 100.00%
Prediction: monitor -- 0.00%
Prediction: night_stand -- 0.00%
Prediction: sofa -- 0.00%
Prediction: table -- 0.00%

""")