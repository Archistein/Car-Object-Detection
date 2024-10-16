
from ultralytics.engine.results import Results
from ultralytics.engine.model import Model
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import yaml
import cv2
import os


def raw_dataset_to_yolo(
        raw_images_path: str, 
        labels: str, 
        dataset_root: str = 'dataset', 
        train_test_split: float = 0.85
    ) -> None:
    images_path = os.path.join(dataset_root, 'images')
    labels_path = os.path.join(dataset_root, 'labels')

    labels = pd.read_csv(labels)

    os.makedirs(os.path.join(images_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(images_path, 'val'), exist_ok=True)
    os.makedirs(os.path.join(labels_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(labels_path, 'val'), exist_ok=True)

    for image, values in tqdm(labels.groupby('image')):
        img = cv2.imread(os.path.join(raw_images_path, image))
        
        bbox = values[['xmin', 'xmax', 'ymin', 'ymax']].to_numpy()
        w, h, _ = img.shape
        
        split = 'train' if random.random() < train_test_split else 'val'
        
        cv2.imwrite(os.path.join(images_path, split, image), img)

        with open(os.path.join(labels_path, split, image.replace(".jpg", ".txt")), 'w') as label:
            for box in bbox_to_yolo(bbox, (h, w)):
                label.write(f"0 {' '.join(map(str, box))}\n")


def bbox_to_yolo(bbox: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    x_center = np.sum(bbox[..., :2], axis=-1)/2 / size[0]
    y_center = np.sum(bbox[..., 2:], axis=-1)/2 / size[1]
    w = (bbox[..., 1] - bbox[..., 0]) / size[0]
    h = (bbox[..., 3] - bbox[..., 2]) / size[1]

    return np.stack([x_center, y_center, w, h]).T


def point_offset(
        point: tuple[int, int], 
        x_offset: int, 
        y_offset: int
    ) -> tuple[int, int]:
    return (point[0] + x_offset, point[1] + y_offset)


def plot_yolo_result(img: np.ndarray, result: Results) -> None:
    for box in result.boxes:
            box_coords = box.xyxy.squeeze(0).long().tolist()
            cv2.rectangle(img, box_coords[:2], box_coords[2:], (0, 255, 0), 2)
            cv2.rectangle(img, box_coords[:2], point_offset(box_coords[:2], 80, -18), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'Car {box.conf.item():.2f}', point_offset(box_coords[:2], 2, -4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)


def make_video(
        yolo: Model, 
        images_path: str, 
        video_name: str, 
        fps: int, 
        size: tuple[int, int]
    ) -> None:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(video_name, fourcc, fps, size)

    for img_path in tqdm(sorted(os.listdir(images_path))):
        img = cv2.imread(os.path.join(images_path, img_path))

        results = yolo(img, verbose=False)

        for result in results:
            plot_yolo_result(img, result)

        video.write(img)

    video.release()


def create_yaml_conf(
        root_path: str,
        train_path: str,
        val_path: str,
        classes: list[str], 
        yaml_path: str
    ) -> None:

    yaml_dict = {
        'path': root_path,
        'train': train_path,
        'val': val_path,
        'names': dict(zip(range(len(classes)), classes))
    }

    with open(yaml_path, 'w') as yamlf:
        yaml.dump(yaml_dict, yamlf, sort_keys=False)