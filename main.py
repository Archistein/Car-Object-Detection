from utils import raw_dataset_to_yolo, create_yaml_conf, make_video, plot_yolo_result
from ultralytics import YOLO
import numpy as np
import argparse
import random
import torch
import os
import cv2


def main() -> None:
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--train', help='switch to the training mode', action='store_true')  
    parser.add_argument('-b', '--batch_size', help='set batch size', type=int, default=16)
    parser.add_argument('-e', '--epoch', help='set epochs number', type=int, default=150)
    parser.add_argument('-is', '--imgsz', help='set image size', type=int, default=640)
    parser.add_argument('-p', '--params_path', help='set path to pretrained params', default='runs/detect/train/weights/best.pt')
    parser.add_argument('-r', '--root_dir', help='set path to root directory', default='CarObjDetection') 
    parser.add_argument('-l', '--labels', help='set path to labels file', default='CarObjDetection/train_solution_bounding_boxes.csv')  

    args = parser.parse_args()

    train_mode = args.train
    batch_size = args.batch_size
    epoch = args.epoch
    imgsz = args.imgsz
    params_path = args.params_path
    root_dir = args.root_dir
    labels = args.labels

    seed = 42 # Answer to the Ultimate Question of Life, the Universe, and Everything

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    if train_mode:
        print('Train mode activated')

        print('Data preproccessing:')
        raw_dataset_to_yolo(os.path.join(root_dir, 'training_images'), labels)
        create_yaml_conf(os.path.abspath('dataset'), 'images/train', 'images/val', ['car'], 'data.yaml')

        model = YOLO('yolov5s.yaml')
        
        print('Start training')

        model.train(
            data='data.yaml',
            epochs=epoch,
            imgsz=imgsz,
            batch=batch_size,
            seed=seed,
            single_cls=True,
            pretrained=False
        )

        print('Making a video from test images:')
        make_video(model, os.path.join(root_dir, 'testing_images'), 'test.mp4', 7, (676, 380))
        
        print('Export to ONNX')
        model.export(format='onnx', imgsz=imgsz)
    else:
        model = YOLO(params_path)
    
    print('Inference mode')

    while True:
        try:
            img_path = input('Path to image: ')
        except EOFError:
            break

        img = cv2.imread(img_path)

        results = model(img, verbose=False)

        for result in results:
            plot_yolo_result(img, result)

        cv2.imshow('Car Object Detection', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()