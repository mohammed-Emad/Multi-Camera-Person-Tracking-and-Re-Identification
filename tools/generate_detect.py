import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import errno
import argparse
import numpy as np
import cv2
import tensorflow as tf
from torchvision.transforms import functional as F
import torch.nn as nn
import torch
from PIL import Image
import requests
from torchvision import transforms
from torchvision.transforms.functional import resize, normalize
import math


alpha = 1 
beta = 0.6 # transparency for the segmentation map
gamma = 0 # scalar added to each sum
#[topleft_x, topleft_y, w, h]


Sift = cv2.SIFT_create()
def sift_extract_feat(img):
    kp, ds = Sift.detectAndCompute(img, None)
    return [kp, ds]

#out=np.vstack(phlist)
def crop_mask_g(imager, masks,boxes,labels, sizeim):
    phlist = []
    boxes2 = []
    for i in range(len(masks)):
        if labels[i]=='person':
            red_map = np.zeros_like(masks[i]).astype(np.uint8)
            try:
                # apply a randon color mask to each object
                red_map[masks[i] == 1] = 255
                res = cv2.bitwise_and(imager,imager, mask= red_map)
                x00 = (boxes[i][0][0], boxes[i][1][0])
                x11 = (boxes[i][0][1], boxes[i][1][1])
                x = min(x00)
                y = min(x11)
                width = max(x00)
                height = max(x11)
                crop_img = res[y:height, x:width]
                
                crop_img = sift_extract_feat(cv2.resize(crop_img, sizeim))
                phlist.append(crop_img)
                boxx = [x,y,int(width-x), int(height-y)]
                boxes2.append(boxx)
            except:
                print(masks[i].shape)
    return phlist ,np.array(boxes2)


#no faceNet
def create_box_encoder():
    def encoder(image, masks,boxes,labels):
        features, boxes2 = crop_mask_g(image, masks,boxes,labels, (160,160))
        return features, boxes2
    return encoder


def generate_detections(encoder, mot_dir, output_dir, detection_dir=None):
    """Generate detections with features.

    Parameters
    ----------
    encoder : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    mot_dir : str
        Path to the MOTChallenge directory (can be either train or test).
    output_dir
        Path to the output directory. Will be created if it does not exist.
    detection_dir
        Path to custom detections. The directory structure should be the default
        MOTChallenge structure: `[sequence]/det/det.txt`. If None, uses the
        standard MOTChallenge detections.

    """
    if detection_dir is None:
        detection_dir = mot_dir
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_dir)

    for sequence in os.listdir(mot_dir):
        print("Processing %s" % sequence)
        sequence_dir = os.path.join(mot_dir, sequence)

        image_dir = os.path.join(sequence_dir, "img1")
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}

        detection_file = os.path.join(
            detection_dir, sequence, "det/det.txt")
        detections_in = np.loadtxt(detection_file, delimiter=',')
        detections_out = []

        frame_indices = detections_in[:, 0].astype(np.int)
        min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()
        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            if frame_idx not in image_filenames:
                print("WARNING could not find image for frame %d" % frame_idx)
                continue
            bgr_image = cv2.imread(
                image_filenames[frame_idx], cv2.IMREAD_COLOR)
            features = encoder(bgr_image, rows[:, 2:6].copy())
            detections_out += [np.r_[(row, feature)] for row, feature
                               in zip(rows, features)]

        output_filename = os.path.join(output_dir, "%s.npy" % sequence)
        np.save(
            output_filename, np.asarray(detections_out), allow_pickle=False)


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")
    parser.add_argument(
        "--model",
        default="resources/networks/mars-small128.pb",
        help="Path to freezed inference graph protobuf.")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        required=True)
    parser.add_argument(
        "--detection_dir", help="Path to custom detections. Defaults to "
        "standard MOT detections Directory structure should be the default "
        "MOTChallenge structure: [sequence]/det/det.txt", default=None)
    parser.add_argument(
        "--output_dir", help="Output directory. Will be created if it does not"
        " exist.", default="detections")
    return parser.parse_args()


def main():
    args = parse_args()
    encoder = create_box_encoder(args.model, batch_size=32)
    generate_detections(encoder, args.mot_dir, args.output_dir,
                        args.detection_dir)


if __name__ == "__main__":
    main()
