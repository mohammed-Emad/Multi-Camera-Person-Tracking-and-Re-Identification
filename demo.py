# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
from keras.backend import set_session as KTF

from timeit import time
import warnings
import argparse

import sys
import cv2
import numpy as np
import base64
import requests
import urllib
from urllib import parse
import json
import random
import time
from PIL import Image
from collections import Counter
import operator

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

from reid import REID
import copy

import cv2
import numpy as np
import random
import torch
import torch
import torchvision
import cv2
import argparse
from PIL import Image

from torchvision.transforms import transforms as transforms




coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))


def get_outputs(image, model, threshold):
    with torch.no_grad():
        # forward pass of the image through the modle
        outputs = model(image)
    
    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    # get the masks
    masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    # discard masks for objects which are below threshold
    masks = masks[:thresholded_preds_count]
    # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]
    # discard bounding boxes below threshold value
    boxes = boxes[:thresholded_preds_count]
    # get the classes labels
    labels = [coco_names[i] for i in outputs[0]['labels']]
    return masks, boxes, labels

def draw_segmentation_map(image, masks, boxes, labels):
    alpha = 1 
    beta = 0.6 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    for i in range(len(masks)):
        red_map = np.zeros_like(masks[i]).astype(np.uint8)
        green_map = np.zeros_like(masks[i]).astype(np.uint8)
        blue_map = np.zeros_like(masks[i]).astype(np.uint8)
        # apply a randon color mask to each object
        color = COLORS[random.randrange(0, len(COLORS))]
        red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1]  = color
        # combine all the masks into a single image
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
        #convert the original PIL image into NumPy format
        image = np.array(image)
        # convert from RGN to OpenCV BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # apply mask on the image
        cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
        # draw the bounding boxes around the objects
        cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color, 
                      thickness=2)
        # put the label text above the objects
        cv2.putText(image , labels[i], (boxes[i][0][0], boxes[i][0][1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 
                    thickness=2, lineType=cv2.LINE_AA)
    
    return image

parser = argparse.ArgumentParser()
parser.add_argument('--version', help='Model(yolo_v3 or yolo_v4)', default='yolo_v4')
parser.add_argument('--videos', nargs='+', help='List of videos', required=True)
parser.add_argument('-all', help='Combine all videos into one', default=True)
args = parser.parse_args()  # vars(parser.parse_args())

def rescale_frameA(frame_input, percent=75):
    width = int(frame_input[0] * percent / 100)
    height = int(frame_input[1] * percent / 100)
    return width, height

class LoadVideo:  # for inference
    def __init__(self, path, img_size=(720, 416)):
        if not os.path.isfile(path):
            raise FileExistsError

        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #self.vw, self.vh = rescale_frameA([self.vw, self.vh], percent=53)
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        print('Length of {}: {:d} frames'.format(path, self.vn))

    def get_VideoLabels(self):
        return self.cap, self.frame_rate, self.vw, self.vh

def rescale_frame(frame_input, percent=75):
    width = int(frame_input.shape[1] * percent / 100)
    height = int(frame_input.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame_input, dim, interpolation=cv2.INTER_AREA)


def main(yolo):
    print(f'Using {yolo} model')
    # Definition of the parameters
    max_cosine_distance = 0.2
    nn_budget = None
    nms_max_overlap = 0.4
    threshold = 0.969 #0.97 # 0.965

    # initialize the model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True, 
                                                              num_classes=91)
    # set the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load the modle on to the computation device and set to eval mode
    model.to(device).eval()



    # transform to convert the image to tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # deep_sort
    model_filename = 'model_data/models/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)  # use to get feature

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=100)

    output_frames = []
    output_rectanger = []
    output_areas = []
    output_wh_ratio = []

    is_vis = True
    out_dir = 'videos/output/'
    print('The output folder is', out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    all_frames = []
    for video in args.videos:
        all_frames0 = []
        loadvideo = LoadVideo(video)
        video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()
        while True:
            ret, frame = video_capture.read()
            
            #print(frame.shape)
            if ret is not True:
                video_capture.release()
                break

            all_frames0.append(frame)
        all_frames.append(all_frames0)
    print(frame_rate, (w, h))
    frame_nums = len(all_frames)
    tracking_path = out_dir + 'tracking' + '.avi'
    combined_path = out_dir + 'allVideos' + '.avi'
    if is_vis:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(out_dir + 'tracking' + '.avi', fourcc, frame_rate, (w, h))
        out0 = cv2.VideoWriter(out_dir + 'tracking0' + '.avi', fourcc, frame_rate, (w, h))
        out01 = cv2.VideoWriter(out_dir + 'tracking01' + '.avi', fourcc, frame_rate, (w, h))
        out02 = cv2.VideoWriter(out_dir + 'tracking02' + '.avi', fourcc, frame_rate, (w, h))
        out2 = cv2.VideoWriter(combined_path, fourcc, frame_rate, (w, h))
        # Combine all videos
        for frame in all_frames:
            out2.write(frame)
        out2.release()

    # Initialize tracking file
    filename = out_dir + '/tracking.txt'
    open(filename, 'w')

    fps = 0.0
    frame_cnt = 0
    t1 = time.time()

    track_cnt = dict()
    images_by_id = dict()
    ids_per_frame = []
    for frame in all_frames:
        #print("frame",frame.shape)
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        image0 = transform(image)
        # add a batch dimension
        image0 = image0.unsqueeze(0).to(device)
        masks, boxs, labels = get_outputs(image0, model, threshold)
        #print(len(masks.shape))
        if len(masks.shape) >2:
           features,boxs = encoder(frame.copy(), masks.copy(),boxs,labels)  # n * 128
        else:
           boxs=[]
           print(masks.shape)

        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]  # length = n
        text_scale, text_thickness, line_thickness = get_FrameLabels(frame)

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.delete_overlap_box(boxes, nms_max_overlap, scores)
        # indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]  # length = len(indices)

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        tmp_ids = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
            if bbox[0] >= 0 and bbox[1] >= 0 and bbox[3] < h and bbox[2] < w:
                tmp_ids.append(track.track_id)
                if track.track_id not in track_cnt:
                    track_cnt[track.track_id] = [
                        [frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area]
                    ]
                    images_by_id[track.track_id] = [frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]]
                else:
                    track_cnt[track.track_id].append([
                        frame_cnt,
                        int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                        area
                    ])
                    images_by_id[track.track_id].append(frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
            cv2_addBox(
                track.track_id,
                frame,
                int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                line_thickness,
                text_thickness,
                text_scale
            )
            write_results(
                filename,
                'mot',
                frame_cnt + 1,
                str(track.track_id),
                int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                w, h
            )
        ids_per_frame.append(set(tmp_ids))

        # save a frame
        if is_vis:
            out.write(frame)
        t2 = time.time()

        frame_cnt += 1
        print(frame_cnt, '/', frame_nums)

    if is_vis:
        out.release()
    print('Tracking finished in {} seconds'.format(int(time.time() - t1)))
    print('Tracked video : {}'.format(tracking_path))
    print('Combined video : {}'.format(combined_path))

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    reid = REID()
    threshold = 320
    exist_ids = set()
    final_fuse_id = dict()

    print(f'Total IDs = {len(images_by_id)}')
    feats = dict()
    for i in images_by_id:
        print(f'ID number {i} -> Number of frames {len(images_by_id[i])}')
        feats[i] = reid._features(images_by_id[i])  # reid._features(images_by_id[i][:min(len(images_by_id[i]),100)])

    for f in ids_per_frame:
        if f:
            if len(exist_ids) == 0:
                for i in f:
                    final_fuse_id[i] = [i]
                exist_ids = exist_ids or f
            else:
                new_ids = f - exist_ids
                for nid in new_ids:
                    dis = []
                    if len(images_by_id[nid]) < 10:
                        exist_ids.add(nid)
                        continue
                    unpickable = []
                    for i in f:
                        for key, item in final_fuse_id.items():
                            if i in item:
                                unpickable += final_fuse_id[key]
                    print('exist_ids {} unpickable {}'.format(exist_ids, unpickable))
                    for oid in (exist_ids - set(unpickable)) & set(final_fuse_id.keys()):
                        tmp = np.mean(reid.compute_distance(feats[nid], feats[oid]))
                        print('nid {}, oid {}, tmp {}'.format(nid, oid, tmp))
                        dis.append([oid, tmp])
                    exist_ids.add(nid)
                    if not dis:
                        final_fuse_id[nid] = [nid]
                        continue
                    dis.sort(key=operator.itemgetter(1))
                    if dis[0][1] < threshold:
                        combined_id = dis[0][0]
                        images_by_id[combined_id] += images_by_id[nid]
                        final_fuse_id[combined_id].append(nid)
                    else:
                        final_fuse_id[nid] = [nid] 
    print('Final ids and their sub-ids:', final_fuse_id)
    print('MOT took {} seconds'.format(int(time.time() - t1)))
    t2 = time.time()

    # To generate MOT for each person, declare 'is_vis' to True
    is_vis = False
    if is_vis:
        print('Writing videos for each ID...')
        output_dir = 'videos/output/tracklets/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        loadvideo = LoadVideo(combined_path)
        video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        for idx in final_fuse_id:
            tracking_path = os.path.join(output_dir, str(idx)+'.avi')
            out = cv2.VideoWriter(tracking_path, fourcc, frame_rate, (w, h))
            for i in final_fuse_id[idx]:
                for f in track_cnt[i]:
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, f[0])
                    _, frame = video_capture.read()
                    text_scale, text_thickness, line_thickness = get_FrameLabels(frame)
                    cv2_addBox(idx, frame, f[1], f[2], f[3], f[4], line_thickness, text_thickness, text_scale)
                    out.write(frame)
            out.release()
        video_capture.release()

    # Generate a single video with complete MOT/ReID
    if args.all:
        loadvideo = LoadVideo(combined_path)
        video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        complete_path = out_dir+'/Complete'+'.avi'
        out = cv2.VideoWriter(complete_path, fourcc, frame_rate, (w, h))
        
        for frame in range(len(all_frames)):
            frame2 = all_frames[frame]
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            _, frame2 = video_capture.read()
            #frame2 = rescale_frame(frame2.copy(), 53)
            for idx in final_fuse_id:
                for i in final_fuse_id[idx]:
                    for f in track_cnt[i]:
                        # print('frame {} f0 {}'.format(frame,f[0]))
                        if frame == f[0]:
                            text_scale, text_thickness, line_thickness = get_FrameLabels(frame2)
                            cv2_addBox(idx, frame2, f[1], f[2], f[3], f[4], line_thickness, text_thickness, text_scale)
            out.write(frame2)
        out.release()
        video_capture.release()

    print('\nWriting videos took {} seconds'.format(int(time.time() - t2)))
    print('Final video at {}'.format(complete_path))
    print('Total: {} seconds'.format(int(time.time() - t1)))


def get_FrameLabels(frame):
    text_scale = max(1, frame.shape[1] / 1600.)
    text_thickness = 2 if text_scale > 2 else 2
    line_thickness = max(1, int(frame.shape[1] / 500.))
    return text_scale, text_thickness, line_thickness


def cv2_addBox(track_id, frame, x1, y1, x2, y2, line_thickness, text_thickness, text_scale):
    color = get_color(abs(track_id))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=line_thickness)
    cv2.putText(
        frame, str(track_id), (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 0), thickness=text_thickness)


def write_results(filename, data_type, w_frame_id, w_track_id, w_x1, w_y1, w_x2, w_y2, w_wid, w_hgt):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{x2},{y2},{w},{h}\n'
    else:
        raise ValueError(data_type)
    with open(filename, 'a') as f:
        line = save_format.format(frame=w_frame_id, id=w_track_id, x1=w_x1, y1=w_y1, x2=w_x2, y2=w_y2, w=w_wid, h=w_hgt)
        f.write(line)
    # print('save results to {}'.format(filename))


warnings.filterwarnings('ignore')


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


if __name__ == '__main__':
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    main('no')
