#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import pandas as pd

import os
import glob
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
from shutil import rmtree

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')


def doOverlap(l1, r1, l2, r2):

  # If one rectangle is on left side of other
  if(l1[0] > r2[0] or l2[0] > r1[0]):
    return False

  # If one rectangle is above other
  if(l2[1] > r1[1] or l1[1] > r2[1]):
    return False

  return True


def main(yolo):
  max_cosine_distance = 0.3
  nn_budget = None
  nms_max_overlap = 1.0

  # deep_sort
  model_filename = 'model_data/mars-small128.pb'
  encoder = gdet.create_box_encoder(model_filename, batch_size=1)

  metric = nn_matching.NearestNeighborDistanceMetric(
      "cosine", max_cosine_distance, nn_budget)
  tracker = Tracker(metric)

  writeVideo_flag = False
  _fps = 10
  width = 320
  height = 240
  count = 0
  for video_location in glob.glob("../input/*.m4v"):
    # Definition of the parameters
    video_name = os.path.basename(video_location)
    output_dir = os.path.join('../output', video_name)

    if os.path.exists(output_dir):
      rmtree(output_dir)

    output_dir_raw = os.path.join(output_dir, 'raw')
    output_dir_annotated = os.path.join(output_dir, 'annotated')
    if not os.path.exists(output_dir_raw):
      os.makedirs(output_dir_raw)
    if not os.path.exists(output_dir_annotated):
      os.makedirs(output_dir_annotated)

    video_capture = cv2.VideoCapture(video_location)
    video_capture.set(cv2.CAP_PROP_FPS, _fps)
    video_capture.set(3, width)
    video_capture.set(4, height)

    timestamps = [video_capture.get(cv2.CAP_PROP_POS_MSEC)]
    calc_timestamps = [0.0]
    #list_file = open(os.path.join(output_dir, 'metadata.txt'), 'w')
    dfObj = pd.DataFrame(
        columns=['FrameId', 'PersonId', 'x1', 'y1', 'x2', 'y2'])

    frame_index = -1

    if writeVideo_flag:
      # Define the codec and create VideoWriter object
      w = int(video_capture.get(3))
      h = int(video_capture.get(4))
      fourcc = cv2.VideoWriter_fourcc(*'MJPG')
      out = cv2.VideoWriter(os.path.join(output_dir), fourcc, 15, (w, h))

    fps = 0.0
    while True:
      ret, frame = video_capture.read()  # frame shape 320*240*3
      if ret != True:
        break
      t1 = time.time()

      timestamps.append(video_capture.get(cv2.CAP_PROP_POS_MSEC))
      calc_timestamps.append(int(calc_timestamps[-1] + 1000/_fps))
      cv2.imwrite(os.path.join(output_dir_raw,
                               str(calc_timestamps[-1]) + '.jpg'), frame)
    # image = Image.fromarray(frame)
      image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
      boxs = yolo.detect_image(image)
    # print("box_num",len(boxs))
      features = encoder(frame, boxs)

      # score to 1.0 here).
      detections = [Detection(bbox, 1.0, feature)
                    for bbox, feature in zip(boxs, features)]

      # Run non-maxima suppression.
      boxes = np.array([d.tlwh for d in detections])
      scores = np.array([d.confidence for d in detections])
      indices = preprocessing.non_max_suppression(
          boxes, nms_max_overlap, scores)
      detections = [detections[i] for i in indices]

      # Call the tracker
      tracker.predict()
      tracker.update(detections)

      # to find the boxes
      occupied_boxes = []
      final_box = []
      for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
          continue
        area = [0]*len(detections)
        _t = track.to_tlbr()
        if len(detections) != 0:
          for i in range(0, len(detections)):
            _boxs = detections[i].to_tlbr()
            if not doOverlap((_boxs[0], _boxs[1]), (_boxs[2], _boxs[3]),
                             (_t[0], _t[1]), (_t[2], _t[3])):
              continue
            y_axis = sorted([_t[1], _t[3], _boxs[1], _boxs[3]])
            x_axis = sorted([_t[0], _t[2], _boxs[0], _boxs[2]])
            if i not in occupied_boxes:
              area[i] = abs((y_axis[2] - y_axis[1])*(x_axis[2] - x_axis[1]))
          person = - \
              1 if area.index(max(area)) == 0 and max(
                  area) == 0 else area.index(max(area))
          if person > -1:
            occupied_boxes.append(person)
            _d = detections[person].to_tlbr()
            final_box.append((track.track_id, _d))

          # list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
      for p in final_box:
        cv2.putText(frame, '________' +
                    str(p[0]), (int(p[1][0]), int(p[1][1])), 0, 5e-3 * 200, (0, 255, 0), 2)
        dfObj = dfObj.append({
            'FrameId': calc_timestamps[-1],
            'PersonId': p[0],
            'x1': int(p[1][0]),
            'y1': int(p[1][1]),
            'x2': int(p[1][2]),
            'y2': int(p[1][3])
        }, ignore_index=True)

        #list_file.write(str(calc_timestamps[-1]) + '\t' + str(p[0]) + '\t' + str(int(p[1][0])), str(int(p[1][1])) + '\t' + str(int(p[1][2])) + '\t' + str(int(p[1][3])))

      # list_file.write('\n')
      dfObj.to_csv(os.path.join(output_dir, 'metadata.txt'))
      # validation
      if True:
        for track in tracker.tracks:
          if not track.is_confirmed() or track.time_since_update > 1:
            continue
          bbox = track.to_tlbr()
          cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(
              bbox[2]), int(bbox[3])), (255, 255, 255), 2)
          cv2.putText(frame, str(track.track_id), (int(
              bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

        for det in detections:
          bbox = det.to_tlbr()
          cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(
              bbox[2]), int(bbox[3])), (255, 0, 0), 2)

        # cv2.imshow('', frame)
        # save frame as JPEG file
        cv2.imwrite(os.path.join(output_dir_annotated,
                                 str(calc_timestamps[-1]) + '.jpg'), frame)

      # save a frame
      # out.write(frame)
      # frame_index = frame_index + 1
      # list_file.write(str(frame_index)+' ')
      # if len(boxs) != 0:
      #   for i in range(0, len(boxs)):
      #     list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) +
      #                     ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
      # list_file.write('\n')

      # list_file.flush()
      # os.fsync(list_file.fileno())

      if count % 200 == 0:
        fps = (fps + (1./(time.time()-t1))) / 2
        print("fps= %f" % (fps))
      count += 1
      # Press Q to stop!
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    video_capture.release()
    if writeVideo_flag:
      out.release()

    # list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
  main(YOLO())
