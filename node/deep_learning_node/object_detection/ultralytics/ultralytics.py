#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

import cv2
import numpy as np
from ultralytics import YOLO

from logging import getLogger

class YOLO_Detect(object):
    def __init__(
        self,
        model_path='yolov8n.onnx',
        class_score_th=0.0,
        nms_th=0.45,
        nms_score_th=0.1,
        with_p6=False,
        providers=[
            # ('TensorrtExecutionProvider', {
            #     'trt_engine_cache_enable': True,
            #     'trt_engine_cache_path': '.',
            #     'trt_fp16_enable': True,
            # }),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        logger = getLogger('ultralytics')
        logger.disabled = True

        # 閾値
        self.class_score_th = class_score_th
        self.nms_th = nms_th

        # モデル読み込み
        self.model = YOLO(model_path)

        # 各種設定
        self.device = None
        if providers == 'CUDAExecutionProvider':
            self.device = 'cuda:0'

    def __call__(self, image):
        results = self.model.predict(image, iou=self.nms_th, show=False)
        return results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    # Load model
    model_path = 'model/yolov8n.onnx'
    model = YOLO(model_path)

    while True:
        # Capture read
        ret, frame = cap.read()
        if not ret:
            break

        # Inference execution
        results = model(frame)

        # Draw
        frame = results[0].plot()

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        cv2.imshow('YOLO', frame)

    cap.release()
    cv2.destroyAllWindows()
