#!/usr/bin/env python
import copy

import cv2 as cv
from ultralytics import YOLO

from logging import getLogger

class YOLO_Pose(object):

    def __init__(
        self,
        model_path,
        providers=None,
    ):
        logger = getLogger('ultralytics')
        logger.disabled = True

        # モデル読み込み
        self.model = YOLO(model_path)
        
    def __call__(self, image):
        results = self.model.predict(image, show=False)
        return results

if __name__ == '__main__':
    cap = cv.VideoCapture(0)

    # Load model
    model = YOLO(None)

    while True:
        # Capture read
        ret, frame = cap.read()
        if not ret:
            break

        # Inference execution
        results = model(frame)

        # Draw
        frame = results[0].plot(boxes=False)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('YOLO Pose', frame)
    cap.release()
    cv.destroyAllWindows()
