import sys
import os
path = os.getcwd()
parent = os.path.dirname(path)
sys.path.insert(0, parent)

import cv2
import numpy as np
from typing import Union
import torch
from ObjectDetection.label_dict import label_dict

from ultralytics import YOLO

class ObjectDetection():
    def __init__(self, objects: Union[str, list], model: str="yolov8m.pt"):
        self.bboxes = None
        self.objects = self.extractObjects(objects=objects)
        
        print(f"Loading model: {model}")
        self.device = self.getDeviceType()
        self.model = YOLO(model)
    
    def extractObjects(self, objects):
        _objects = {}
        if type(objects) == str:
            _objects[label_dict[objects]] = None
        elif type(objects) == list:
            for obj in objects:
                _objects[label_dict[obj]] = None
        return _objects
    
    def getDeviceType(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        return device

    def displayBoxes(self, frame, bboxes: Union[list,None]=None):
        if bboxes!=None:
            self.bboxes = bboxes
        for bbox in self.bboxes:
            (x, y, x2, y2) = bbox
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
        cv2.imshow("Img", frame)
        return frame
    
    def getObjectBoxes(self, frame, display=False):
        _objs = []
        result = self.model(frame, device=self.device)[0]
        self.bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        classes = np.array(result.boxes.cls.cpu(), dtype="int")
        for cls, bbox in zip(classes, self.bboxes):
            if cls in self.objects:
                _objs.append(tuple(bbox))
        if display:
            self.displayBoxes(frame)
    
        return _objs
